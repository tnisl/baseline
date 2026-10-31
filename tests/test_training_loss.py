"""Actual CPU autograd and PEFT/Trainer smoke tests; no Qwen4B weights needed."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

from helpers import BASE

AVAILABLE=all(importlib.util.find_spec(x) for x in ('torch','transformers','peft'))


@unittest.skipUnless(AVAILABLE,'Install torch/transformers/peft for training smoke tests')
class TrainingLossTests(unittest.TestCase):
    def test_ranking_pushes_rejected_token_down_instead_of_teaching_it(self):
        import torch
        from types import SimpleNamespace
        from lifemate_v8_finetune.training.train import local_loss
        torch.set_num_threads(2)
        class BiasModel(torch.nn.Module):
            def __init__(self):super().__init__();self.bias=torch.nn.Parameter(torch.zeros(32))
            def forward(self,input_ids,attention_mask=None,use_cache=False,logits_to_keep=None):
                return SimpleNamespace(logits=self.bias[None,None,:].expand(input_ids.shape[0],len(logits_to_keep),-1))
        def row(target):return {'input_ids':torch.tensor([[1,2,target]]),'attention_mask':torch.ones((1,3),dtype=torch.long),
                                'labels':torch.tensor([[-100,-100,target]])}
        model=BiasModel();inputs={'chosen':row(10),'rejected':row(11),'paired':torch.tensor([True])}
        loss,_=local_loss(model,inputs,negative_weight=0);loss.backward();baseline=float(model.bias.grad[11])
        model.zero_grad();loss,_=local_loss(model,inputs,negative_weight=.5);loss.backward()
        self.assertLess(float(model.bias.grad[10]),0)
        self.assertGreater(float(model.bias.grad[11]),baseline)
        self.assertTrue(torch.isfinite(loss))

    def test_actual_tiny_qwen3_lora_trainer_updates_only_adapters_and_saves(self):
        import torch
        from transformers import Qwen3Config,Qwen3ForCausalLM,TrainingArguments
        from peft import LoraConfig,get_peft_model
        from lifemate_v8_finetune.training.encoding import PairCollator
        from lifemate_v8_finetune.training.train import trainer_class
        torch.set_num_threads(2);torch.manual_seed(42)
        config=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
                           num_attention_heads=4,num_key_value_heads=2,head_dim=8,pad_token_id=0,eos_token_id=3)
        model=get_peft_model(Qwen3ForCausalLM(config),LoraConfig(r=2,lora_alpha=4,
            target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM',lora_dropout=0))
        model.config.use_cache=False
        model.enable_input_require_grads()
        def row(token):return {'input_ids':[1,2,token,3],'attention_mask':[1]*4,'labels':[-100,-100,token,3]}
        items=[{'chosen':row(10),'rejected':row(11),'case_id':'synthetic'}]*4
        before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        with tempfile.TemporaryDirectory() as d:
            args=TrainingArguments(output_dir=d,max_steps=2,per_device_train_batch_size=1,
                gradient_accumulation_steps=2,learning_rate=.01,use_cpu=True,report_to='none',
                gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
                save_strategy='no',eval_strategy='no',disable_tqdm=True,remove_unused_columns=False,
                label_names=[],prediction_loss_only=True)
            trainer=trainer_class()(model=model,args=args,train_dataset=items,eval_dataset=items,
                                    data_collator=PairCollator(0),negative_weight=.2)
            trainer.train();metrics=trainer.evaluate();trainer.save_model(d)
            self.assertTrue((Path(d)/'adapter_model.safetensors').is_file())
            self.assertEqual(metrics['eval_pairs'],4)
            self.assertIn('eval_preference_accuracy',metrics)
        self.assertTrue(any(not torch.equal(before[n],p.detach()) for n,p in model.named_parameters() if n in before))
        self.assertTrue(all(p.grad is None for n,p in model.named_parameters() if 'lora_' not in n))
        self.assertTrue(all('lora_' in n for n,p in model.named_parameters() if p.requires_grad))


if __name__=='__main__':unittest.main()
