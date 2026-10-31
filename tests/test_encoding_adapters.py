"""Loss boundaries and per-request adapter routing, including mixed-role batches."""
import hashlib
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
import tempfile
import unittest

from helpers import BASE
from lifemate_v8_finetune.adapters import ROLE_SYSTEMS,resolve_adapters,role_for_messages
from lifemate_v8_finetune.training.encoding import encode_completion,PairCollator,RoleDataset
from lifemate_v8_finetune.models import QwenVLLM,Qwen


class CharTokenizer:
    def __call__(self,text,add_special_tokens=False):return {'input_ids':[ord(x) for x in text]}
    def apply_chat_template(self,messages,tokenize=True,add_generation_prompt=False):
        text=''.join('<'+m['role']+'>'+m['content']+'</end>' for m in messages)
        if add_generation_prompt:text+='<assistant>'
        return [ord(x) for x in text] if tokenize else text


class EncodingAdapterTests(unittest.TestCase):
    def test_dataset_progress_preserves_encoding_negatives_and_overlength_counts(self):
        messages=[{'role':'user','content':'request'}]
        rows=[{'case_id':'one','messages':messages,'target':'{"arguments":{}}',
               'rejected':['{"type":"skip"}'],'loss_scope':'completion'},
              {'case_id':'two','messages':messages,'target':'{"next":"done","answer":"masked"}',
               'rejected':['{"next":"give_up","answer":"masked"}'],'loss_scope':'next'},
              {'case_id':'long','messages':messages,'target':'x'*1000,
               'rejected':[],'loss_scope':'completion'}]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'role.train.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            quiet=RoleDataset(p,CharTokenizer(),max_length=256)
            output=io.StringIO()
            with redirect_stdout(output):
                visible=RoleDataset(p,CharTokenizer(),max_length=256,progress=True,total=3)
            self.assertEqual(visible.items,quiet.items)
            self.assertEqual(visible.stats,quiet.stats)
            self.assertEqual(visible.stats['overlength'],1)
            self.assertIn('start: 0/3',output.getvalue())
            self.assertIn('complete: 3/3',output.getvalue())
            self.assertIn('overlength=1',output.getvalue())

    def test_completion_masks_prompt_and_decision_masks_entire_answer(self):
        t=CharTokenizer();messages=[{'role':'system','content':'test'},{'role':'user','content':'request'}]
        target='{"next":"done","answer":"This is not an answer training label."}'
        encoded=encode_completion(t,messages,target,'next')
        learned=''.join(chr(x) for x in encoded['labels'] if x!=-100)
        self.assertEqual(learned,'{"next":"done"')
        self.assertNotIn('answer',learned)
        ordinary=encode_completion(t,messages,'{"arguments":{}}','completion')
        self.assertEqual(''.join(chr(x) for x in ordinary['labels'] if x!=-100),'{"arguments":{}}</end>')
        self.assertTrue(all(x==-100 for x in ordinary['labels'][:len(t.apply_chat_template(messages,add_generation_prompt=True))]))

    @unittest.skipUnless(importlib.util.find_spec('torch'),'torch not installed')
    def test_padding_is_never_supervised(self):
        items=[{'chosen':{'input_ids':[1,2],'attention_mask':[1,1],'labels':[-100,2]},'rejected':None},
               {'chosen':{'input_ids':[1,2,3],'attention_mask':[1,1,1],'labels':[-100,2,3]},'rejected':None}]
        batch=PairCollator(0)(items)
        self.assertEqual(batch['chosen']['labels'][0,-1],-100)
        self.assertEqual(batch['chosen']['attention_mask'][0,-1],0)
        self.assertFalse(batch['paired'].any())

    def test_adapter_metadata_prevents_wrong_role_base_or_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            for role,system in ROLE_SYSTEMS.items():
                p=Path(d)/role;p.mkdir()
                (p/'adapter_config.json').write_text(json.dumps({'peft_type':'LORA','r':16}))
                (p/'adapter_model.safetensors').write_bytes(b'file-presence-only-test')
                (p/'role_metadata.json').write_text(json.dumps({'role':role,'base_model':'test-base',
                    'global_finetune':False,'prompt_sha256':hashlib.sha256(system.encode()).hexdigest()}))
            paths=resolve_adapters(d,'test-base')
            self.assertEqual(set(paths),set(ROLE_SYSTEMS))
            with self.assertRaises(ValueError):resolve_adapters(d,'wrong-base')
            p=Path(d)/'caller/role_metadata.json';meta=json.loads(p.read_text());meta['role']='planner';p.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):resolve_adapters(d,'test-base')

    def test_mixed_role_vllm_batch_gets_a_distinct_lora_request_per_prompt(self):
        adapter=QwenVLLM.__new__(QwenVLLM)
        adapter.temperature,adapter.seed,adapter.max_model_len=0,42,64
        adapter.structured_outputs=False
        adapter.encode=lambda messages:[1,2]
        adapter.SamplingParams=lambda **kwargs:kwargs
        adapter.lora_requests={r:r+'-adapter' for r in ROLE_SYSTEMS}
        captured=[]
        def generate(prompts,sampling,use_tqdm=False,**kwargs):
            captured.extend(kwargs['lora_request'])
            return [SimpleNamespace(outputs=[SimpleNamespace(text=x,token_ids=[3])]) for x in kwargs['lora_request']]
        adapter.engine=SimpleNamespace(generate=generate)
        roles=['summarizer','planner','caller','planner']
        requests=[([{'role':'system','content':ROLE_SYSTEMS[r]}],8,None) for r in roles]
        outputs=adapter.complete_batch(requests)
        self.assertEqual(captured,[r+'-adapter' for r in roles])
        self.assertEqual([o['text'] for o in outputs],captured)
        with self.assertRaises(ValueError):role_for_messages([{'role':'system','content':'modified prompt'}])

    def test_transformers_dispatch_selects_role_without_loading_three_bases(self):
        adapter=Qwen.__new__(Qwen)
        adapter._adapter_lock=RLock();adapter.role_adapters={r:r for r in ROLE_SYSTEMS}
        calls=[];adapter.model=SimpleNamespace(set_adapter=lambda role:calls.append(role))
        adapter._complete=lambda messages,limit:{'selected':calls[-1]}
        for role,system in ROLE_SYSTEMS.items():
            self.assertEqual(adapter.complete([{'role':'system','content':system}])['selected'],role)
        self.assertEqual(calls,list(ROLE_SYSTEMS))


if __name__=='__main__':unittest.main()
