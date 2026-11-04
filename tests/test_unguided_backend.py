"""Exercise the Transformers sampler with no CUDA/model downloads."""
import contextlib,sys,types,unittest
from unittest.mock import patch
from umilife.models import Backend

class Tokens:
    shape=(1,3)
    def to(self,device):return self
    def __getitem__(self,key):return self

class UnguidedBackend(unittest.TestCase):
    def test_adapter_generation_has_no_token_enforcer(self):
        backend=Backend.__new__(Backend)
        backend.roles={'planner'};backend.max_context=8192;backend.max_new_tokens=512
        captured={}
        def generate(**kwargs):captured.update(kwargs);return Tokens()
        backend.model=types.SimpleNamespace(device='cpu',set_adapter=lambda role:None,generate=generate)
        class Tokenizer:
            pad_token_id=0;eos_token_id=1
            def apply_chat_template(self,*args,**kwargs):return 'prompt'
            def __call__(self,*args,**kwargs):return {'input_ids':Tokens()}
            def batch_decode(self,*args,**kwargs):return ['{"thought":"Done","next":"summarizer"}']
        backend.tokenizer=Tokenizer()
        with patch.dict(sys.modules,{'torch':types.SimpleNamespace(inference_mode=contextlib.nullcontext),
            'transformers':types.SimpleNamespace(StoppingCriteria=object,StoppingCriteriaList=list)}):
            result=backend.generate_batch('planner',[[{'role':'user','content':'request'}]])
        self.assertEqual(len(result),1)
        self.assertNotIn('prefix_allowed_tokens_fn',captured)
        self.assertNotIn('logits_processor',captured)
        self.assertFalse(captured['do_sample'])

if __name__=='__main__':unittest.main()
