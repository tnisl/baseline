"""CPU checks for label isolation, scorer denominators, and atomic batch resume."""
import ast
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
ev=module('planner_eval',HERE/'runtime/umilife/eval_planner.py')
policy=module('planner_policy',HERE/'runtime/umilife/prompt_policy.py')

def fixtures():
    keys=[('a',0),('a',1),('b',0),('b',1),('b',2)]
    refs=[('caller','demo.a'),('summarizer',None),('caller','demo.a'),('caller','demo.b'),('summarizer',None)]
    guesses=[('caller','demo.a'),('caller','demo.a'),('caller','demo.b'),None,('summarizer',None)]
    records=[];labels=[];responses=[]
    for (sid,step),(route,tool),guess in zip(keys,refs,guesses):
        target={'thought':'GOLD_TARGET_CANARY','next':route}
        if tool:target['tool']=tool
        records.append({'id':sid,'step':step,'role':'planner',
            'messages':[{'role':'user','content':'Synthetic public request'}],
            'schema':policy.request_schema('planner',{'demo.a':{},'demo.b':{}})})
        labels.append({'id':sid,'step':step,'target':json.dumps(target)})
        prediction={'thought':'Synthetic generated rationale','next':guess[0]} if guess else None
        if guess and guess[1]:prediction['tool']=guess[1]
        responses.append({'raw':json.dumps(prediction) if guess else '{broken',
                          'schema_valid':guess is not None,'finish_reason':'stop'})
    return records,labels,responses

class FakeBackend:
    def __init__(self,responses):self.responses=list(responses);self.position=0;self.batches=[]
    def generate(self,requests):
        self.batches.append(len(requests))
        for r in requests:
            assert set(r)=={'role','messages','schema'}
            assert r['role']=='planner' and 'GOLD_TARGET_CANARY' not in json.dumps(r)
        out=self.responses[self.position:self.position+len(requests)]
        self.position+=len(requests);return out

class Checks(unittest.TestCase):
    def test_metric_denominators_and_label_isolation(self):
        records,labels,responses=fixtures();backend=FakeBackend(responses)
        with tempfile.TemporaryDirectory() as temp:
            predictions,timing=ev.generate_all(records,backend,temp)
            self.assertEqual(backend.batches,[4,1]);self.assertEqual(timing['generated'],5)
            self.assertNotIn('GOLD_TARGET_CANARY',(Path(temp)/'predictions.jsonl').read_text())
            metrics,details=ev.score_outputs(predictions,labels)
            self.assertEqual(metrics['route_accuracy'],3/5)
            self.assertEqual(metrics['route_and_tool_accuracy'],2/5)
            self.assertEqual(metrics['caller_tool_accuracy'],1/3)
            self.assertEqual(metrics['terminal_route_accuracy'],1/2)
            self.assertEqual(metrics['first_route_and_tool_accuracy'],1/2)
            self.assertEqual(metrics['schema_invalid'],1)
            self.assertEqual(metrics['all_reference_steps_correct_episodes'],0)
            # Every record is reused and no inference is needed on a complete resume.
            cached,resume=ev.generate_all(records,None,temp)
            self.assertEqual(cached,predictions);self.assertEqual(resume['reused'],5)
            changed=json.loads(json.dumps(records));changed[0]['messages'][0]['content']='changed'
            with self.assertRaises(AssertionError):ev.generate_all(changed,None,temp)

    def test_pause_before_generation(self):
        records,labels,responses=fixtures();backend=FakeBackend(responses)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(SystemExit) as c:ev.generate_all(records,backend,temp,deadline=0)
            self.assertEqual(c.exception.code,75);self.assertEqual(backend.batches,[])
            self.assertTrue((Path(temp)/'progress.json').exists())
            ev.generate_all(records,backend,temp)
            self.assertFalse((Path(temp)/'progress.json').exists())

    def test_duplicate_or_misaligned_labels_rejected(self):
        records,labels,responses=fixtures()
        outputs=[{'id':r['id'],'step':r['step'],**o} for r,o in zip(records,responses)]
        labels[0]['step']=7
        with self.assertRaises(AssertionError):ev.score_outputs(outputs,labels)

    def test_policy_and_notebook(self):
        self.assertEqual(policy.digest(policy.PROMPTS),'0caf7a12a791a2414a6f7419447c32b1c1c07ffe41376f6a1782ad1e90631f7f')
        notebook=json.loads((HERE/'kaggle_planner_validation_327.ipynb').read_text())
        for c in notebook['cells']:
            if c['cell_type']=='code':ast.parse(c['source'])
        for path in (HERE/'runtime').rglob('*.py'):ast.parse(path.read_text())

    @unittest.skipUnless(os.environ.get('PLANNER_EVAL_INPUTS'),'Optional exact private-data check')
    def test_original_327_inputs_and_last_batch(self):
        folder=Path(os.environ['PLANNER_EVAL_INPUTS'])
        records=ev.readl(folder/'planner_inputs.jsonl')
        labels=ev.readl(folder/'planner_labels.jsonl')
        manifest=json.loads((folder/'evaluation_manifest.json').read_text())
        for name,expected in manifest['files'].items():self.assertEqual(ev.sha(folder/name),expected)
        self.assertEqual(len(records),327)
        self.assertEqual(len({r['id'] for r in records}),100)
        # Dummy responses only: no model executed and no labels passed to generator.
        responses=[{'raw':'{"thought":"Synthetic","next":"summarizer"}',
                    'schema_valid':True,'finish_reason':'stop'} for r in records]
        backend=FakeBackend(responses)
        with tempfile.TemporaryDirectory() as temp:
            outputs,_=ev.generate_all(records,backend,temp)
            self.assertEqual(backend.batches,[4]*81+[3])
            metrics,_=ev.score_outputs(outputs,labels)
            self.assertEqual(metrics['caller_steps'],227)
            self.assertEqual(metrics['terminal_steps'],100)
            self.assertEqual(metrics['route_and_tool_accuracy'],100/327)

if __name__=='__main__':unittest.main()
