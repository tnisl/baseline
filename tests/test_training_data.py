"""Data correctness: no split leakage, no rejected targets, actual env negatives."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy

from helpers import ROOT, sample
from lifemate_v8_finetune.training.data import Builder,build_dataset,split_ids,prompt_hashes
from lifemate_v8_finetune.pipeline import check_summary,check_call


class TrainingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.builder=Builder(ROOT)

    def test_teacher_prompts_follow_runtime_order_and_do_not_include_labels(self):
        examples,_=self.builder.build_case(sample())
        p=next(x for x in examples if x['role']=='planner')
        payload=json.loads(p['messages'][1]['content'])
        self.assertNotIn('env_feedback',payload)
        self.assertNotIn('label',payload)
        self.assertNotIn('final_state',payload)
        calls=[x for x in examples if x['source']=='gold_call']
        self.assertEqual(json.loads(calls[0]['messages'][1]['content'])['env_feedback'],[])
        self.assertEqual(json.loads(p['target'])['tools'],[a['name'] for a in sample()['label']['tool_calls']])
        for row in examples:
            obj=json.loads(row['messages'][1]['content'])
            self.assertNotIn('label',obj)
            self.assertNotIn('gold_tool_calls',obj)
            self.assertNotIn('final_state',obj)
            self.assertNotIn('metadata',obj)
            self.assertTrue(all(r!=row['target'] for r in row['rejected']))
            if row['role']=='summarizer':
                self.assertEqual(row['loss_scope'],'next')
                check_summary(json.loads(row['target']),obj['step_status'],obj['remaining_plan_steps'])

    def test_gold_ask_becomes_current_business_slot_and_terminal_wait(self):
        examples,_=self.builder.build_case(sample('train_0135'))
        p=next(x for x in examples if x['role']=='planner')
        c=next(x for x in examples if x['role']=='caller')
        s=next(x for x in examples if x['role']=='summarizer')
        self.assertEqual(json.loads(p['target']),{'tools':['alarm.create']})
        self.assertEqual(json.loads(c['target']),{'type':'ask','params':['time']})
        self.assertEqual(json.loads(s['target'])['next'],'wait')
        self.assertEqual(json.loads(s['messages'][1]['content'])['step_status'],'waiting_for_user')
        self.assertEqual(s['rejected'],[])

    def test_empty_gold_does_not_invent_done_or_give_up_training_labels(self):
        examples,flags=self.builder.build_case(sample('train_0458'))
        self.assertEqual([x['role'] for x in examples],['planner'])
        self.assertEqual(json.loads(examples[0]['target']),{'tools':[]})
        self.assertTrue(flags['empty_gold_summary_unlabeled'])

    def test_wrong_effect_and_incomplete_plan_train_give_up_not_the_wrong_action(self):
        examples,_=self.builder.build_case(sample())
        negatives=[x for x in examples if x['source'] in ('counterfactual_wrong_effect','counterfactual_incomplete_plan')]
        self.assertTrue(negatives)
        for x in negatives:
            self.assertEqual(x['role'],'summarizer')
            self.assertEqual(json.loads(x['target'])['next'],'give_up')
            self.assertIn('done',[json.loads(v)['next'] for v in x['rejected']])
            self.assertEqual(json.loads(x['messages'][1]['content'])['remaining_plan_steps'],0)

    def test_recovery_examples_only_have_real_error_feedback_and_verified_targets(self):
        found=[]
        for sid in ('train_0015','train_0236','train_0075','train_0169','train_0486'):
            examples,_=self.builder.build_case(sample(sid))
            found.extend(x for x in examples if x['source']=='counterfactual_recovery')
        self.assertTrue(found)
        for x in found:
            p=json.loads(x['messages'][1]['content'])
            self.assertEqual(p['previous_review'],{'next':'retry'})
            last=p['env_feedback'][-1]
            self.assertTrue(last['response'].get('ok') is False or last['response'].get('done') is False)
            self.assertEqual(json.loads(x['target'])['type'],'call')
            self.assertTrue(x['rejected'])

    def test_split_is_grouped_before_augmentation_with_holdout_protected(self):
        rows=[{'id':str(i)} for i in range(30)]
        a=split_ids(rows,.2,42,['0','1'])
        self.assertEqual(a,split_ids(rows,.2,42,['0','1']))
        self.assertEqual(a['0'],'holdout')
        self.assertEqual(a['1'],'holdout')
        train={x for x,s in a.items() if s=='train'}
        valid={x for x,s in a.items() if s=='validation'}
        self.assertFalse(train & valid)
        self.assertEqual(len(valid),6)

    def test_smoke_dataset_files_and_manifest_have_no_cross_split_case(self):
        with tempfile.TemporaryDirectory() as d:
            manifest=build_dataset(ROOT,d,limit=8,holdout_ids=['train_0236'])
            case_sets={}
            for split in ('train','validation'):
                case_sets[split]=set()
                for role in ('planner','caller','summarizer'):
                    name=f'{role}.{split}.jsonl'
                    self.assertEqual(manifest['files_sha256'][name],hashlib.sha256((Path(d)/name).read_bytes()).hexdigest())
                    for line in (Path(d)/f'{role}.{split}.jsonl').read_text().splitlines():
                        r=json.loads(line)
                        self.assertEqual(r['split'],split)
                        self.assertNotEqual(r['case_id'],'train_0236')
                        case_sets[split].add(r['case_id'])
            self.assertFalse(case_sets['train'] & case_sets['validation'])
            self.assertEqual(manifest['prompt_sha256'],prompt_hashes())
            self.assertEqual(json.loads((Path(d)/'holdout_ids.json').read_text()),['train_0236'])


if __name__=='__main__':unittest.main()
