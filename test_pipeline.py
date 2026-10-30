"""CPU integration tests: real ToolEnv, scripted model, no training labels in input."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_pipeline import Agent, ContractError, check_action, select_fields, parse_object
from common import load_catalog, import_toolenv, evaluate_agent_case
from inference_errors import InferenceBackendError, ContextLimitError

ROOT = Path(__file__).resolve().parent.parent / 'train+public_test'


def sample(sample_id='train_0236'):
    with (ROOT / 'train/train.jsonl').open() as handle:
        return next(json.loads(line) for line in handle if json.loads(line)['id'] == sample_id)


class ScriptedLLM:
    model_id, backend, dtype, temperature, seed = 'scripted', 'test', 'none', 0, 42
    structured_outputs = True

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.inputs = []

    def count(self, messages):
        return 100

    def complete(self, messages, max_new_tokens=512, output_schema=None):
        self.inputs.append((deepcopy(messages), deepcopy(output_schema)))
        output = next(self.outputs)
        if isinstance(output, BaseException):
            raise output
        return {'text': output if isinstance(output, str) else json.dumps(output, ensure_ascii=False),
                'prompt_tokens': self.count(messages), 'generation_tokens': 5, 'latency_s': 0.01}


def call(name):
    return {'type': 'call', 'name': name}


def args(*fields):
    return {'selected_arguments': list(fields)}


def vals(**fields):
    return {'arguments': fields}


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog(ROOT)
        cls.ToolEnv = staticmethod(import_toolenv(ROOT))

    def run_script(self, outputs, **limits):
        llm = ScriptedLLM(outputs)
        row = sample()
        result = Agent(llm, self.catalog, self.ToolEnv, ROOT, **limits).run(row, row['audio_text'])
        return result, llm, row

    def test_real_environment_complete_workflow_and_no_explanations(self):
        result, llm, row = self.run_script([
            call('bill_payment.list_bills'), args(),
            call('ewallet.get_balance'),
            call('bill_payment.pay'), vals(bill_id='bill_01', source='ewallet'),
            call('contacts.search'), args('query'), vals(query='Sếp Dũng'),
            call('sms.send'), args('contact_id'),
            vals(contact_id='c_04', content='đã đóng tiền truyền hình rồi nhé'),
            {'type': 'done'},
        ])
        self.assertEqual(result['stop_reason'], 'done')
        self.assertEqual(evaluate_agent_case(row, result, ROOT, self.catalog, self.ToolEnv)['score_proxy'], 1)
        self.assertEqual(len(llm.inputs), 12)
        self.assertTrue(result['trace'][1]['argument_selector_skipped'])
        self.assertTrue(result['trace'][1]['value_selector_skipped'])
        self.assertTrue(result['trace'][2]['argument_selector_skipped'])
        self.assertNotIn('task_plan', result)
        for messages, _ in llm.inputs:
            payload = json.loads(messages[1]['content'])
            for key in ('label', 'final_state', 'tool_reason', 'value_reasons', 'validation_feedback', 'task_plan'):
                self.assertNotIn(key, payload)
            for entry in payload['env_feedback']:
                self.assertIn(entry['stage'], ('validate', 'call'))
                self.assertEqual(set(entry), {'stage', 'tool', 'arguments', 'response'})

    def test_missing_value_is_environment_error_then_lookup(self):
        result, llm, _ = self.run_script([
            call('sms.send'), args('contact_id'), vals(content='Đã đóng'),
            call('contacts.search'), args('query'), vals(query='Sếp Dũng'), {'type': 'done'},
        ])
        self.assertEqual(result['repair_attempts'], 1)
        self.assertEqual([c['name'] for c in result['tool_calls']], ['contacts.search'])
        next_selector = json.loads(llm.inputs[3][0][1]['content'])
        feedback = next_selector['env_feedback']
        expected = self.ToolEnv('train', sample()['id'], root=ROOT).validate('sms.send', {'content': 'Đã đóng'})
        self.assertEqual(feedback, [{'stage': 'validate', 'tool': 'sms.send',
                                     'arguments': {'content': 'Đã đóng'}, 'response': expected}])

    def test_environment_business_failure_preserved_without_success_rewrite(self):
        result, llm, _ = self.run_script([
            call('bill_payment.pay'), vals(bill_id='does_not_exist', source='ewallet'),
            {'type': 'done'},
        ])
        response = result['trace'][0]['tool_result']
        self.assertEqual(response['ok'], True)
        self.assertEqual(response['done'], False)
        payload = json.loads(llm.inputs[-1][0][1]['content'])
        self.assertEqual(payload['env_feedback'][-1]['response'], response)
        self.assertEqual(result['tool_failures'], 1)

    def test_ask_only_structured_target_and_fields(self):
        result, _, _ = self.run_script([{'type': 'ask', 'tool_name': 'sms.send', 'params': ['contact_id']}])
        self.assertEqual(result['stop_reason'], 'ask_missing_info')
        self.assertEqual(result['tool_calls'], [{'name': 'sys.ask_missing_info',
                                                'arguments': {'tool_name': 'sms.send', 'params': ['contact_id']}}])

    def test_reason_issue_revise_and_unknown_fields_stop_without_feedback_loop(self):
        scripts = [
            [dict(call('contacts.search'), tool_reason='invented feedback')],
            [{'type': 'revise', 'note': 'invented feedback'}],
            [call('sms.send'), {'issue': {'type': 'missing_data', 'reason': 'invented'}}],
            [call('bill_payment.pay'), {'arguments': {}, 'value_reasons': {'bill_id': 'invented'}}],
            [call('ewallet.get_balance'), args('balance')],  # extra response is never requested
        ]
        for script in scripts[:4]:
            with self.subTest(script=script):
                result, _, _ = self.run_script(script)
                self.assertEqual(result['stop_reason'], 'invalid_model_output')
                self.assertEqual(result['tool_calls'], [])
        result, llm, _ = self.run_script([call('contacts.search'), args('balance')])
        self.assertEqual(result['stop_reason'], 'invalid_model_output')
        self.assertEqual(len(llm.inputs), 2)

    def test_repeated_validation_and_reads_stop_with_bounded_calls(self):
        result, _, _ = self.run_script([item for _ in range(3) for item in
                                       [call('sms.send'), args('contact_id'), vals()]], max_repairs=3)
        self.assertEqual(result['stop_reason'], 'validation_no_progress')
        self.assertEqual(result['tool_calls'], [])
        result, _, _ = self.run_script([call('ewallet.get_balance')] * 3)
        self.assertEqual(result['stop_reason'], 'environment_no_progress')
        self.assertEqual(len(result['tool_calls']), 3)

    def test_duplicate_successful_write_stops_before_second_side_effect(self):
        result, _, _ = self.run_script([
            call('sms.send'), args('phone_number'), vals(content='test', phone_number='0900000000'),
            call('sms.send'), args('phone_number'), vals(phone_number='0900000000', content='test'),
        ])
        self.assertEqual(result['stop_reason'], 'duplicate_write')
        self.assertEqual(len(result['tool_calls']), 1)

    def test_grammar_or_context_failures_propagate_not_scored_as_agent_errors(self):
        for error in (InferenceBackendError('backend failed'), ContextLimitError('too long')):
            with self.subTest(error=error), self.assertRaises(type(error)):
                self.run_script([error])

    def test_strict_json_no_prose_duplicates_or_constants(self):
        for text in ('```json\n{}\n```', '{} trailing', '{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}', '[]'):
            with self.subTest(text=text), self.assertRaises((ValueError, ContractError)):
                parse_object(text)

    def test_required_added_and_x_one_of_remains_environment_responsibility(self):
        spec = self.catalog['sms.send']
        self.assertEqual(select_fields(args('contact_id'), spec), ['content', 'contact_id'])
        self.assertEqual(select_fields(args(), spec), ['content'])
        for fields in (('content',), ('contact_id', 'contact_id'), ('returns',)):
            with self.assertRaises(ContractError):
                select_fields(args(*fields), spec)
        for action in ({'type': 'ask', 'tool_name': 'sms.send', 'params': ['not_a_field']},
                       {'type': 'ask', 'tool_name': 'sms.send', 'params': ['content', 'content']}):
            with self.assertRaises(ContractError):
                check_action(action, self.catalog)

    def test_over_budget_prompt_never_sends_gold_or_hidden_state(self):
        class LengthModel(ScriptedLLM):
            def count(self, messages):
                p = json.loads(messages[1]['content'])
                return 100 + 10 * len(p['history']) + 10 * len(p['env_feedback'])
        row = sample()
        row['history'] = [{'role': 'user', 'audio_text': 'history'}] * 3
        row['context']['installed_apps']['leak'] = {'tools': [], 'state': {'secret': 999}}
        llm = LengthModel([])
        agent = Agent(llm, self.catalog, self.ToolEnv, ROOT, max_prompt_tokens=110)
        msgs, trimmed = agent.messages(row, 'current', {}, [
            {'stage': 'call', 'response': {'old': True}}, {'stage': 'call', 'response': {'new': True}}], 2, 'select')
        payload = json.loads(msgs[1]['content'])
        self.assertTrue(trimmed)
        self.assertEqual(payload['env_feedback'], [{'stage': 'call', 'response': {'new': True}}])
        self.assertNotIn('secret', msgs[1]['content'])
        agent.max_prompt_tokens = 1
        with self.assertRaises(ContextLimitError):
            agent.messages(row, 'current', {}, [], 2, 'select')

    def test_runner_writes_v3_reports_without_task_plan(self):
        from run import agent_experiment, workflow_sha256
        row = sample()
        agent = Agent(ScriptedLLM([{'type': 'done'}]), self.catalog, self.ToolEnv, ROOT)
        with tempfile.TemporaryDirectory() as directory:
            agent_experiment([row], {row['id']: row['audio_text']}, ROOT, directory,
                             'workflow_gold', agent, self.catalog, self.ToolEnv)
            dest = Path(directory) / 'workflow_gold'
            data = json.loads((dest / 'traces.jsonl').read_text())
            self.assertNotIn('task_plan', data)
            self.assertEqual(json.loads((dest / 'run_status.json').read_text())['status'], 'complete')
            summary = json.loads((dest / 'summary.json').read_text())
            self.assertEqual(summary['feedback_source'], 'env.validate/env.call')
            self.assertEqual(summary['workflow_sha256'], workflow_sha256())
            self.assertNotIn('selector_issues_per_sample', summary)


if __name__ == '__main__':
    unittest.main()
