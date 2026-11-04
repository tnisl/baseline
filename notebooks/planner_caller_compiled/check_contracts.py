"""CPU checks for role separation and the executable notebook configuration."""
import ast,json,sys,unittest
from pathlib import Path
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H/'runtime'))
from umilife.english_labels import rationale,terminal
from umilife.prompt_policy import build_messages,request_schema,PROMPTS

class Contracts(unittest.TestCase):
 def test_argument_values_not_serialized(self):
  row={'audio_text':'Delete the requested alarm','history':[]}
  a={'name':'alarm.delete','arguments':{'alarm_id':'PRIVATE_ID_123'}}
  prior=[{'action':{'name':'alarm.list'},'observation':{'items':[{'id':'PRIVATE_ID_123'}]}}]
  thought=rationale(row,a,prior)
  self.assertIn('alarm_id from the alarm.list result',thought)
  self.assertNotIn('PRIVATE_ID_123',thought)
  row['label']={'secret':'FUTURE'}
  self.assertEqual(thought,rationale(row,a,prior))
  self.assertLessEqual(len(thought.split()),60)
 def test_history_and_required_information(self):
  row={'audio_text':'Send it','history':[{'role':'user','text':'recipient','tool_results':{'phone':'HISTORY_SECRET'}}]}
  a={'name':'sms.send','arguments':{'phone':'HISTORY_SECRET','content':'MESSAGE_SECRET'}}
  thought=rationale(row,a,[])
  self.assertIn('phone from conversation history',thought)
  self.assertNotIn('HISTORY_SECRET',thought);self.assertNotIn('MESSAGE_SECRET',thought)
  ask=rationale(row,{'name':'sys.ask_missing_info','arguments':{'tool_name':'sms.send','params':['phone']}},[])
  self.assertIn('sms.send: phone',ask)
 def test_caller_input_retains_evidence_and_typescript(self):
  specs={'alarm.delete':{'kind':'write','description':'Delete an alarm','parameters':{'type':'object','properties':{'alarm_id':{'type':'string'}},'required':['alarm_id']}}}
  row={'context':{},'history':[{'role':'user','text':'original reference'}]}
  plan={'thought':'Delete the requested alarm.','next':'caller','tool':'alarm.delete'}
  messages=build_messages('caller',row,'current',specs,[{'observation':{'id':'observed'}}],plan)
  x=json.loads(messages[1]['content'])
  self.assertEqual(x['history'],row['history']);self.assertEqual(x['trajectory'][0]['observation']['id'],'observed')
  self.assertIn('interface InstalledTools',x['selected_tool_ts'])
  self.assertIn('"alarm.delete"(args:',x['selected_tool_ts'])
  self.assertEqual(request_schema('caller',specs,plan)['required'],['name','arguments'])
 def test_notebook_full_only_and_compile(self):
  nb=json.loads((H/'kaggle_planner_caller_compiled.ipynb').read_text())
  code='\n'.join(c['source'] for c in nb['cells'] if c['cell_type']=='code')
  for c in nb['cells']:
   if c['cell_type']=='code':ast.parse(c['source'])
  calls=[n for n in ast.walk(ast.parse(code)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='train_profile']
  self.assertEqual([n.args[0].value for n in calls],['full'])
  self.assertIn("UNSLOTH_COMPILE_DISABLE='0'",code)
  self.assertNotIn("execute('umilife.preflight_2gpu'",code)
  self.assertIn("LOCAL_STAGES=('planner','caller')",code)
if __name__=='__main__':unittest.main()
