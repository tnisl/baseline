import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'runtime/umilife'))
from english_labels import rationale,terminal

class Labels(unittest.TestCase):
 def test_future_independence_and_threshold(self):
  row={'audio_text':'Vàng bán trên 122 triệu thì nhắc tôi.','history':[],'label':{'future':'SECRET'}}
  action={'name':'banking.get_balance','arguments':{}}
  prior=[{'action':{'name':'gold_price.get_price'},'observation':{'prices':[{'sell':121400000}]}}]
  a=rationale(row,action,prior);self.assertIn('below',a);self.assertIn('122000000',a)
  row['label']={'future':'CHANGED'};self.assertEqual(a,rationale(row,action,prior));self.assertNotIn('SECRET',a)
 def test_observed_time_calculation(self):
  row={'audio_text':'Báo trước một tiếng','history':[]}
  prior=[{'action':{'name':'calendar.list_events'},'observation':{'events':[{'start':'2026-11-19T10:30:00+07:00'}]}}]
  a=rationale(row,{'name':'alarm.create','arguments':{'time':'2026-11-19T09:30:00+07:00'}},prior)
  self.assertIn('minus 60 minutes',a)
 def test_zero_action_and_missing(self):
  self.assertEqual(terminal({'audio_text':'Thôi không cần nữa'},[])[0]['next'],'summarizer')
  self.assertEqual(terminal({'audio_text':'Đặt phòng khách sạn'},[])[0]['next'],'give_up')
  a=rationale({'audio_text':'','history':[]},{'name':'sys.ask_missing_info','arguments':{'tool_name':'tv.control','params':['volume']}},[])
  self.assertIn('volume',a);self.assertIn('instead of inventing',a)
if __name__=='__main__':unittest.main()
