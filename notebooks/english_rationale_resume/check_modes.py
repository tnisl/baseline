"""CPU checks: actual notebook stage dispatch, base/global initialization and lineage."""
import ast,json,sys,tempfile,unittest,hashlib
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).parent/'runtime/umilife'))
from training_modes import stages_for,parent_for,validate_initialization,verify_parent

class Modes(unittest.TestCase):
 def test_dispatch_and_initialization(self):
  self.assertEqual(stages_for('global'),('global',))
  self.assertEqual(stages_for('local'),('planner','caller','summarizer'))
  self.assertEqual(stages_for('global_to_local'),('global','planner','caller','summarizer'))
  self.assertEqual(stages_for('local',('caller',)),('caller',))
  for role in ('planner','caller','summarizer'):
   self.assertIsNone(parent_for(role,'local','base',None,'/run'))
   self.assertEqual(parent_for(role,'global_to_local','base',None,'/run'),Path('/run/adapters/global'))
   self.assertEqual(parent_for(role,'local','global','/external','/run'),Path('/external'))
   self.assertEqual(validate_initialization(role,'base',None),'base')
  with self.assertRaises(AssertionError):validate_initialization('global','global','/parent')
  with self.assertRaises(AssertionError):validate_initialization('caller','global',None)
  with self.assertRaises(AssertionError):stages_for('local',('global',))

 def test_actual_notebook_runner(self):
  nb=json.loads((Path(__file__).parent/'kaggle_english_rationale_resume.ipynb').read_text())
  source=nb['cells'][-1]['source'];tree=ast.parse(source)
  function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_profile')
  for mode,expected in [('global',['global']),('local',['planner','caller','summarizer']),('global_to_local',['global','planner','caller','summarizer'])]:
   with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);data=root/'prepared';data.mkdir();(data/'manifest.json').write_text('{}')
    digest=hashlib.sha256(b'{}').hexdigest();calls=[]
    def execute(module,args,**kw):
     role=args[args.index('--stage')+1];init=args[args.index('--init-from')+1];calls.append((role,init))
     validate_initialization(role,init,args[args.index('--global-adapter')+1] if '--global-adapter' in args else None)
     target=root/'full/adapters'/role;target.mkdir(parents=True)
     (target/'adapter_model.safetensors').write_bytes(b'fixture')
     parent_hash=hashlib.sha256(b'fixture').hexdigest() if init=='global' else None
     (target/'lineage.json').write_text(json.dumps({'stage':role,'model':'fixture','revision':'rev',
      'dataset_manifest_sha256':digest,'prompt_sha256':'prompt','final_adapter':{'finite':True},'smoke':False,
      'initialization':init,'global_adapter_sha256':parent_hash}))
     return SimpleNamespace(returncode=0)
    import time
    scope=dict(WORK=root,DATA=data,EPOCHS={'global':1,'planner':1,'caller':1,'summarizer':2},
     SELECTED_STAGES=stages_for(mode),TRAIN_MODE=mode,LOCAL_INIT='base',GLOBAL_ADAPTER=None,
     MODEL='fixture',REVISION='rev',m={'prompts_sha256':'prompt'},ENV={},DEADLINE=time.time()+100,
     parent_for=parent_for,verify_parent=verify_parent,execute=execute,json=json,hashlib=hashlib,time=time)
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<notebook runner>','exec'),scope)
    scope['train_profile']('full');self.assertEqual([r for r,i in calls],expected)
    self.assertTrue(all(i==('global' if mode=='global_to_local' and r!='global' else 'base') for r,i in calls))
    scope['train_profile']('full');self.assertEqual(len(calls),len(expected)) # completed stage skip
    if mode=='global_to_local':
     parent=root/'full/adapters/global';record=json.loads((parent/'lineage.json').read_text())
     record['dataset_manifest_sha256']='old-split';(parent/'lineage.json').write_text(json.dumps(record))
     with self.assertRaises(AssertionError):verify_parent(parent,'fixture','rev',digest,'prompt',False)

if __name__=='__main__':unittest.main()
