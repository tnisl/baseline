"""Build a separate English/TypeScript experiment from the stable two-GPU notebook."""
import ast,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
base=json.loads((HERE.parent/'finetune_2gpu/kaggle_finetune_2gpu.ipynb').read_text())
payload={str(p.relative_to(HERE/'runtime')):p.read_text() for p in sorted((HERE/'runtime').rglob('*')) if p.is_file() and '__pycache__' not in str(p) and p.suffix!='.pyc'}
for name,source in payload.items():
 if name.endswith('.py'):ast.parse(source,filename=name)
sha=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
c=base['cells']
c[0]['source']='''# English prompts + TypeScript contracts: T4 global-to-local fine-tuning

New experiment. Original Qwen3-4B-Instruct-2507, Unsloth QLoRA, two T4 GPUs, batch 4/GPU.
System instructions, runtime feedback and completion review use English. User text, original catalog descriptions, Codex-authored rationale labels and user-facing answers retain Vietnamese.

The linked paper https://arxiv.org/pdf/2402.01030 is CodeAct (executable Python actions). TypeScript interface documentation with JSON actions is our adaptation, not a method claimed by that paper. Global-to-local training remains the earlier alpha-UMi adaptation.

PROFILE="smoke" is the default. Run All performs package/NCCL checks, then 2 optimizer attempts for EACH of global/planner/caller/summarizer, with the longest 16 train and 8 valid records per role, checkpoint restore and lineage checks, then ASR + four held-out workflow evaluations. This smoke is an integration check, not a quality estimate.
PROFILE="full" first requires/runs the same smoke and inference checks, then executes full fine-tuning and all 100 validation workflows. Set PROFILE and RUN_NAME before Run All; completed matching stages can be skipped. Inference/ASR installation is checked before training.

T4 optimizations: compact TypeScript contracts; loss projection chunks enlarged from 32 to configurable 256 supervised tokens; cached tokenization shared by both ranks; reduced object collectives; full validation every 100 steps rather than 50. Full validation loss still uses all records. Eval frequency changes early-stopping responsiveness. Timings are written to history/lineage; speedup is not yet GPU-measured.

No mid-stage optimizer resume. Use a new RUN_NAME after a failed stage. The official split.txt and original rationale hash are preserved. No held-out labels/examples are added to prompts or optimizer data.
'''
c[1]['source']=c[1]['source'].replace("RUN_NAME = 'full_2gpu_chunked_loss_v2'", "PROFILE = 'smoke'  # 'smoke' or 'full'\nRUN_NAME = 'english_ts_t4_v2_anti_repeat'\nLOSS_CHUNK = 256  # 32 is the old conservative fallback\nEVAL_STEPS = 100\nassert PROFILE in ('smoke','full') and LOSS_CHUNK>0 and EVAL_STEPS>0")
c[1]['source']=c[1]['source'].replace("Path('/kaggle/working/paper_finetune_2gpu_source')", "Path('/kaggle/working/paper_en_ts_source')").replace("Path('/kaggle/working/paper_finetune_2gpu')/RUN_NAME", "Path('/kaggle/working/paper_en_ts_t4')/RUN_NAME")
s=c[2]['source'];tree=ast.parse(s);first=tree.body[0];lines=s.splitlines(True);s=''.join(lines[first.end_lineno:]);s='PAYLOAD='+repr(payload)+'\n'+s
s=s.replace(next(line for line in s.splitlines() if line.startswith('RUNTIME_SHA=')),f'RUNTIME_SHA={sha!r}')
s=s.replace("'asr_finetune':False", "'asr_finetune':False,'loss_chunk':LOSS_CHUNK,'eval_steps':EVAL_STEPS")
c[2]['source']=s
# Runtime cell index four after the input discovery cell.
c[4]['source']=c[4]['source'].replace("HF_HOME='/tmp/lifemate_hf_cache'", "LIFEMATE_LOSS_CHUNK=str(LOSS_CHUNK),HF_HOME='/tmp/lifemate_hf_cache'")
# Keep prepared full data separate from adapters for smoke/full stages.
c[6]['source']='''## Smoke before full
The smoke uses original full-length examples, so it tests batch-4 memory on long contexts. A failure stops the notebook before full training. It does not guarantee every full-data batch fits. Full epochs remain global 1, planner 1, caller 1, summarizer 2; patience 3 and cosine warmup 3% remain enabled.
'''
c[7]['source']='''def train_profile(name):
    output=WORK/name
    for stage,epochs in EPOCHS.items():
        target=output/'adapters'/stage
        if (target/'lineage.json').exists():
            meta=json.loads((target/'lineage.json').read_text())
            assert meta['dataset_manifest_sha256']==hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest()
            assert meta['smoke']==(name=='smoke') and meta['loss_chunk']==LOSS_CHUNK
            assert meta['micro_batch']==TRAIN_MICRO_BATCH and meta['eval_batch_per_device']==VALID_LOSS_BATCH
            continue
        args=['--stage',stage,'--data-dir',DATA,'--output-dir',output,'--model',MODEL,'--revision',MODEL_REVISION,
              '--epochs',1 if name=='smoke' else epochs,'--train-batch-size',TRAIN_MICRO_BATCH,
              '--eval-batch-size',VALID_LOSS_BATCH,'--eval-steps',EVAL_STEPS]
        if name=='smoke':args+=['--smoke']
        if stage!='global':args+=['--global-adapter',output/'adapters/global']
        train_module('umilife.train_2gpu',args,distributed=True)
    global_meta=json.loads((output/'adapters/global/lineage.json').read_text())
    for stage in EPOCHS:
        meta=json.loads((output/'adapters'/stage/'lineage.json').read_text())
        assert meta['optimizer_updates']>0 and meta['final_adapter']['finite']
        assert meta['initial_adapter']['sha256']!=meta['final_adapter']['sha256']
        if name=='smoke':assert meta['global_step']==2
        if stage!='global':assert meta['initial_adapter']['sha256']==global_meta['final_adapter']['sha256']
    return output
train_profile('smoke')
'''
c[8]['source']='''## ASR + vLLM validation
Smoke validates the four adapters, fresh guided-decoding contracts, ASR and four held-out episodes. Full mode only starts after the smoke has passed. A successful smoke means the pipeline completed; it does not promise good task scores.
'''
c[9]['source']='''def validate_profile(name,limit):
    output=WORK/name
    (output/'prepared').symlink_to(DATA,target_is_directory=True) if not (output/'prepared').exists() else None
    (output/'split.txt').write_text(SPLIT_FILE.read_text())
    args=[ASR_PYTHON,'-u','-m','umilife.transcribe_supplied','--data-root',str(DATA_ROOT),'--work',str(output),
          '--model-dir',str(ASR_MODEL_DIR),'--threads',str(ASR_THREADS)]
    if limit is not None:args+=['--limit',str(limit)]
    subprocess.run(args,cwd=SOURCE,env=ASR_ENV,check=True)
    args=[INFER_PYTHON,'-u','-m','umilife.infer_2gpu','--data-root',str(DATA_ROOT),'--work',str(output),
          '--model',MODEL,'--revision',MODEL_REVISION,'--adapters',str(output/'adapters'),'--modes','multi']
    if limit is not None:args+=['--limit',str(limit)]
    subprocess.run(args,cwd=SOURCE,env=INFER_ENV,check=True)
    report=json.loads((output/'vllm_runtime.json').read_text())
    assert report['finetuned'] and report['adapter_loaded'] and report['schema_violations']==0
    assert report['samples_per_mode']==(100 if limit is None else limit)
    assert report['schema_enforced'] and not report['asr_report']['oracle_fallback']
    return report
report=validate_profile('smoke',4)
(WORK/'smoke_pass.json').write_text(json.dumps({'runtime_sha':RUNTIME_SHA,'config':RUN_CONFIG,'runtime':report},indent=2))
print('SMOKE PASSED: optimizer updates, checkpoint restore, global-to-local lineage, ASR and vLLM inference')
'''
c[10]['source']='''if PROFILE=='full':
    gate=json.loads((WORK/'smoke_pass.json').read_text());assert gate['runtime_sha']==RUNTIME_SHA and gate['config']==RUN_CONFIG
    train_profile('full')
    report=validate_profile('full',None)
    (WORK/'full_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print('Profile:',PROFILE,'Artifacts:',WORK)
print(json.dumps(report['metrics'],ensure_ascii=False,indent=2))
'''
c[11]['source']+='\nOriginal Codex rationale strings are reused unchanged; English is the instruction language, not a translation of supervised thoughts. Benchmark the new loss chunk and schema lengths on your T4 before making throughput claims.\n'
for i,cell in enumerate(c):
 cell['id']=f'en-ts-t4-{i:02}'
 if cell['cell_type']=='code':ast.parse(cell['source'],filename=cell['id']);cell['outputs']=[];cell['execution_count']=None
meta=json.loads((HERE.parent/'finetune_2gpu/kernel-metadata.json').read_text());meta.update(id='anartt/lifemate-english-typescript-t4',title='LifeMate English TypeScript T4',code_file='kaggle_en_ts_t4.ipynb')
meta['dataset_sources']=[v.replace('cymeu4l0t/lifemate','anartt/lifemate') for v in meta['dataset_sources']]
c[3]['source']=c[3]['source'].replace('cymeu4l0t/lifemate','anartt/lifemate')
(HERE/'kaggle_en_ts_t4.ipynb').write_text(json.dumps(base,ensure_ascii=False,indent=1)+'\n')
(HERE/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
print('Built',sha)
