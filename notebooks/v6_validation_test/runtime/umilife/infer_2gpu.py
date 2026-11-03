"""Fine-tuned multi-LoRA workflow; held-out labels are never model inputs."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES","0")
os.environ.setdefault("VLLM_USE_V1","0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND","XFORMERS")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD","spawn")
os.environ.setdefault("VLLM_USAGE_STATS_ENABLED","0")
os.environ.setdefault("DO_NOT_TRACK","1")
import json,time,argparse,math
from pathlib import Path
from collections import Counter
from umilife.core import read_jsonl,write_json,dumps,parse_json,allowed_specs,load_catalog,import_toolenv
from umilife.prompt_policy import (PROMPTS,COMPLETION_REVIEW,POLICY_AUDIT,POLICY_VERSION,
    build_messages,request_schema,validate_meta_action,repeated_read,canonical,digest)
from copy import deepcopy
SYSTEMS=PROMPTS
from umilife.data import normalized_row
from umilife.grammar import role_schema

def rollout(row,text,specs,env,mode,max_calls=10,max_decisions=16):
    started=time.monotonic();trajectory=[];trace=[];answer=None;reason="decision_limit"
    feedback=None;failures=0;reviews=0;read_skips=0;repairs=0
    def generate(role,plan=None,feedback=None):
        schema=request_schema(role,specs,plan)
        response=yield {"role":role,"messages":build_messages(role,row,text,specs,trajectory,plan,feedback),"schema":schema}
        trace.append({"role":role,**response})
        if response.get("schema_valid") is False:
            raise ValueError("Model output violates request schema: "+response.get("schema_error",""))
        return parse_json(response["raw"])
    for _ in range(max_decisions):
        try:
            role="global" if mode=="single" else "planner"
            planner_feedback=feedback
            plan=yield from generate(role,feedback=planner_feedback);feedback=None
            trace.append({"role":role,"output":plan})
            if plan["next"] in ("summarizer","give_up"):
                reviews+=1
                trace.append({"event":"completion_review","proposal":plan})
                plan=yield from generate(role,feedback={"kind":"completion_review",'proposal':plan,'instruction':COMPLETION_REVIEW})
                trace.append({"role":role,"output":plan,"reviewed":True})
            route=plan["next"]
            if route=="give_up":reason="give_up";break
            if route=="summarizer":
                answer=plan.get("answer") if mode=="single" else (yield from generate("summarizer",plan)).get("answer")
                reason="complete";break
            if route!="caller" or plan.get("tool") not in specs:raise ValueError("Invalid planner route or unavailable tool")
            if len(env.get_tool_calls())>=max_calls:reason="call_limit";break
            action_feedback=planner_feedback;action=None
            for attempt in range(3):
                try:
                    candidate=plan.get("action") if mode=="single" and attempt==0 else (
                        (yield from generate(role,feedback=action_feedback)).get("action") if mode=="single" else
                        (yield from generate("caller",plan,action_feedback)))
                    if not isinstance(candidate,dict):raise ValueError("Invalid action object")
                    name,args=candidate.get("name"),candidate.get("arguments")
                    if name!=plan['tool']:raise ValueError("Caller must use the tool chosen by Planner")
                    check=env.validate(name,args) if name in specs and isinstance(args,dict) else {"ok":False,"error":"Unknown tool or invalid arguments"}
                    if check.get('ok'):
                        extra=validate_meta_action(candidate,specs)
                        if extra:check={'ok':False,'error':extra}
                    trace.append({"role":"caller","output":candidate,"validation":check})
                    if check.get('ok'):action=candidate;break
                    error=check.get('error','Invalid arguments')
                except (ValueError,KeyError,TypeError) as exc:error=str(exc)
                repairs+=1
                action_feedback={"kind":"tool_contract_error","error":error,
                    "instruction":"Sửa theo schema và dữ kiện đã có; nếu thiếu căn cứ thì cần lập kế hoạch tra cứu hoặc hỏi bổ sung."}
            if action is None:
                failures+=1;feedback=action_feedback
                if failures>=3:reason="format_error";trace.append({"error":"tool contract repair exhausted"});break
                continue
            if repeated_read(action,specs,trajectory):
                read_skips+=1
                feedback={'kind':'repeated_read','action':action,
                    'instruction':'Kết quả lời gọi này đã có trong trajectory và chưa có thao tác ghi làm thay đổi dữ liệu. Dùng kết quả đó; nếu rỗng hãy đổi bộ lọc theo schema hoặc hỏi thiếu thông tin, không lặp nguyên lời gọi.'}
                trace.append({'event':'repeat_read_prevented','action':action});continue
            name,args=action['name'],action['arguments']
            observed=env.call(name,args);failures=0
            trajectory.append({'step':len(trajectory),'thought':plan.get('thought',''),'action':action,'observation':observed})
            if name=='sys.ask_missing_info':
                terminal={'thought':'Đã yêu cầu bổ sung thông tin; chờ người dùng trả lời.','next':'summarizer'}
                answer=(yield from generate('summarizer',terminal)).get('answer')
                reason='needs_user_input';break
        except (ValueError,KeyError,TypeError) as exc:
            failures+=1;repairs+=1;trace.append({'error':str(exc)})
            if failures>=3:reason='format_error';break
            feedback={'kind':'planning_contract_error','error':str(exc),'instruction':'Lập lại đúng một bước theo dữ kiện và schema công cụ đã cài.'}
    return {'id':row['id'],'tool_calls':env.get_tool_calls(),'state_hash':env.get_state_hash()},{
        'id':row['id'],'reason':reason,'elapsed_s':time.monotonic()-started,
        'over_deadline':False,'time_limit_enforced':False,'answer':answer,
        'trace':trace,'trajectory':trajectory,'transcript':text,
        'completion_reviews':reviews,'repeat_reads_prevented':read_skips,'repairs':repairs,'prompt_policy':POLICY_VERSION}

def run_batch(generators,backend):
    active={}
    completed={}
    for index,generator in enumerate(generators):
        try:active[index]=next(generator)
        except StopIteration as stop:completed[index]=stop.value
    while active:
        good=[];bad=[]
        for index,request in active.items():
            try:backend.validate(request);good.append(index)
            except ValueError as exc:bad.append((index,exc))
        for index,error in bad:
            try:active[index]=generators[index].throw(error)
            except StopIteration as stop:completed[index]=stop.value;del active[index]
        if not good:continue
        responses=backend.generate([active[index] for index in good])
        assert len(responses)==len(good)
        for index,response in zip(good,responses,strict=True):
            try:active[index]=generators[index].send(response)
            except StopIteration as stop:completed[index]=stop.value;del active[index]
    return [completed[index] for index in range(len(generators))]

def prepare_t4_lora_kernel():
    """Work around Triton 3.3 fp16 conversion failure on SM75; preserve FP32 accumulation."""
    import torch,importlib.util,importlib.metadata,hashlib
    path=Path(importlib.util.find_spec("vllm").origin).parent/"lora/ops/triton_ops/kernel_utils.py"
    original="        accumulator += tl.dot(\n            tiled_a,\n            tiled_b,\n        )"
    patched="        # LifeMate SM75 compatibility: explicit exact fp16->fp32 before dot.\n        accumulator += tl.dot(tiled_a.to(tl.float32), tiled_b.to(tl.float32), input_precision=\"ieee\")"
    source=path.read_text()
    capability=torch.cuda.get_device_capability(0)
    if capability[0]>=8:
        if patched in source:path.write_text(source.replace(patched,original))
        return {"applied":False,"capability":list(capability)}
    assert importlib.metadata.version("vllm")=="0.9.2"
    assert importlib.metadata.version("triton")=="3.3.0"
    assert original in source or patched in source,"Unexpected LoRA kernel source"
    if original in source:path.write_text(source.replace(original,patched))
    # Exercise the exact patched helper on the GPU against an FP32 PyTorch reference.
    global triton,tl,mm_k
    import triton
    import triton.language as tl
    from vllm.lora.ops.triton_ops.kernel_utils import mm_k
    @triton.jit
    def check_kernel(A,B,C,K:tl.constexpr):
        rows=tl.arange(0,32)
        cols=tl.arange(0,16)
        kk=tl.arange(0,32)
        a=A+rows[:,None]*K+kk[None,:]
        b=B+kk[:,None]*16+cols[None,:]
        acc=mm_k(a,b,1,16,kk,K,32,16,32,True,1,False,tl.float16)
        tl.store(C+rows[:,None]*16+cols[None,:],acc)
    errors=[]
    torch.manual_seed(42)
    for k in (128,2560):
        a=torch.randn(32,k,device="cuda",dtype=torch.float16)
        b=torch.randn(k,16,device="cuda",dtype=torch.float16)
        c=torch.empty(32,16,device="cuda",dtype=torch.float32)
        check_kernel[(1,)](a,b,c,k,num_warps=4)
        ref=a.float()@b.float()
        torch.testing.assert_close(c,ref,rtol=5e-4,atol=2e-3)
        errors.append(float((c-ref).abs().max()))
    return {"applied":True,"capability":list(capability),"operation":"LoRA dot with FP32 accumulation",
            "reference_test":"pass","maximum_errors":errors,"kernel_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}

class Backend:
    def __init__(self,model,revision,adapters,max_context=8192):
        self.t4_compatibility=prepare_t4_lora_kernel()
        print(dumps({"t4_compatibility":self.t4_compatibility}),flush=True)
        from vllm import LLM,SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
        from vllm.lora.request import LoRARequest
        from transformers import AutoConfig,AutoModelForCausalLM
        from safetensors import safe_open
        import torch
        assert torch.cuda.is_available()
        assert torch.cuda.get_device_capability(0)>=(7,0)
        self.max_context=max_context
        self.requests={}
        adapter_parameters=0
        lineage_records={}
        for index,role in enumerate(("global","planner"),start=1):
            path=Path(adapters)/role
            meta=json.loads((path/"lineage.json").read_text())
            assert meta["stage"]==role and meta["model"]==model and meta["revision"]==revision
            assert meta["final_adapter"]["finite"]
            assert meta['prompt_sha256']==digest(PROMPTS),'Training/inference policy mismatch'
            assert meta['world_size']==2 and meta['micro_batch']>0 and meta['eval_batch_per_device']>0
            lineage_records[role]=meta
            self.requests[role]=LoRARequest(role,index,str(path))
            with safe_open(path/"adapter_model.safetensors",framework="pt",device="cpu") as handle:
                adapter_parameters+=sum(math.prod(handle.get_slice(k).get_shape()) for k in handle.keys())
        manifest_hashes={v['dataset_manifest_sha256'] for v in lineage_records.values()}
        assert len(manifest_hashes)==1
        for role in ('planner',):
            assert lineage_records[role]['initial_adapter']['sha256']==lineage_records['global']['final_adapter']['sha256']
        # Missing local roles use the completed global adapter explicitly; no fake local checkpoints.
        self.requests['caller']=self.requests['global'];self.requests['summarizer']=self.requests['global']
        self.dataset_manifest_sha256=next(iter(manifest_hashes))
        config=AutoConfig.from_pretrained(model,revision=revision)
        with torch.device("meta"):meta_model=AutoModelForCausalLM.from_config(config)
        base_parameters=meta_model.num_parameters();del meta_model
        self.parameter_report={"base":base_parameters,"comparison_adapters":adapter_parameters,
                               "total_llm":base_parameters+adapter_parameters}
        assert self.parameter_report["total_llm"]<5_000_000_000
        self.llm=LLM(model=model,revision=revision,tokenizer=str(Path(adapters)/"global"),
            dtype="float16",tensor_parallel_size=1,enable_lora=True,max_loras=4,max_cpu_loras=4,
            max_lora_rank=16,max_model_len=max_context,max_num_seqs=4,max_num_batched_tokens=max_context,
            gpu_memory_utilization=.90,enforce_eager=True,enable_prefix_caching=False,
            enable_chunked_prefill=False,seed=42,trust_remote_code=False,
            generation_config="vllm",disable_log_stats=True)
        self.tokenizer=self.llm.get_tokenizer()
        self.max_tokens={"global":768,"planner":320,"caller":512,"summarizer":384}
        self.batch_sizes=[];self.roles_seen=Counter();self.schema_violations=0
    def make_sampling(self,request):
        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
        # Fresh per request: never share mutable grammar state between batch members.
        return SamplingParams(temperature=0,seed=42,
            max_tokens=64 if request.get("probe") else self.max_tokens[request["role"]],
            guided_decoding=GuidedDecodingParams(
                json=request.get("schema",role_schema(request["role"])),backend="lm-format-enforcer"))
    def validate(self,request):
        role=request["role"]
        prompt=self.tokenizer.apply_chat_template(request["messages"],tokenize=False,add_generation_prompt=True)
        length=len(self.tokenizer.encode(prompt,add_special_tokens=False))
        if length+(64 if request.get("probe") else self.max_tokens[role])>self.max_context:
            raise ValueError("Context budget exceeded; no evidence is truncated")
    def generate(self,requests):
        assert 1<=len(requests)<=4
        for r in requests:self.validate(r)
        self.batch_sizes.append(len(requests));self.roles_seen.update(r["role"] for r in requests)
        prompts=[self.tokenizer.apply_chat_template(r["messages"],tokenize=False,add_generation_prompt=True) for r in requests]
        results=self.llm.generate(prompts,[self.make_sampling(r) for r in requests],
            lora_request=[self.requests[r["role"]] for r in requests],use_tqdm=False)
        from jsonschema import Draft202012Validator,ValidationError
        responses=[]
        for request,result in zip(requests,results,strict=True):
            response={"raw":result.outputs[0].text,"finish_reason":result.outputs[0].finish_reason,
                      "stop_reason":result.outputs[0].stop_reason,
                      "prompt_sha256":digest(request["messages"]),"schema_sha256":digest(request.get("schema",role_schema(request["role"])))}
            try:
                value=parse_json(response["raw"])
                Draft202012Validator(request.get("schema",role_schema(request["role"]))).validate(value)
                response["schema_valid"]=True
            except (ValueError,ValidationError) as exc:
                response.update(schema_valid=False,schema_error=str(exc)[:300])
                self.schema_violations+=1
            responses.append(response)
        return responses
    def preflight(self):
        probes=[]
        for role in ("global","planner","caller","summarizer"):
            schema={"type":"object","properties":{"probe":{"type":"string","enum":[role]}},
                    "required":["probe"],"additionalProperties":False}
            probes.append({"role":role,"probe":True,"schema":schema,"messages":[
                {"role":"user","content":"Just say hello, no JSON."}]})
        for request,result in zip(probes,self.generate(probes),strict=True):
            assert result["schema_valid"] and parse_json(result["raw"])=={"probe":request["role"]},result
        specs={"demo.inspect":{"name":"demo.inspect","kind":"read","description":"Read a synthetic probe record",
            "parameters":{"type":"object","properties":{"token":{"type":"string","enum":["probe"]}},"required":["token"]}}}
        row={"context":{},"history":[]};plan={"thought":"Cần đọc dữ liệu kiểm tra.","next":"caller","tool":"demo.inspect"}
        requests=[]
        for role in ("global","planner","caller","summarizer"):
            trajectory=[] if role!='summarizer' else [{'action':{'name':'demo.inspect','arguments':{'token':'probe'}},'observation':{'ok':True,'value':'probe'}}]
            requests.append({"role":role,"messages":build_messages(role,row,'Đọc bản ghi kiểm tra được cung cấp.',specs,trajectory,plan),
                             'schema':request_schema(role,specs,plan)})
        results=self.generate(requests)
        assert all(r['schema_valid'] for r in results),results
        assert parse_json(results[2]['raw'])=={'name':'demo.inspect','arguments':{'token':'probe'}}
        contracts=[]
        for name,parameters in [
            ('demo.integer',{'type':'object','properties':{'count':{'type':'integer','enum':[2,4]}},'required':['count']}),
            ('demo.alternative',{'type':'object','properties':{'reference':{'type':'string'},'code':{'type':'integer'}},'required':[],'x-one-of':[['reference'],['code']]}),
            ('demo.optional',{'type':'object','properties':{'choice':{'type':'string','enum':['alpha','beta']}},'required':[]}),
            ('demo.flag',{'type':'object','properties':{'enabled':{'type':'boolean'}},'required':['enabled']})]:
            item={name:{'name':name,'kind':'read','description':'Synthetic tool contract probe','parameters':parameters}}
            probe_plan={'thought':'Kiểm tra schema giả lập.','next':'caller','tool':name}
            contracts.append({'role':'caller','schema':request_schema('caller',item,probe_plan),
                'messages':build_messages('caller',row,'Thực hiện kiểm tra schema công cụ giả lập.',item,[],probe_plan)})
        results=self.generate(contracts)
        assert all(r['schema_valid'] for r in results),results
        return {'real_vllm_batch_size':4,'tested_prompt_roles':list(self.requests),'json_contracts':'pass',
            'adversarial_schema_probe':'pass','selected_tool_and_argument_schema_probes':'pass',
            'probe_source':'synthetic fixtures only; no benchmark examples'}

def main():
    p=argparse.ArgumentParser()
    for name in ('data-root','work','model','revision','adapters','ids-file'):p.add_argument('--'+name,required=True)
    p.add_argument('--deadline-monotonic',type=float)
    p.add_argument('--split',choices=['validation','public_test'],required=True)
    p.add_argument('--mode',choices=['single','multi_fallback'],default='multi_fallback')
    a=p.parse_args();root=Path(a.data_root);work=Path(a.work);work.mkdir(parents=True,exist_ok=True)
    split='train' if a.split=='validation' else 'public_test'
    ids=Path(a.ids_file).read_text().splitlines();assert len(ids)==len(set(ids)) and ids
    source=read_jsonl(root/split/f'{split}.jsonl');lookup={r['id']:r for r in source};assert set(ids)<=lookup.keys()
    if a.split=='public_test':assert set(ids)==set(lookup),'Test must include every public-test ID'
    rows=[]
    for sid in ids:
        raw=lookup[sid]
        context={k:deepcopy(raw['context'][k]) for k in ('current_time','timezone','location','user_profile') if k in raw['context']}
        context['installed_apps']={app:{'tools':list(meta.get('tools',[]))} for app,meta in raw['context']['installed_apps'].items()}
        history=[]
        for turn in raw.get('history',[]):
            keys=('role','audio') if turn['role']=='user' else ('role','text','tool_calls','tool_results')
            history.append({k:deepcopy(turn[k]) for k in keys if k in turn})
        rows.append({'id':sid,'audio':raw['audio'],'context':context,'history':history})
    del source,lookup
    transcripts=json.loads((work/'asr_transcripts.json').read_text());asr=json.loads((work/'asr_report.json').read_text())
    assert not asr['oracle_fallback'] and asr['validation_samples']==len(rows)
    backend=Backend(a.model,a.revision,a.adapters);preflight=backend.preflight();write_json(work/'preflight.json',preflight)
    catalog=load_catalog(root,split);ToolEnv=import_toolenv(root,split)
    batches=work/'batches';batches.mkdir(exist_ok=True)
    active=work/'active_batch.json'
    if active.exists():
        marker=json.loads(active.read_text())
        assert (batches/marker['file']).exists(),'An interrupted batch has unrecorded tool execution. Do not silently replay it.'
        active.unlink()
    predictions=[];traces=[];mode='single' if a.mode=='single' else 'multi'
    with (work/'submission.jsonl').open('w') as pf,(work/'traces.jsonl').open('w') as tf:
        for start in range(0,len(rows),4):
            if a.deadline_monotonic and time.monotonic()>a.deadline_monotonic and not (batches/f'{start:06}.json').exists():
                write_json(work/'progress.json',{'paused':True,'completed':len(predictions),'total':len(rows),'next_index':start})
                print('PAUSED_AT_BATCH_BOUNDARY: preserve this run directory and resume in another session',flush=True)
                raise SystemExit(75)
            group=rows[start:start+4]
            batch_file=batches/f'{start:06}.json'
            if batch_file.exists():
                completed=json.loads(batch_file.read_text())
                assert [p[0]['id'] for p in completed]==[r['id'] for r in group]
            else:
                write_json(active,{'file':batch_file.name,'ids':[r['id'] for r in group]})
            if not batch_file.exists():
                envs=[ToolEnv(split,raw['id'],root=root) for raw in group]
                safe_rows=[normalized_row(raw,transcript=transcripts) for raw in group]
                gens=[rollout(row,transcripts[raw['audio']],allowed_specs(row,catalog),env,mode)
                      for raw,row,env in zip(group,safe_rows,envs,strict=True)]
                completed=run_batch(gens,backend)
                temp=batch_file.with_suffix('.tmp');write_json(temp,completed);temp.replace(batch_file);active.unlink()
            for pred,trace in completed:
                predictions.append(pred);traces.append(trace);pf.write(dumps(pred)+'\n');pf.flush();tf.write(dumps(trace)+'\n');tf.flush()
            print(dumps({'split':a.split,'mode':a.mode,'completed':len(predictions),'total':len(rows)}),flush=True)
    assert [r['id'] for r in predictions]==ids
    report={'split':a.split,'mode':a.mode,'samples':len(rows),'asr':asr,'preflight':preflight,
            'actual_batch_sizes':dict(Counter(backend.batch_sizes)),'schema_violations':backend.schema_violations,
            'role_adapter_map':{'global':'global','planner':'planner','caller':'global','summarizer':'global'},
            'missing_local_adapters':['caller','summarizer'],'policy':POLICY_AUDIT,
            'termination_reasons':dict(Counter(t['reason'] for t in traces))}
    if a.split=='validation':
        # Scorer-only gold replay after ALL model predictions have been generated.
        from umilife.evaluate import evaluate
        references=[]
        for raw in read_jsonl(root/'train/train.jsonl'):
            if raw['id'] not in set(ids):continue
            env=ToolEnv('train',raw['id'],root=root)
            for action in raw['label']['tool_calls']:env.call(action['name'],action['arguments'])
            references.append({'id':raw['id'],'tool_calls':raw['label']['tool_calls'],'state_hash':env.get_state_hash()})
        report['metrics']=evaluate(predictions,references,catalog,traces)
    write_json(work/'report.json',report)
    print(dumps({k:v for k,v in report.items() if k!='metrics'}),flush=True)
if __name__=='__main__':main()
