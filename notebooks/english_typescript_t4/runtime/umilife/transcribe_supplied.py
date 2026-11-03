"""Recognize current/history audio using only the supplied Zipformer2 ONNX files."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import sherpa_onnx

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def transcribe(root,work,model,limit=None,threads=4):
    root=Path(root);work=Path(work);model=Path(model)
    manifest=json.loads((model/'model_manifest.json').read_text())
    assert manifest['model_type']=='zipformer2' and not manifest['streaming']
    for name,info in manifest['files'].items():assert digest(model/name)==info['sha256'],f'Wrong ASR file: {name}'
    fingerprint=digest(model/'model_manifest.json')
    valid=set((work/'split.txt').read_text().splitlines());assert 0<len(valid)<=100
    rows=[json.loads(line) for line in (root/'train/train.jsonl').open()]
    selected=[r for r in rows if r['id'] in valid];assert len(selected)==len(valid)
    if limit:selected=selected[:limit]
    audio=[]
    for row in selected:
        audio.append(row['audio'])
        audio.extend(t['audio'] for t in row.get('history',[]) if t['role']=='user')
    audio=list(dict.fromkeys(audio))
    config={'model_manifest_sha256':fingerprint,'split_sha256':digest(work/'split.txt'),
        'selected_ids':[r['id'] for r in selected],'sample_rate':16000,'decoding_method':'greedy_search',
        'provider':'cpu','sherpa_onnx':sherpa_onnx.__version__,'source':'supplied_asr_model'}
    config_path=work/'asr_config.json'
    if config_path.exists():assert json.loads(config_path.read_text())==config,'Stale ASR cache configuration'
    else:config_path.write_text(json.dumps(config,indent=2))
    cache_path=work/'asr_transcripts.jsonl';cache={}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            record=json.loads(line);assert record['model_manifest_sha256']==fingerprint
            cache[record['audio']]=record
    recognizer=sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(model/'encoder.onnx'),decoder=str(model/'decoder.onnx'),joiner=str(model/'joiner.onnx'),
        tokens=str(model/'tokens.txt'),num_threads=threads,sample_rate=16000,feature_dim=80,
        decoding_method='greedy_search',provider='cpu')
    started=time.monotonic()
    with cache_path.open('a',encoding='utf-8') as output:
        for index,name in enumerate(audio,1):
            path=(root/'train'/name).resolve();assert path.is_relative_to((root/'train').resolve())
            audio_sha=digest(path)
            if name in cache:
                assert cache[name]['audio_sha256']==audio_sha,'Audio changed after transcription'
                continue
            tick=time.monotonic()
            try:
                samples,sample_rate=sf.read(path,dtype='float32',always_2d=True)
                samples=samples.mean(axis=1)
                assert len(samples)>0 and np.isfinite(samples).all(),'Empty or non-finite waveform'
                duration=len(samples)/sample_rate
                if sample_rate!=16000:
                    factor=math.gcd(sample_rate,16000)
                    samples=resample_poly(samples,16000//factor,sample_rate//factor)
                stream=recognizer.create_stream()
                stream.accept_waveform(16000,np.ascontiguousarray(samples,dtype=np.float32))
                recognizer.decode_stream(stream)
                record={'audio':name,'text':stream.result.text.strip(),'audio_sha256':audio_sha,
                    'model_manifest_sha256':fingerprint,'elapsed_s':time.monotonic()-tick,'duration_s':duration,
                    'source':'supplied_asr_model','provider':'cpu'}
            except Exception as exc:
                (work/'asr_error.json').write_text(json.dumps({'audio':name,'error':str(exc)}))
                raise
            output.write(json.dumps(record,ensure_ascii=False)+'\n');output.flush();cache[name]=record
            print(json.dumps({'stage':'asr','completed':index,'total_audio':len(audio),
                'validation_samples':len(selected),'audio':name,'elapsed_s':record['elapsed_s']},ensure_ascii=False),flush=True)
    assert set(audio)<=cache.keys()
    transcripts={name:cache[name]['text'] for name in audio}
    (work/'asr_transcripts.json').write_text(json.dumps(transcripts,ensure_ascii=False,indent=2))
    report={'status':'complete','validation_samples':len(selected),'unique_audio':len(audio),
        'current_and_history_audio':True,'oracle_fallback':False,'training':False,
        'empty_transcripts':sum(not value.strip() for value in transcripts.values()),
        'wall_s_this_run':time.monotonic()-started,
        'audio_duration_s':sum(cache[name]['duration_s'] for name in audio),
        'initializer_elements':sum(info.get('initializer_elements',0) for info in manifest['files'].values()),
        'model_manifest_sha256':fingerprint,'provider':'cpu','model_type':'zipformer2'}
    (work/'asr_report.json').write_text(json.dumps(report,indent=2))
    print('ASR_COMPLETE',json.dumps(report),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('data-root','work','model-dir'):p.add_argument('--'+name,required=True)
    p.add_argument('--limit',type=int);p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();transcribe(a.data_root,a.work,a.model_dir,a.limit,a.threads)
