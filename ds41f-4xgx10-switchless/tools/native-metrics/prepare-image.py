#!/usr/bin/env python3
"""Assemble a new context from explicitly extracted baseline metadata and new binaries.
Never starts a container, touches serving, downloads a model, or reads credentials.
"""
import argparse, hashlib, json, pathlib, re, shutil
p=argparse.ArgumentParser();p.add_argument('--context',required=True,type=pathlib.Path)
p.add_argument('--baseline-profile',required=True,type=pathlib.Path);p.add_argument('--baseline-entrypoint',required=True,type=pathlib.Path)
p.add_argument('--engine',required=True,type=pathlib.Path);p.add_argument('--frontend',required=True,type=pathlib.Path)
p.add_argument('--source-revision',required=True);p.add_argument('--base-image',required=True)
a=p.parse_args()
assert re.fullmatch(r'[0-9a-f]{40}',a.source_revision), 'Need an immutable source commit'
assert re.search(r'@sha256:[0-9a-f]{64}$',a.base_image), 'Need pinned baseline image'
a.context.mkdir() # refuse an existing context; never overwrite a previous build
here=pathlib.Path(__file__).resolve().parent
profile=json.loads(a.baseline_profile.read_text())
assert profile.get('model') and profile.get('model_revision'), 'Missing model lineage'
for src,name,key in [(a.engine,'dsgb10_rank','engine_sha256'),(a.frontend,'dsfront','frontend_sha256')]:
    shutil.copyfile(src,a.context/name)
    profile[key]=hashlib.sha256(src.read_bytes()).hexdigest()
profile['adoption']={'status':'UNQUALIFIED native telemetry trial','production_swapped':False}
profile['production_note']='Not deployed; requires independent review and full TP4 GPU qualification'
profile['build_policy']='Off-Spark ARM64 cross-build only; no native serving-host fallback'
profile['build_method']='Full ARM64 engine CUDA13 cross-build and locked offline Rust frontend cross-build; inherited pinned runtime/model/vision assets'
profile['baseline_source_commit']=profile.get('patched_source_commit',profile.get('source_commit'))
profile['rollback']={'image':a.base_image,'authorization':'Separate coordinated restart authorization required; image rollback does not recover sessions'}
profile['telemetry']={'schema':1,'source_revision':a.source_revision,'base_image':a.base_image,
    'rank_port':9108,'frontend_port':9109,'qualification':'UNQUALIFIED; TP4 GPU gates pending',
    'measurement_changes':'none intended; GPU parity and regression NOT RUN'}
profile['source_commit']=profile['patched_source_commit']=profile['frontend_source_commit']=a.source_revision
s=a.baseline_entrypoint.read_text()
engine="    children.append(subprocess.Popen(['stdbuf','-oL','-eL','/opt/pico/dsgb10_rank',*a]))"
assert s.count(engine)==1, 'Baseline engine launcher changed; review required'
s=s.replace(engine,"""    # Metrics-only additions; preserve all baseline model/serving argv.
    if '--metrics-listen' not in a:a+=['--metrics-listen',os.environ.get('PICO_ENGINE_METRICS_LISTEN','127.0.0.1:9108')]
    if '--metrics-model-name' not in a:a+=['--metrics-model-name','deepseek-v4.1-flash']
"""+engine)
front="        if enabled:cmd+=['--vision-addr','127.0.0.1:18950']"
assert s.count(front)==1, 'Baseline frontend launcher changed; review required'
s=s.replace(front,"        cmd+=['--metrics-listen',os.environ.get('PICO_FRONTEND_METRICS_LISTEN','127.0.0.1:9109')]\n"+front)
compile(s,'entrypoint.py','exec')
(a.context/'entrypoint.py').write_text(s)
(a.context/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
shutil.copyfile(here/'Dockerfile',a.context/'Dockerfile')
shutil.copyfile(here.parent.parent/'docs/native-telemetry.md',a.context/'native-telemetry.md')
print(json.dumps({'model':profile['model'],'model_revision':profile['model_revision'],
    'source_revision':a.source_revision,'engine_sha256':profile['engine_sha256'],
    'frontend_sha256':profile['frontend_sha256'],'qualification':profile['telemetry']['qualification']},indent=2))
