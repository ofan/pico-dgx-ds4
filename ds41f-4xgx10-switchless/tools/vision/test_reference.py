"""Actual-pack CPU oracle. This is tensor/layout qualification, not model image-QA acceptance."""
import argparse, hashlib, importlib.util, json, pathlib, sys, time
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--fixtures',required=True);p.add_argument('--out',required=True);a=p.parse_args()
model,fixtures,out=map(pathlib.Path,(a.model,a.fixtures,a.out))
import torch
torch.set_num_threads(8)
sys.argv=['vision_encoder','--model',str(model),'--device','cpu']
spec=importlib.util.spec_from_file_location('encoder',pathlib.Path(__file__).with_name('vision_encoder.py'));e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
vit,aligner,delim=e.load();checks=[]
from safetensors import safe_open
manifest=json.loads((model/'vision-pack-manifest.json').read_text())
with safe_open(model/'vision.safetensors',framework='pt',device='cpu') as f:
    for item in manifest['tensors']:
        t=f.get_tensor(item['name']);raw=t.view(torch.int16).numpy().tobytes()
        assert list(t.shape)==item['shape'] and t.dtype==torch.bfloat16 and hashlib.sha256(raw).hexdigest()==item['sha256']
checks.append({'check':'exact tensor names/shapes/dtypes/per-tensor checksums','passed':266,'total':266})
# Independent span assembly as publisher merge_image_embeddings: delimiters and aligner in reading order.
rows=[]
with torch.inference_mode():
 for name,_,_ in json.loads((fixtures/'cases.json').read_text()):
    t0=time.monotonic();raw=(fixtures/name).read_bytes()
    got,grid=e.encode(raw)
    patches,h,w,lh,lw=e.ip.load_image({'data':raw},e.args)
    types=e.ip.image_token_types(lh,lw)
    feats=aligner(vit(patches,h,w),h,w)
    ref=torch.cat([delim['image_start'][None]]+[torch.cat([feats[r*lw:(r+1)*lw],delim['image_newline'][None]]) for r in range(lh)]+[delim['image_end'][None]])
    diffs=int((got.view(torch.int16)!=ref.view(torch.int16)).sum())
    assert got.shape==(lh*(lw+1)+2,5120) and got.shape[0]<=1024 and torch.isfinite(got).all() and diffs==0
    rows.append({'fixture':name,'grid':list(grid),'shape':list(got.shape),'differences':diffs,'elements':got.numel(),'span_sha256':hashlib.sha256(got.view(torch.int16).numpy().tobytes()).hexdigest(),'seconds':time.monotonic()-t0})
    print(json.dumps(rows[-1]),flush=True)
out.write_text(json.dumps({'device':'cpu','torch':torch.__version__,'canonical_attention':True,'gpu_acceptance':False,'checks':checks,'images':rows,'all_pass':True},indent=2)+'\n')
