#!/usr/bin/env python3
"""Extract only native vision tensors from a pinned HF revision via bounded ranges.
No text checkpoint conversion. Requires independently retained shard headers.
"""
import argparse, hashlib, json, pathlib, struct, time, urllib.request
REV = '1bae0e1088993b9b82de08f59de5be5a34e42170'
MODEL = 'deepseek-ai/DeepSeek-V4.1-Flash-1017'

def fetch(shard, start, end):
    url = f'https://huggingface.co/{MODEL}/resolve/{REV}/{shard}?download=true&vision_range={start}-{end}'
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding':'identity'})
            with urllib.request.urlopen(req, timeout=180) as r:
                if r.status != 206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
                    raise ValueError('server did not honor exact range')
                data = r.read(end-start+2)
                if len(data) != end-start+1: raise ValueError('short/oversized range')
                return data
        except Exception as e:
            if attempt == 3: raise RuntimeError(f'range failure {shard} {start}:{end}: {type(e).__name__}') from None
            time.sleep(2**attempt)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inventory',type=pathlib.Path,required=True)
    ap.add_argument('--headers',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    inv=json.loads(args.inventory.read_text())['tensors']
    assert len(inv)==266 and sum(t['data_offsets'][1]-t['data_offsets'][0] for t in inv)==970536960
    starts={}; receipts=[]
    for shard in sorted({t['shard'] for t in inv}):
        n=struct.unpack('<Q',fetch(shard,0,7))[0];assert 0<n<16<<20
        raw=fetch(shard,8,7+n)
        expected=(args.headers/(shard+'.header.json')).read_bytes()
        assert raw==expected, 'published header differs from pinned investigation'
        hdr=json.loads(raw);starts[shard]=8+n
        for t in inv:
            if t['shard']==shard:
                assert hdr[t['name']]=={k:t[k] for k in ('dtype','shape','data_offsets')}
        receipts.append({'shard':shard,'header_sha256':hashlib.sha256(raw).hexdigest(),'header_bytes':n,'data_start':8+n})
    ordered=sorted(inv,key=lambda t:(t['shard'],t['data_offsets'][0]))
    h={};offset=0;groups=[]
    for t in ordered:
        lo,hi=t['data_offsets'];size=hi-lo
        h[t['name']]={'dtype':t['dtype'],'shape':t['shape'],'data_offsets':[offset,offset+size]}
        if groups and groups[-1]['shard']==t['shard'] and groups[-1]['end']==lo:
            groups[-1]['end']=hi
        else:groups.append({'shard':t['shard'],'start':lo,'end':hi,'out_start':offset})
        offset+=size
    h['__metadata__']={'source_model':MODEL,'source_revision':REV,'scope':'native vision only; text ranks unchanged'}
    hb=json.dumps(h,separators=(',',':')).encode();hb+=b' '*((-len(hb))%8)
    out=args.out/'vision.safetensors';partial=args.out/'vision.safetensors.partial'
    with partial.open('wb') as f:
        f.write(struct.pack('<Q',len(hb)));f.write(hb)
        for g in groups:
            for lo in range(g['start'],g['end'],16<<20):
                hi=min(lo+(16<<20),g['end'])
                f.seek(8+len(hb)+g['out_start']+lo-g['start'])
                data=fetch(g['shard'],starts[g['shard']]+lo,starts[g['shard']]+hi-1)
                f.write(data);f.flush()
                print(f'{g["shard"]}: {hi-g["start"]}/{g["end"]-g["start"]}',flush=True)
    hashes=[]
    with partial.open('rb') as f:
        for t in ordered:
            lo,hi=h[t['name']]['data_offsets'];f.seek(8+len(hb)+lo)
            sha=hashlib.sha256();remaining=hi-lo
            while remaining:
                b=f.read(min(1<<20,remaining));assert b;sha.update(b);remaining-=len(b)
            hashes.append(dict(name=t['name'],dtype=t['dtype'],shape=t['shape'],bytes=hi-lo,sha256=sha.hexdigest(),source_shard=t['shard'],source_data_offsets=t['data_offsets']))
    partial.rename(out)
    sha=hashlib.sha256()
    with out.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):sha.update(b)
    index={'metadata':{'total_size':offset},'weight_map':{t['name']:'vision.safetensors' for t in inv}}
    (args.out/'model.safetensors.index.json').write_text(json.dumps(index,indent=2)+'\n')
    (args.out/'vision-pack-manifest.json').write_text(json.dumps(dict(model=MODEL,revision=REV,file='vision.safetensors',sha256=sha.hexdigest(),bytes=out.stat().st_size,payload_bytes=offset,tensors=hashes,source_headers=receipts,method='Exact206 pinned-revision tensor ranges; byte-identical source headers; no text payload downloaded'),indent=2)+'\n')
    print('pack complete',sha.hexdigest(),flush=True)
if __name__=='__main__':main()
