"""Strict source-ID manifests and detached prefix cache (offline training only)."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageEnhance

TRANSFORM_VERSION = 'rgb-bicubic-shorter224-center224-imagenet-v1'
MEAN = np.array([.485,.456,.406],dtype=np.float32)
STD = np.array([.229,.224,.225],dtype=np.float32)

def read_manifest(path):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    required = {'id','parent_id','path','label','split','role','domain','episode'}
    ids = set()
    for row in rows:
        if not required <= row.keys(): raise ValueError(f'Missing fields: {required-row.keys()}')
        if row['id'] in ids: raise ValueError('Duplicate row id')
        ids.add(row['id'])
        if row['split'] not in {'train','val','test'} or row['role'] not in {'stream','past','clean'}: raise ValueError('split/role')
        if not isinstance(row['label'],int) or not 0 <= row['label'] < 1000: raise ValueError('ImageNet label must be 0..999')
    return rows

def audit_rows(rows):
    parents = {}
    streams = set()
    for r in rows:
        key=(r['split'],r['role'])
        old=parents.setdefault(r['parent_id'],key)
        if old != key: raise ValueError(f"Parent crosses split or stream/probe role: {r['parent_id']}")
        if r['role']=='stream':
            # Strict new-image protocol across all corruption variants/episodes.
            p=(r['split'],r['parent_id'])
            if p in streams: raise ValueError(f"Repeated stream source image: {p}")
            streams.add(p)
    return {'rows':len(rows),'unique_parent_ids':len(parents),'strict_stream':True}

def image_tensor(row):
    with Image.open(row['path']) as raw: image=raw.convert('RGB')
    aug=row.get('augmentation','none')
    if aug=='brightness': image=ImageEnhance.Brightness(image).enhance(.65)
    elif aug=='contrast': image=ImageEnhance.Contrast(image).enhance(.65)
    elif aug=='jpeg':
        buf=io.BytesIO(); image.save(buf,format='JPEG',quality=45); buf.seek(0)
        image=Image.open(buf).convert('RGB')
    elif aug!='none': raise ValueError(f'Unknown augmentation: {aug}')
    w,h=image.size
    ow,oh=(224,int(224*h/w)) if w<h else (int(224*w/h),224)
    image=image.resize((ow,oh),Image.Resampling.BICUBIC)
    l,t=int(round((ow-224)/2)),int(round((oh-224)/2))
    array=np.asarray(image.crop((l,t,l+224,t+224)),dtype=np.float32)/255
    return torch.from_numpy(((array-MEAN)/STD).transpose(2,0,1).copy()).unsqueeze(0)

class FeatureCache:
    def __init__(self,directory,weights_sha,blocks):
        p=Path(directory)
        self.meta=json.loads((p/'meta.json').read_text())
        if self.meta['weights_sha256']!=weights_sha or self.meta['adapted_blocks']!=blocks: raise ValueError('Cache/backbone mismatch')
        if self.meta['transform']!=TRANSFORM_VERSION: raise ValueError('Cache transform mismatch')
        self.rows=read_manifest(p/'manifest.jsonl')
        audit_rows(self.rows)
        manifest_sha=hashlib.sha256((p/'manifest.jsonl').read_bytes()).hexdigest()
        if self.meta['manifest_sha256']!=manifest_sha: raise ValueError('Cache manifest changed')
        self.array=np.load(p/'features.npy',mmap_mode='r')
        if len(self.rows)!=len(self.array): raise ValueError('Incomplete cache')
    def feature(self,i,device):
        return torch.from_numpy(np.array(self.array[i],copy=True)).float().unsqueeze(0).to(device)

def build_cache(manifest,output,backbone,weights_sha,device,backend='reference',fp16=False,resource_guard=None,batch_size=1):
    if not isinstance(batch_size,int) or not 1 <= batch_size <= 16:
        raise ValueError('Prefix batch size must be 1..16')
    if str(device)!='cpu' and batch_size!=1:
        raise ValueError('Only CPU prefix batching is currently verified')
    rows=read_manifest(manifest); audit_rows(rows)
    if not rows: raise ValueError('Empty manifest')
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    if (out/'meta.json').exists(): raise FileExistsError('Use a new cache directory; never silently overwrite')
    arr=None
    with torch.no_grad():
        for i in range(0,len(rows),batch_size):
            batch=rows[i:i+batch_size]
            if resource_guard: resource_guard.sample('before_prefix')
            x=torch.cat([image_tensor(r) for r in batch]).to(device)
            f=backbone.prefix(x).float().cpu().numpy()
            if resource_guard: resource_guard.sample('prefix')
            if not np.isfinite(f).all(): raise FloatingPointError(f'Nonfinite prefix batch at {batch[0]["id"]}')
            if len(f)!=len(batch): raise ValueError('Prefix batch cardinality changed')
            if arr is None: arr=np.lib.format.open_memmap(out/'features.npy',mode='w+',dtype=np.float16 if fp16 else np.float32,shape=(len(rows),*f.shape[1:]))
            arr[i:i+len(batch)]=f
            if i%100==0: print(json.dumps({'cached':i+len(batch),'total':len(rows)}),flush=True)
    arr.flush()
    text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    (out/'manifest.jsonl').write_text(text)
    meta={'weights_sha256':weights_sha,'adapted_blocks':backbone.adapted_blocks,'transform':TRANSFORM_VERSION,
          'manifest_sha256':hashlib.sha256(text.encode()).hexdigest(),'backend':backend,'device':str(device),
          'torch_version':str(torch.__version__),'prefix_batch_size':batch_size,'dtype':str(arr.dtype),'shape':list(arr.shape)}
    (out/'meta.json').write_text(json.dumps(meta,indent=2))
    if resource_guard:
        (out/'memory_profile.json').write_text(json.dumps(resource_guard.sample('cache_complete'),indent=2))
    return meta
