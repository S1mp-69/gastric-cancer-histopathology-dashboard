"""Verify acquired HiESD pilot and run a train-only CUDA engineering smoke check."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[k]='1'
import csv
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
from PIL import Image,ImageOps
import torch
from hiesd_pilot import ROOT,digest,save,NAMES
sys.path.insert(0,str(ROOT))
from app.ml.model import GastricCancerRNN
from app.ml.preprocessing import image_to_patch_sequence

def rows(path):
    with path.open(newline='',encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))

def main():
    base=ROOT/'data/external/hiesd_pilot_v1'
    pilot=ROOT/'data/external/hiesd_pilot_v2'
    stamp=datetime.now(ZoneInfo('Asia/Calcutta')).strftime('%Y%m%d_%H%M%S')
    run=ROOT/'stain_research/runs'/f'hiesd_smoke_{stamp}'
    portable=ROOT/'stain_research/results'/f'hiesd_smoke_{stamp}'
    run.mkdir(parents=True,exist_ok=False);portable.mkdir(parents=True,exist_ok=False)
    mapping={r['slide_id']:r for r in rows(ROOT/'data/external/hiesd_metadata/prospective_grouped_manifest.csv')}
    records=rows(pilot/'patch_manifest.csv')
    source_checks=[]
    for folder in [base,pilot]:
        plan=json.loads((folder/'PLAN.json').read_text())
        assert plan['manifest_sha256']==digest(ROOT/'data/external/hiesd_metadata/prospective_grouped_manifest.csv')
        for f in plan['selected']:
            path=folder/'slides'/f['name']
            assert mapping[f['slide_id']]['split']==f['split'] and f['split'] in ['train','validation']
            assert path.stat().st_size==f['size'] and digest(path,'md5')==f['computed_md5']
            source_checks.append({'slide_id':f['slide_id'],'bytes':f['size'],'md5_verified':True})
    train_patients={r['patient_id'] for r in records if r['split']=='train'}
    val_patients={r['patient_id'] for r in records if r['split']=='validation'}
    assert not train_patients & val_patients
    hashes={'train':set(),'validation':set()}
    for r in records:
        assert r['split'] in hashes and mapping[r['slide_id']]['split']==r['split']
        assert mapping[r['slide_id']]['patient_id']==r['patient_id']
        assert NAMES[int(r['class_id'])]==r['class_code'] and float(r['mask_group_purity'])>=.9
        path=(ROOT/r['image_path']).resolve()
        assert path.is_relative_to((ROOT/'data/external').resolve())
        with Image.open(path) as im:
            assert im.size==(224,224) and im.mode=='RGB';im.load()
        hashes[r['split']].add(digest(path))
    assert not hashes['train'] & hashes['validation']
    counts={s:dict(Counter(r['class_code'] for r in records if r['split']==s)) for s in hashes}
    assert all(counts[s].get(n,0)>0 for s in hashes for n in NAMES)
    verification={'source_slides_verified':source_checks,'patches_verified':len(records),'counts':counts,
                  'train_patients':sorted(train_patients),'validation_patients':sorted(val_patients),
                  'patient_overlap':False,'exact_patch_overlap':False,'reserved_test_acquired':False,
                  'patch_manifest_sha256':digest(pilot/'patch_manifest.csv'),
                  'limitations':['Annotation-guided selected subset','12 validation lymphoid patches','Coarse annotation mapping without expert review','Provisional five-class grouping']}
    save(portable/'verification.json',verification)
    print('Verified all 13 raw slides and 1811 patch files; one original slide remains excluded.',flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing CPU training fallback')
    torch.set_num_threads(2);torch.manual_seed(20261116);random.seed(20261116);np.random.seed(20261116)
    torch.cuda.manual_seed_all(20261116)
    device=torch.device('cuda:0');model=GastricCancerRNN(num_classes=5).to(device)
    assert next(model.parameters()).is_cuda
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    scaler=torch.amp.GradScaler('cuda',init_scale=4096)
    selected=[]
    for name in NAMES:
        selected.extend([r for r in records if r['split']=='train' and r['class_code']==name][:8])
    # Fixed [0.5,0.5,0.5] normalization is an execution-check choice only.
    # No validation fitting, normalization fitting, or clinical accuracy claim.
    def batch(indices,augment):
        xs=[];ys=[]
        for i in indices:
            r=selected[i]
            with Image.open(ROOT/r['image_path']) as source:
                im=source.convert('RGB')
            if augment and random.random()<.5:im=ImageOps.mirror(im)
            xs.append(image_to_patch_sequence(im,56,16,(.5,.5,.5),(.5,.5,.5)))
            ys.append(int(r['class_id']))
        return torch.stack(xs).pin_memory().to(device,non_blocking=True),torch.tensor(ys,device=device)
    timings=[];losses=[];updates=0;backoffs=0
    torch.cuda.reset_peak_memory_stats()
    for step in range(6):
        torch.cuda.synchronize();start=time.perf_counter()
        idx=np.random.default_rng(20261116+step).choice(len(selected),32,replace=False)
        x,y=batch(idx,True)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        with torch.autocast('cuda',dtype=torch.float16):
            logits,_=model(x)
            loss=torch.nn.functional.cross_entropy(logits,y,label_smoothing=.05)
        assert torch.isfinite(loss)
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        before=scaler.get_scale();scaler.step(optimizer);scaler.update()
        if scaler.get_scale()<before:backoffs+=1
        else:
            assert torch.isfinite(grad);updates+=1
        torch.cuda.synchronize()
        timings.append(time.perf_counter()-start);losses.append(float(loss))
    assert updates>=4 and all(torch.isfinite(t).all() for t in model.state_dict().values())
    model.eval();probe,_=batch(range(4),False)
    with torch.inference_mode():expected=model(probe)[0]
    checkpoint=run/'smoke_checkpoint.pt'
    torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                'scaler_state_dict':scaler.state_dict(),'torch_rng_state':torch.get_rng_state(),
                'cuda_rng_state':torch.cuda.get_rng_state_all(),'numpy_rng_state':np.random.get_state(),
                'python_rng_state':random.getstate(),'step':6,'class_order':NAMES,
                'purpose':'Engineering smoke only; no validation evaluation or checkpoint selection'},checkpoint)
    state=torch.load(checkpoint,map_location=device,weights_only=False)
    restored=GastricCancerRNN(num_classes=5).to(device);restored.load_state_dict(state['model_state_dict']);restored.eval()
    with torch.inference_mode():actual=restored(probe)[0]
    assert torch.allclose(expected,actual,atol=1e-6,rtol=1e-6)
    throughput=32*4/sum(timings[2:])
    free,total=torch.cuda.mem_get_info()
    result={'purpose':'CUDA execution, data loading and checkpoint reload smoke only; not convergence or classifier validation',
            'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,
            'device':str(device),'mixed_precision':'FP16 AMP','batch_size':32,'cpu_threads':2,'loader_workers':0,
            'train_only_source_patches':len(selected),'warmup_steps':2,'measured_steps':4,'successful_optimizer_updates':updates,
            'amp_backoffs':backoffs,'losses':losses,'step_seconds':timings,'measured_images_per_second':throughput,
            'peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2,'free_MiB_at_end':free/1024**2,
            'checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint),'reload_predictions_match':True,
            'rough_train_epoch_seconds_from_short_smoke':sum(counts['train'].values())/throughput,
            'estimate_limitations':'Short warm-cache sample, no validation or stain augmentation time; not a full-campaign runtime estimate.',
            'full_training_started':False,'reserved_test_used':False}
    save(portable/'smoke.json',result)
    save(pilot/'POST_DOWNLOAD_VERIFICATION.json',{'verification':str(portable/'verification.json'),'smoke':str(portable/'smoke.json'),'passed':True,'full_training_started':False})
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
