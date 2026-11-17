"""Training-only finite-difference study; no fitting or validation/test decoding."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import csv,json,time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as F
from scipy.stats import rankdata,spearmanr
from hiesd_pilot import ROOT,NAMES,digest
from hiesd_resnet18 import GastricCancerRNN,atomic
from efficient_cache import cached_apply

def correlation(x,y):
    if np.std(x)<1e-12 or np.std(y)<1e-12:return None
    return float(spearmanr(x,y).statistic)

def centered_ranks(x,classes):
    result=np.zeros(len(x))
    for c in np.unique(classes):
        ids=np.flatnonzero(classes==c);r=rankdata(x[ids]);r-=r.mean()
        result[ids]=r/max(float(r.std()),1e-12)
    return result

def main():
    torch.set_num_threads(2)
    stamp=datetime.now(ZoneInfo('Asia/Calcutta')).strftime('%Y%m%d_%H%M%S')
    out=ROOT/'stain_research/results'/f'assignment_feasibility_{stamp}';out.mkdir(parents=True,exist_ok=False)
    base=ROOT/'stain_research/runs/hiesd_resnet18_seed42_v3'
    campaign=ROOT/'stain_research/runs/hiesd_resnet_ablation_v1'
    directory=Path(json.loads((campaign/'cache_reference.json').read_text())['directory'])
    with (directory/'index.csv').open(newline='') as stream:index=list(csv.DictReader(stream))
    with (ROOT/'data/external/hiesd_metadata/prospective_grouped_manifest.csv').open(newline='') as stream:
        mapping={r['slide_id']:r for r in csv.DictReader(stream)}
    train=[(i,r) for i,r in enumerate(index) if r['split']=='train']
    for _,r in train:assert mapping[r['slide_id']]['split']=='train' and mapping[r['slide_id']]['patient_id']==r['patient_id']
    rng=np.random.default_rng(20261117);selected=[]
    # Round-robin patients within each class, chosen before model evaluation.
    for name in NAMES:
        groups=defaultdict(list)
        for i,r in train:
            if r['class_code']==name:groups[r['patient_id']].append(i)
        for values in groups.values():rng.shuffle(values)
        chosen=[];patients=sorted(groups);rng.shuffle(patients)
        while len(chosen)<32 and any(groups.values()):
            for p in patients:
                if groups[p] and len(chosen)<32:chosen.append(groups[p].pop())
        selected.extend(chosen)
    assert len(selected)==160
    checkpoints={'initial':base/'initial.pt','trained_last':base/'last.pt'}
    hashes={n:digest(p) for n,p in checkpoints.items()}
    plan={'created':stamp,'sample_n':160,'per_class':32,'sample_selection':'Seed20261117, class-stratified patient round robin; TRAIN only',
          'sample_cache_indices':selected,'sample_records':[index[i] for i in selected],
          'cache_key':json.loads((campaign/'cache_reference.json').read_text())['key'],
          'checkpoint_sha256':hashes,'strengths':[0.,.05,.25,.5,1.],'directions':3,
          'probe_epsilon':.05,'primary_target_strength':.5,
          'predefined_engineering_gate':'For trained_last: >=3/5 class-wise same-direction Spearman correlations >.3 and within-class-rank patient-bootstrap lower bound >0; no clinical/statistical superiority interpretation.',
          'cross_direction_check':'Small-probe scores for direction0 compared with strong loss changes averaged over independent directions1/2.',
          'model_mode':'eval, frozen BatchNorm, FP32; no optimizer or gradient updates',
          'pipeline':'Raw RGB, ImageNet normalization; deliberately no geometric or RGB jitter to isolate stain response. Does not validate the eventual full training pipeline.',
          'historical_selection_limit':'trained_last derives from a prior validation-early-stopped run. This analysis reads only TRAIN pixels/labels, but its checkpoint is not validation-naive.',
          'test_and_validation_pixels_accessed':False,'automatic_training':False}
    atomic(out/'PLAN.json',plan)
    maps={n:np.load(directory/f'{n}.npy',mmap_mode='r') for n in ['rgb','basis','concentrations','diagnostics','valid']}
    assert all(bool(maps['valid'][i]) for i in selected)
    params=json.loads((directory/'train_parameters.json').read_text());levels=plan['strengths']
    model=GastricCancerRNN(5).cuda();model.eval();all_records=[];summaries={};started=time.perf_counter()
    def output(image):return F.normalize(F.to_tensor(Image.fromarray(image)),[.485,.456,.406],[.229,.224,.225])
    classes=np.repeat(np.arange(5),32*3)
    patients=np.array([index[i]['patient_id'] for i in selected for _ in range(3)])
    for model_name,path in checkpoints.items():
        state=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(state['model_state_dict']);model.eval();del state
        losses=np.empty((len(selected),3,len(levels)));rms=np.empty_like(losses)
        pending=[];positions=[];targets=[]
        def flush():
            if not pending:return
            with torch.inference_mode():
                logits,_=model(torch.stack(pending).cuda());y=torch.tensor(targets,device='cuda')
                values=torch.nn.functional.cross_entropy(logits,y,reduction='none').cpu().numpy()
            assert np.isfinite(values).all()
            for pos,value in zip(positions,values):losses[pos]=value
            pending.clear();positions.clear();targets.clear()
        for j,i in enumerate(selected):
            rgb=np.array(maps['rgb'][i]);label=NAMES.index(index[i]['class_code'])
            for direction in range(3):
                for k,alpha in enumerate(levels):
                    local_rng=np.random.default_rng(np.random.SeedSequence([20261117,int(i),direction,811]))
                    augmented,_,_=cached_apply(rgb,maps['basis'][i],maps['concentrations'][i],maps['diagnostics'][i],True,params,local_rng,alpha)
                    rms[j,direction,k]=float(np.sqrt(np.mean((augmented.astype(float)-rgb)**2)))
                    pending.append(output(augmented));positions.append((j,direction,k));targets.append(label)
                    if len(pending)==16:flush()
            if (j+1)%32==0:
                atomic(out/'progress.json',{'stage':'inference','checkpoint':model_name,'images_complete':j+1,'images_total':160})
        flush();np.savez_compressed(out/f'{model_name}_responses.npz',losses=losses,rgb_rms=rms,indices=selected,strengths=levels)
        small=(losses[:,:,1]-losses[:,:,0])/.05
        strong=losses[:,:,3]-losses[:,:,0]
        x=centered_ranks(small.ravel(),classes);y=centered_ranks(strong.ravel(),classes)
        observed=float(np.corrcoef(x,y)[0,1]);unique=sorted(set(patients));boot=[]
        for _ in range(1000):
            picked=np.random.default_rng(701+_).choice(unique,len(unique),replace=True)
            ids=np.concatenate([np.flatnonzero(patients==p) for p in picked])
            value=float(np.corrcoef(x[ids],y[ids])[0,1])
            if np.isfinite(value):boot.append(value)
        per_class={name:correlation(small[c*32:(c+1)*32].ravel(),strong[c*32:(c+1)*32].ravel()) for c,name in enumerate(NAMES)}
        lo,hi=np.quantile(boot,[.025,.975])
        cross_classes=np.repeat(np.arange(5),32)
        cx=centered_ranks(small[:,0],cross_classes);cy=centered_ranks(strong[:,1:].mean(1),cross_classes)
        summary={'same_direction_within_class_standardized_rank_correlation':observed,'per_class_same_direction_spearman':per_class,
                 'patient_cluster_bootstrap_interval':[float(lo),float(hi)],'bootstrap_scope':'Ranks fixed before patient resampling; descriptive, conditional on selected patches and existing checkpoint; not a patient-independent confirmatory test.',
                 'cross_direction_within_class_rank_correlation':float(np.corrcoef(cx,cy)[0,1]),
                 'mean_probe_rgb_rms':float(rms[:,:,1].mean()),'zero_probe_fraction':float((rms[:,:,1]==0).mean()),
                 'secondary_target_correlations':{str(levels[k]):correlation(small.ravel(),(losses[:,:,k]-losses[:,:,0]).ravel()) for k in [2,4]},
                 'engineering_gate_passed':bool(sum(v is not None and v>.3 for v in per_class.values())>=3 and lo>0)}
        summaries[model_name]=summary
        for j,i in enumerate(selected):
            for direction in range(3):
                all_records.append({'checkpoint':model_name,'cache_index':i,'patient_id':index[i]['patient_id'],'class':index[i]['class_code'],'direction':direction,
                                    'small_probe_slope':small[j,direction],'strong_loss_change':strong[j,direction]})
        print(model_name,json.dumps(summary),flush=True)
    assert all(digest(checkpoints[n])==h for n,h in hashes.items())
    with (out/'responses.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(all_records[0]));writer.writeheader();writer.writerows(all_records)
    result={'stage':'complete','summaries':summaries,'elapsed_seconds':time.perf_counter()-started,'original_checkpoints_unchanged':True,
            'training_started':False,'new_validation_or_test_pixels_read':False,
            'interpretation':'This measures loss-response prediction only. Even a passed gate does not establish classification improvement, beneficial assignment, true morphology preservation, novelty, or publication readiness.'}
    atomic(out/'RESULTS.json',result);atomic(out/'progress.json',result)
    s=summaries['trained_last'];text=['# Stain-assignment feasibility: training-only','',
        f"Sample:160 training patches,32/class, three stain draws each; probe alpha=.05, primary stronger alpha=.5. Initial and frozen trained-last checkpoints evaluated without fitting.",'',
        f"Trained checkpoint within-class rank correlation: {s['same_direction_within_class_standardized_rank_correlation']:.4f}; descriptive patient-cluster interval [{s['patient_cluster_bootstrap_interval'][0]:.4f}, {s['patient_cluster_bootstrap_interval'][1]:.4f}].",
        f"Cross-direction correlation: {s['cross_direction_within_class_rank_correlation']:.4f}.",
        f"Predefined engineering gate passed: {s['engineering_gate_passed']}.",'',
        '| Tissue | Same-direction Spearman |','|---|---:|']
    for name,value in s['per_class_same_direction_spearman'].items():text.append(f'|{name}|{value:.4f}|' if value is not None else f'|{name}|Undefined|')
    text+=['',result['interpretation'],'','No new validation/test pixels were accessed; the trained checkpoint historically used validation for early stopping. No automatic training campaign launched. The sample/thresholds are engineering choices, and probe/target share a stain direction by design. This can inflate apparent predictability relative to unseen directions. FP32 inference reduces mixed-precision rounding but uint8 reconstruction still quantizes small probes. Geometric/RGB jitter was omitted here; utility in the full training pipeline remains untested.','',
           'Next: if the gate passed, test the exact within-class assignment invariant and actual loss under ranked versus random strengths before any classifier comparison; otherwise investigate the failed surrogate without searching validation for a favorable replacement.']
    (out/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8');print('REPORT',out,flush=True)

if __name__=='__main__':main()
