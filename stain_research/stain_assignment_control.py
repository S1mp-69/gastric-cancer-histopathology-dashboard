"""Exact loss comparison for fixed within-class, within-batch strength banks."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import csv,json,time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as F
from hiesd_pilot import ROOT,NAMES,digest
from hiesd_resnet18 import GastricCancerRNN,atomic
from controller import strength
from efficient_cache import cached_apply

def main():
    torch.set_num_threads(2)
    previous=ROOT/'stain_research/results/assignment_feasibility_20261117_045112'
    old=json.loads((previous/'PLAN.json').read_text());selected=np.array(old['sample_cache_indices'])
    out=ROOT/'stain_research/results'/('assignment_control_'+datetime.now(ZoneInfo('Asia/Calcutta')).strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True,exist_ok=False)
    cam=ROOT/'stain_research/runs/hiesd_resnet_ablation_v1'
    directory=Path(json.loads((cam/'cache_reference.json').read_text())['directory'])
    assert json.loads((cam/'cache_reference.json').read_text())['key']==old['cache_key']
    with (directory/'index.csv').open(newline='') as stream:index=list(csv.DictReader(stream))
    assert all(index[i]['split']=='train' for i in selected)
    path=ROOT/'stain_research/runs/hiesd_resnet18_seed42_v3/last.pt'
    assert digest(path)==old['checkpoint_sha256']['trained_last']
    checkpoint_hash=digest(path)
    slopes=np.load(previous/'trained_last_responses.npz')['losses']
    slopes=(slopes[:,0,1]-slopes[:,0,0])/.05
    classes=np.array([index[i]['class_code'] for i in selected]);patients=np.array([index[i]['patient_id'] for i in selected])
    order=np.random.default_rng(2026111702).permutation(len(selected));groups=[]
    for start in range(0,len(order),16):
        batch=order[start:start+16]
        for code in sorted(set(classes[batch])):groups.append(batch[classes[batch]==code])
    plan={'parent_plan_sha256':digest(previous/'PLAN.json'),'checkpoint_sha256':checkpoint_hash,'sample_n':len(selected),
          'sample_policy':'Reuse balanced 32/class training-only feasibility sample; randomized fixed batches of16; not representative of natural class-imbalanced loader.',
          'batch_order':order.tolist(),'group_indices':[g.tolist() for g in groups],'random_assignments':20,
          'scoring':'Previously computed direction0 epsilon=.05 slopes; model eval FP32; no new fitting.',
          'directions':[0,1],'direction0':'Same stain target used to estimate sensitivity','direction1':'Independent stain target; transfer diagnostic only.',
          'primary':'Mean actual CE loss under ranked assignment minus mean across20 random assignments, direction0.',
          'secondary':'Reverse-ranked/original assignment and independent-direction comparison.',
          'engineering_gate':'Primary actual ranked mean CE exceeds random mean CE and exact class/batch strength multiset invariant holds. Not an accuracy or significance criterion.',
          'uncertainty':'Patient-cluster bootstrap of ranked-minus-random-mean image losses, conditional on the selected sample/model/groups; descriptive.',
          'budget':'Finite exact candidate table; every policy evaluated against the same image/strength/direction cells; cache reuse timing is not online training overhead.',
          'automatic_training':False,'validation_or_test_pixels_read':False}
    atomic(out/'PLAN.json',plan)
    params=json.loads((directory/'train_parameters.json').read_text())
    maps={n:np.load(directory/f'{n}.npy',mmap_mode='r') for n in ['rgb','basis','concentrations','diagnostics','valid']}
    assert all(maps['valid'][i] for i in selected)
    alpha=np.array([strength(maps['diagnostics'][i],params['controller']) for i in selected])
    policies={'ranked':np.arange(160),'reverse_ranked':np.arange(160),'original':np.arange(160)}
    for rep in range(20):policies[f'random_{rep:02}']=np.arange(160)
    for g in groups:
        sorted_images=g[np.argsort(slopes[g],kind='stable')];sorted_bank=g[np.argsort(alpha[g],kind='stable')]
        policies['ranked'][sorted_images]=sorted_bank;policies['reverse_ranked'][sorted_images]=sorted_bank[::-1]
        for rep in range(20):policies[f'random_{rep:02}'][g]=np.random.default_rng(np.random.SeedSequence([20261117,rep,int(g[0])])).permutation(g)
    for assignments in policies.values():
        for g in groups:assert np.array_equal(np.sort(alpha[assignments[g]]),np.sort(alpha[g]))
    np.savez_compressed(out/'assignments.npz',alpha=alpha,slopes=slopes,indices=selected,**policies)
    model=GastricCancerRNN(5).cuda();state=torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model_state_dict']);model.eval();del state
    loss_tables={};timings={};start_all=time.perf_counter()
    for direction in [0,1]:
        start=time.perf_counter();table=np.full((160,160),np.nan);pending=[];positions=[];labels=[];calls=0;transform_seconds=0.;inference_seconds=0.
        def flush():
            nonlocal calls,inference_seconds
            if not pending:return
            torch.cuda.synchronize();t=time.perf_counter()
            with torch.inference_mode():
                logits,_=model(torch.stack(pending).cuda());y=torch.tensor(labels,device='cuda')
                losses=torch.nn.functional.cross_entropy(logits,y,reduction='none').cpu().numpy()
            torch.cuda.synchronize();inference_seconds+=time.perf_counter()-t
            assert np.isfinite(losses).all()
            for pair,value in zip(positions,losses):table[pair]=value
            calls+=1;pending.clear();positions.clear();labels.clear()
        for g in groups:
            for j in g:
                i=int(selected[j]);rgb=np.array(maps['rgb'][i])
                for bank in g:
                    t=time.perf_counter();rng=np.random.default_rng(np.random.SeedSequence([20261117,i,direction,811]))
                    image,_,_=cached_apply(rgb,maps['basis'][i],maps['concentrations'][i],maps['diagnostics'][i],True,params,rng,float(alpha[bank]))
                    pending.append(F.normalize(F.to_tensor(Image.fromarray(image)),[.485,.456,.406],[.229,.224,.225]));positions.append((j,bank));labels.append(NAMES.index(classes[j]));transform_seconds+=time.perf_counter()-t
                    if len(pending)==16:flush()
        flush();loss_tables[direction]=table
        timings[direction]={'candidate_evaluations':int(np.isfinite(table).sum()),'forward_batches':calls,'transform_and_tensor_seconds':transform_seconds,
                            'inference_and_transfer_seconds':inference_seconds,'total_seconds':time.perf_counter()-start}
        np.save(out/f'actual_ce_direction{direction}.npy',table)
        atomic(out/'progress.json',{'stage':'exact_loss_tables','direction_complete':direction,'timings':timings[direction]})
    summaries={};rng=np.random.default_rng(991);unique=sorted(set(patients))
    for direction,table in loss_tables.items():
        losses={name:table[np.arange(160),perm] for name,perm in policies.items()};assert all(np.isfinite(v).all() for v in losses.values())
        random_means=np.array([losses[f'random_{r:02}'].mean() for r in range(20)])
        random_per_image=np.mean([losses[f'random_{r:02}'] for r in range(20)],axis=0);delta=losses['ranked']-random_per_image
        boot=[]
        for _ in range(1000):
            ids=np.concatenate([np.flatnonzero(patients==p) for p in rng.choice(unique,len(unique),replace=True)])
            boot.append(float(delta[ids].mean()))
        summaries[direction]={'policy_mean_actual_ce':{name:float(v.mean()) for name,v in losses.items()},
             'random_mean_ce':float(random_means.mean()),'random_assignment_sd':float(random_means.std(ddof=1)),
             'ranked_minus_random_mean_ce':float(delta.mean()),'patient_cluster_descriptive_interval':np.quantile(boot,[.025,.975]).tolist(),
             'random_assignments_at_least_as_high_as_ranked':int((random_means>=losses['ranked'].mean()).sum()),
             'ranked_minus_random_by_class':{c:float(delta[classes==c].mean()) for c in NAMES}}
    assert digest(path)==checkpoint_hash
    result={'stage':'complete','summaries':summaries,'timings':timings,'elapsed_seconds':time.perf_counter()-start_all,
            'all_class_batch_strength_multisets_exact':True,'singleton_groups':sum(len(g)==1 for g in groups),'total_groups':len(groups),
            'reassignable_image_fraction':sum(len(g) for g in groups if len(g)>1)/160,
            'primary_engineering_gate_passed':summaries[0]['ranked_minus_random_mean_ce']>0,
            'checkpoint_unchanged':True,'training_started':False,'validation_or_test_pixels_read':False,
            'interpretation':'Higher CE indicates a harder adversarial challenge on this frozen model, not improved classifier accuracy or clinical benefit. Rank assignment is exact for its linear surrogate, not guaranteed optimal for nonlinear CE.'}
    atomic(out/'RESULTS.json',result);atomic(out/'progress.json',result)
    lines=['# Fixed-strength assignment control','',result['interpretation'],'',
           '| Stain direction | Ranked CE | Random mean CE | Difference | Reverse-ranked CE |','|---|---:|---:|---:|---:|']
    for d,s in summaries.items():lines.append(f"|{'Same as probe' if d==0 else 'Independent'}|{s['policy_mean_actual_ce']['ranked']:.5f}|{s['random_mean_ce']:.5f}|{s['ranked_minus_random_mean_ce']:+.5f}|{s['policy_mean_actual_ce']['reverse_ranked']:.5f}|")
    lines+=['',f"Exact class/batch strength multiset checks passed for all23 policies. Reassignable images: {result['reassignable_image_fraction']:.1%}. Primary engineering gate passed: {result['primary_engineering_gate_passed']}.",
            '', 'All160 images are reused training-only development patches (32/class); there are20 random assignments, not20 independent datasets. The source checkpoint historically used validation early stopping. The small balanced diagnostic sample and synthetic batches differ from natural imbalanced training batches. No geometric/RGB jitter was applied. Paired image losses share patients; cluster intervals are descriptive and condition on this model, sample and fixed groups.',
            '', 'The exact candidate table amortizes repeated policy scoring; its timing does not measure an online training policy. Extra probe passes, CPU reconstruction and assignment overhead must be measured during a subsequent controlled pilot. No retraining or validation/test image access occurred. This check cannot establish novelty or downstream benefit.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(result,indent=2),flush=True);print('REPORT',out,flush=True)

if __name__=='__main__':main()
