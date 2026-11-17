"""Finite local queue; no AI calls and no result-dependent hyperparameter search."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import argparse,csv,json,shutil,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from hiesd_pilot import ROOT,NAMES,digest
from hiesd_resnet18 import atomic,Images as OriginalImages,GastricCancerRNN
from hiesd_ablation_data import configure,Images,CAM
from publication_controls import assigned_strengths
from hiesd_ablation_report import summarize,ARMS,BASE

def preflight():
    torch.set_num_threads(2)
    with (ROOT/'data/external/hiesd_expanded_v3/patch_manifest.csv').open() as stream:records=list(csv.DictReader(stream))
    train=[r for r in records if r['split']=='train'];val=[r for r in records if r['split']=='validation']
    c=json.loads((BASE/'config.json').read_text());mean=c['normalization_mean'];std=c['normalization_std']
    configure('standard',42);cached=Images(train,mean,std,True);original=OriginalImages(train,mean,std,True)
    checked=0
    for epoch in [0,1,7]:
        cached.epoch=original.epoch=epoch
        for i in np.linspace(0,len(train)-1,20,dtype=int):assert torch.equal(cached[int(i)][0],original[int(i)][0]);checked+=1
    cv=Images(val,mean,std,False);ov=OriginalImages(val,mean,std,False)
    for i in range(len(val)):assert torch.equal(cv[i][0],ov[i][0])
    for seed in [42,43,44]:
        for epoch in range(1,41):
            p=assigned_strengths(cached.alpha,cached.valid,cached.codes,seed,epoch)
            assert np.array_equal(np.sort(p[cached.valid]),np.sort(cached.alpha[cached.valid]))
            assert (p[~cached.valid]==0).all()
    for arm in ARMS:
        configure(arm,42);d=Images(train,mean,std,True);d.epoch=1
        for i in np.linspace(0,len(train)-1,5,dtype=int):assert torch.isfinite(d[i][0]).all()
    config={'arms':ARMS,'seeds':[42,43,44],'max_epochs':40,'early_stopping_patience':7,'start_budget_hours':6,
            'primary_comparison':'Adaptive minus global shuffled assignment; exact valid-training alpha multiset each epoch.',
            'seed42_standard':'Reuse completed frozen baseline. Four missing seed42 arms first; then whole five-arm seed43/44 blocks where start budget permits.',
            'source_hashes':{n:digest(ROOT/'stain_research'/n) for n in ['hiesd_ablation_data.py','hiesd_ablation_train.py','hiesd_ablation_report.py','hiesd_ablation_supervisor.py','efficient_cache.py','controller.py','transforms.py','publication_controls.py']},
            'manifest_sha256':digest(ROOT/'data/external/hiesd_expanded_v3/patch_manifest.csv'),
            'initial_seed42_sha256':digest(BASE/'initial.pt'),'gpu_policy':'Reuse verified CUDA environment; no routine GPU probes or smoke reruns.',
            'fallbacks':int((~cached.valid).sum()),'uniform_strength':cached.params['uniform_strength'],
            'scope':'Exploratory follow-up with validation early stopping; no test inference, data re-selection or dashboard activation.'}
    p=CAM/'campaign_config.json'
    if p.exists():assert json.loads(p.read_text())==config
    else:atomic(p,config)
    for seed in [42,43,44]:
        target=CAM/f'initial_seed{seed}.pt'
        if target.exists():continue
        if seed==42:shutil.copy2(BASE/'initial.pt',target)
        else:
            torch.manual_seed(seed);model=GastricCancerRNN(5)
            torch.save({'model_state_dict':model.state_dict()},target)
    atomic(CAM/'preflight.json',{'gpu_check_performed':False,'standard_training_tensor_equivalence_cases':checked,
           'deterministic_validation_tensor_equivalence_cases':len(val),'strength_multiset_seed_epoch_checks':120,
           'finite_cpu_outputs_all_arms':True,'cache_equivalence':json.loads((CAM/'cache_equivalence.json').read_text()),
           'initial_checkpoint_sha256':{str(s):digest(CAM/f'initial_seed{s}.pt') for s in [42,43,44]}})
    print('CPU cache/input/control checks passed. No GPU checks run.',flush=True)

def main():
    preflight();started=time.monotonic();completed=summarize()
    for seed in [42,43,44]:
        pending=[a for a in ARMS if not (seed==42 and a=='standard') and not (CAM/f'seed{seed}_{a}/RESULTS.json').exists()]
        if seed!=42 and pending:
            observed=[]
            for result in CAM.glob('seed42_*/history.json'):
                h=json.loads(result.read_text());observed += [r['epoch_seconds'] for r in h]
            predicted_block=len(pending)*40*(max(observed) if observed else 45)
            if time.monotonic()-started+predicted_block>6*3600:
                atomic(CAM/'STATUS.json',{'stage':'budget_complete','completed_runs':len(completed),'deferred_seed':seed,'reason':'Insufficient six-hour start budget for a whole matched seed block'});return
        for arm in pending:
            run=CAM/f'seed{seed}_{arm}';run.mkdir(exist_ok=True)
            # Reuse the existing measured baseline timing as a reference only;
            # actual augmentation throughput is recorded by every training epoch.
            bench=json.loads((BASE/'benchmark.json').read_text());bench['scope']='Inherited reference timing from frozen baseline; no new GPU benchmark; actual stain-arm throughput saved in history.json.'
            atomic(run/'benchmark.json',bench)
            atomic(CAM/'STATUS.json',{'stage':'training','seed':seed,'arm':arm,'run':str(run),'completed_runs':len(completed),'max_runs':15})
            command=[sys.executable,'-u',str(ROOT/'stain_research/hiesd_ablation_train.py'),'--arm',arm,'--seed',str(seed),'--run',str(run)]
            if (run/'last.pt').exists():command+=['--resume']
            with (run/'training.log').open('a') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
            completed=summarize()
    atomic(CAM/'STATUS.json',{'stage':'complete','completed_runs':len(completed),'elapsed_seconds':time.monotonic()-started})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    try:
        if (CAM/'FAILURE.json').exists():(CAM/'FAILURE.json').rename(CAM/f'FAILURE_history_{time.time_ns()}.json')
        if args.preflight_only:preflight()
        else:main()
    except Exception as exc:
        atomic(CAM/'FAILURE.json',{'error':str(exc),'traceback':traceback.format_exc()});raise
