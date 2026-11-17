"""CPU-only provenance, checkpoint and prediction audit before publication."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import csv,json,gc
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from hiesd_pilot import ROOT,digest
from hiesd_resnet18 import atomic

def main():
    torch.set_num_threads(2)
    out=ROOT/'evaluation_package/publication_readiness/publication_audit_20261117';out.mkdir(parents=True,exist_ok=True)
    cam=ROOT/'stain_research/runs/hiesd_resnet_ablation_v1';base=ROOT/'stain_research/runs/hiesd_resnet18_seed42_v3'
    cfg=json.loads((cam/'campaign_config.json').read_text());reference=json.loads((base/'config.json').read_text())
    for n,h in cfg['source_hashes'].items():assert digest(ROOT/'stain_research'/n)==h,n
    manifest=ROOT/'data/external/hiesd_expanded_v3/patch_manifest.csv';assert digest(manifest)==cfg['manifest_sha256']
    with manifest.open() as f:rows=list(csv.DictReader(f))
    tr=[r for r in rows if r['split']=='train'];va=[r for r in rows if r['split']=='validation']
    assert len(tr)==2543 and len(va)==1251 and not {r['patient_id'] for r in tr}&{r['patient_id'] for r in va}
    paths={s:{str((ROOT/r['image_path']).resolve()) for r in rows if r['split']==s} for s in ['train','validation']}
    directory=Path(json.loads((cam/'cache_reference.json').read_text())['directory'])
    with (directory/'index.csv').open() as f:index=list(csv.DictReader(f))
    for split,label in [('train','train'),('validation','val')]:
        assert paths[split]=={str(Path(r['image_path']).resolve()) for r in index if r['split']==label}
    assert all(digest(Path(r['image_path']))==r['sha256'] for r in index)
    assert len({r['sha256'] for r in index})==len(index)
    valid=np.load(directory/'valid.npy',mmap_mode='r');features=np.load(directory/'diagnostics.npy',mmap_mode='r')
    params=json.loads((directory/'train_parameters.json').read_text());mask=np.array([r['split']=='train' for r in index])&valid
    assert np.array_equal(np.quantile(features[mask],[.1,.9],axis=0),np.array([params['controller']['low'],params['controller']['high']]))
    runs=[];identity=None
    for seed in [42,43,44]:
        for arm in ['standard','residual_fixed','uniform_weaker','shuffled','adaptive']:
            p=base if seed==42 and arm=='standard' else cam/f'seed{seed}_{arm}'
            c=json.loads((p/'config.json').read_text());r=json.loads((p/'RESULTS.json').read_text());h=json.loads((p/'history.json').read_text())
            for k in reference:
                if k not in ['arm','seed','source_sha256']:assert c[k]==reference[k],(p.name,k)
            assert c['seed']==seed and r['stage']=='complete' and r['reload_predictions_match'] and not r['reserved_test_used']
            assert digest(p/'initial.pt')==digest(cam/f'initial_seed{seed}.pt')
            assert digest(p/'best.pt')==r['best_checkpoint_sha256'] and len(h)==r['completed_epochs']<=40
            expected=-1.;best_epoch=None;bad=0
            for item in h:
                if item['validation']['macro_f1']>expected+.0001:expected=item['validation']['macro_f1'];best_epoch=item['epoch'];bad=0
                else:bad+=1
            assert best_epoch==r['best_epoch'] and (len(h)==40 or bad==7)
            for name in ['best.pt','last.pt']:
                state=torch.load(p/name,map_location='cpu',weights_only=False)
                assert all(torch.isfinite(v).all() for v in state['model_state_dict'].values())
                for key in ['optimizer_state_dict','scheduler_state_dict','scaler_state_dict','torch_rng_state','cuda_rng_state','python_rng_state','numpy_rng_state']:assert key in state
                assert state['epoch']==(best_epoch if name=='best.pt' else len(h));del state;gc.collect()
            with (p/'validation_predictions.csv').open() as f:pred=list(csv.DictReader(f))
            ids=[(a['image_path'],a['patient_id'],a['class_id']) for a in pred]
            if identity is None:identity=ids
            else:assert identity==ids
            prob=np.array([[float(v) for k,v in a.items() if k.startswith('p_')] for a in pred]);y=np.array([int(a['class_id']) for a in pred]);guess=prob.argmax(1)
            assert np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-5)
            assert abs(accuracy_score(y,guess)-r['validation']['accuracy'])<1e-12
            assert abs(f1_score(y,guess,average='macro')-r['validation']['macro_f1'])<1e-12
            assert np.array_equal(confusion_matrix(y,guess),r['validation']['confusion_matrix'])
            runs.append({'seed':seed,'arm':arm,'epochs':len(h),'best_epoch':best_epoch,'passed':True})
    protected=json.loads((cam/'baseline_preservation.json').read_text());assert all(digest(base/n)==h for n,h in protected.items())
    atomic(out/'AUDIT.json',{'passed':True,'runs':runs,'cache_images_verified':len(index),'train_only_quantiles_verified':True,'initialization_config_stopping_and_checkpoint_checks':True,
           'same_validation_order':True,'finite_best_and_last_model_tensors':True,'resume_state_keys_present':True,'reserved_test_accessed':False,
           'original_baseline_preserved':True,'limitations':['No new inference or bitwise interrupted-vs-uninterrupted equivalence proof','No expert pathology label audit','No near-duplicate assessment beyond exact file hashes']})
    print('PASS:15 HiESD runs; cache hashes, splits, initializations, configs, stopping, checkpoints and predictions audited.',flush=True)

if __name__=='__main__':main()
