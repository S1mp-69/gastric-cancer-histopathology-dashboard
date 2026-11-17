"""Complete the explicitly frozen final test and refresh portable materials.

No model selection/tuning after reserved-test acquisition. No automatic new runs.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import csv,json,sys,subprocess,time,traceback,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import classification_report,confusion_matrix,f1_score,accuracy_score
from hiesd_pilot import ROOT,NAMES,digest
from hiesd_resnet18 import GastricCancerRNN,atomic
from hiesd_final_test_extract import acquire,OUT,PORTABLE

LOCK=ROOT/'stain_research/runs/final_test_hiesd_v1'
def read_csv(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

class TestImages(Dataset):
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        with Image.open(ROOT/self.rows[i]['image_path']) as im:image=im.convert('RGB')
        return F.normalize(F.to_tensor(image),[.485,.456,.406],[.229,.224,.225]),int(self.rows[i]['class_id'])

def main():
    LOCK.mkdir(parents=True,exist_ok=True);PORTABLE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    protocol_path=LOCK/'PROTOCOL.json';assert digest(protocol_path)==(LOCK/'PROTOCOL.sha256').read_text().strip()
    protocol=json.loads(protocol_path.read_text())
    from publication_control_audit import main as audit_controls
    audit_controls()
    assert digest(ROOT/'data/external/hiesd_metadata/prospective_grouped_manifest.csv')==protocol['manifest_sha256']
    within=ROOT/'stain_research/runs/hiesd_within_class_v1';cam=ROOT/'stain_research/runs/hiesd_resnet_ablation_v1'
    base=ROOT/'stain_research/runs/hiesd_resnet18_seed42_v3'
    for seed in protocol['seeds']:
        r=json.loads((within/f'within_seed{seed}/RESULTS.json').read_text());assert r['stage']=='complete' and r['reload_predictions_match']
    runs=[]
    for seed in protocol['seeds']:
        for arm in protocol['models']:
            p=within/f'within_seed{seed}' if arm=='within_class_shuffled' else (base if seed==42 and arm=='standard' else cam/f'seed{seed}_{arm}')
            r=json.loads((p/'RESULTS.json').read_text());assert digest(p/'best.pt')==r['best_checkpoint_sha256'] and r['reload_predictions_match']
            state=torch.load(p/'best.pt',map_location='cpu',weights_only=False)
            assert all(torch.isfinite(t).all() for t in state['model_state_dict'].values());del state
            runs.append({'seed':seed,'arm':arm,'checkpoint':str(p/'best.pt'),'sha256':r['best_checkpoint_sha256']})
    freeze={'protocol_sha256':digest(protocol_path),'checkpoints':runs,
            'implementation_sha256':{n:digest(ROOT/'stain_research'/n) for n in ['publication_finalize_pipeline.py','hiesd_final_test_extract.py','hiesd_expand_pilot.py','hiesd_pilot.py','hiesd_expand_v3.py','verify_final_test.py']}}
    if (LOCK/'CHECKPOINT_FREEZE.json').exists():assert json.loads((LOCK/'CHECKPOINT_FREEZE.json').read_text())==freeze
    else:atomic(LOCK/'CHECKPOINT_FREEZE.json',freeze)
    atomic(LOCK/'STATUS.json',{'stage':'final_test_acquisition','slides':len(protocol['selected']),'GiB':protocol['raw_bytes']/1024**3})
    if not (OUT/'patch_manifest.csv').exists() or not (PORTABLE/'QC_REPORT.json').exists():
        if not acquire(protocol):raise RuntimeError('A final-test class is absent; five-class confirmation stopped under frozen protocol')
    patch_manifest_sha=digest(OUT/'patch_manifest.csv')
    assert patch_manifest_sha==json.loads((PORTABLE/'QC_REPORT.json').read_text())['manifest_sha256']
    rows=read_csv(OUT/'patch_manifest.csv');mapping={r['slide_id']:r for r in read_csv(ROOT/'data/external/hiesd_metadata/prospective_grouped_manifest.csv')}
    assert all(r['split']=='reserved_test' and mapping[r['slide_id']]['split']=='reserved_test' and mapping[r['slide_id']]['patient_id']==r['patient_id'] for r in rows)
    assert set(int(r['class_id']) for r in rows)==set(range(5))
    loader=DataLoader(TestImages(rows),batch_size=16,shuffle=False,num_workers=0,pin_memory=True)
    torch.set_num_threads(2);model=GastricCancerRNN(5).cuda();evaluations=[];predictions={};labels=np.array([int(r['class_id']) for r in rows])
    for n,run in enumerate(runs):
        atomic(LOCK/'STATUS.json',{'stage':'final_test_inference','completed_models':n,'models':18,'seed':run['seed'],'arm':run['arm']})
        name=f"seed{run['seed']}_{run['arm']}";saved=PORTABLE/f'{name}_predictions.npz'
        assert digest(Path(run['checkpoint']))==run['sha256']
        if saved.exists():
            data=np.load(saved);prob=data['probabilities'];assert np.array_equal(data['labels'],labels)
            assert str(data['patch_manifest_sha256'])==patch_manifest_sha and str(data['checkpoint_sha256'])==run['sha256']
        else:
            state=torch.load(run['checkpoint'],map_location='cpu',weights_only=False);model.load_state_dict(state['model_state_dict']);del state;model.eval();chunks=[]
            with torch.inference_mode():
                for x,y in loader:
                    with torch.autocast('cuda',dtype=torch.float16):logits,_=model(x.cuda(non_blocking=True))
                    chunks.append(logits.float().softmax(1).cpu().numpy())
            prob=np.concatenate(chunks)
            temporary=saved.with_suffix('.tmp.npz')
            np.savez_compressed(temporary,probabilities=prob,labels=labels,patch_manifest_sha256=patch_manifest_sha,checkpoint_sha256=run['sha256'])
            temporary.replace(saved)
        assert np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-5)
        guess=prob.argmax(1);confidence=prob.max(1);correct=guess==labels;ece=0.
        for b in range(15):
            ids=(confidence>=b/15)&(confidence<=(b+1)/15 if b==14 else confidence<(b+1)/15)
            if ids.any():ece+=ids.mean()*abs(correct[ids].mean()-confidence[ids].mean())
        report=classification_report(labels,guess,labels=list(range(5)),target_names=NAMES,output_dict=True,zero_division=0)
        evaluations.append({'seed':run['seed'],'arm':run['arm'],'accuracy':float(accuracy_score(labels,guess)),
              'macro_f1':float(f1_score(labels,guess,labels=list(range(5)),average='macro',zero_division=0)),
              'balanced_accuracy':float(np.mean([report[c]['recall'] for c in NAMES])),'ece_15':float(ece),
              'brier':float(((prob-np.eye(5)[labels])**2).sum(1).mean()),'per_class':report,'confusion_matrix':confusion_matrix(labels,guess,labels=list(range(5))).tolist()})
        predictions[(run['seed'],run['arm'])]=guess
    patients=np.array([r['patient_id'] for r in rows]);unique=sorted(set(patients));rng=np.random.default_rng(20261117)
    samples=[];missing=0
    for _ in range(1000):
        ids=np.concatenate([np.flatnonzero(patients==p) for p in rng.choice(unique,len(unique),replace=True)])
        missing+=int(len(set(labels[ids]))<5)
        delta=[]
        for seed in [42,43,44]:
            delta.append(f1_score(labels[ids],predictions[(seed,'adaptive')][ids],labels=list(range(5)),average='macro',zero_division=0)-f1_score(labels[ids],predictions[(seed,'within_class_shuffled')][ids],labels=list(range(5)),average='macro',zero_division=0))
        samples.append(float(np.mean(delta)))
    means={a:{k:float(np.mean([r[k] for r in evaluations if r['arm']==a])) for k in ['accuracy','macro_f1','balanced_accuracy','ece_15','brier']} for a in protocol['models']}
    result={'stage':'inference_complete','patches':len(rows),'patients':len(unique),'class_counts':{c:int((labels==i).sum()) for i,c in enumerate(NAMES)},
            'per_seed':evaluations,'arm_means':means,'primary_adaptive_minus_within_class':means['adaptive']['macro_f1']-means['within_class_shuffled']['macro_f1'],
            'descriptive_patient_cluster_interval':np.quantile(samples,[.025,.975]).tolist(),'bootstrap_draws_missing_classes':missing,
            'bootstrap_convention':'Fixed five classes, zero_division=0 even if a class is absent in a draw; small-cluster interval is descriptive and may be unstable.',
            'protocol_sha256':digest(protocol_path),'checkpoint_freeze_sha256':digest(LOCK/'CHECKPOINT_FREEZE.json'),'patch_manifest_sha256':patch_manifest_sha,
            'scope':'Frozen final test on reserved public-release patients; not independent cross-hospital external testing, not clinical validation. No tuning after test access.'}
    atomic(PORTABLE/'RESULTS.json',result)
    from verify_final_test import main as verify_final
    verify_final()
    result['stage']='complete';result['independent_verification']=True;atomic(PORTABLE/'RESULTS.json',result)
    atomic(LOCK/'STATUS.json',{'stage':'building_publication_materials','test_complete':True})
    builder=ROOT/'stain_research/refresh_publication_package.py'
    if builder.exists():subprocess.run([sys.executable,'-u',str(builder)],cwd=ROOT,check=True)
    atomic(LOCK/'STATUS.json',{'stage':'complete','test_complete':True,'materials_built':builder.exists()})

if __name__=='__main__':
    try:main()
    except Exception as exc:
        atomic(LOCK/'FAILURE.json',{'error':str(exc),'traceback':traceback.format_exc()})
        builder=ROOT/'stain_research/refresh_publication_package.py'
        if builder.exists():subprocess.run([sys.executable,'-u',str(builder)],cwd=ROOT)
        raise
