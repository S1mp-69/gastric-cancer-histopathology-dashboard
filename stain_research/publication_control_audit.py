"""Verify the added conditional-permutation runs on CPU before reserved test."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import csv,json,gc
import numpy as np,torch
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
from hiesd_pilot import ROOT,NAMES,digest
from hiesd_resnet18 import atomic
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    torch.set_num_threads(2)
    out=ROOT/'evaluation_package/publication_readiness/publication_audit_20261117'
    cam=ROOT/'stain_research/runs/hiesd_resnet_ablation_v1';base=ROOT/'stain_research/runs/hiesd_resnet18_seed42_v3'
    assert load(out/'AUDIT.json')['passed']
    reference=load(base/'config.json')
    with (base/'validation_predictions.csv').open(newline='') as f:identity=[(r['image_path'],r['patient_id'],r['class_id']) for r in csv.DictReader(f)]
    records=[]
    for seed in [42,43,44]:
        p=ROOT/f'stain_research/runs/hiesd_within_class_v1/within_seed{seed}'
        config=load(p/'config.json');result=load(p/'RESULTS.json');history=load(p/'history.json');provenance=load(p/'wrapper_provenance.json')
        assert result['stage']=='complete' and result['reload_predictions_match'] and not result['reserved_test_used']
        assert config['arm']=='within_class_shuffled' and config['seed']==seed
        for k in reference:
            if k not in ['seed','arm','source_sha256']:assert config[k]==reference[k],k
        for k,name in [('trainer_sha256','hiesd_ablation_train.py'),('data_sha256','hiesd_ablation_data.py'),('wrapper_sha256','hiesd_within_class_train.py')]:assert provenance[k]==digest(ROOT/'stain_research'/name)
        assert digest(p/'initial.pt')==digest(cam/f'initial_seed{seed}.pt')
        expected=-1.;bad=0;best=None
        for h in history:
            if h['validation']['macro_f1']>expected+.0001:expected=h['validation']['macro_f1'];best=h['epoch'];bad=0
            else:bad+=1
        assert len(history)==result['completed_epochs']<=40 and (len(history)==40 or bad==7)
        assert best==result['best_epoch'] and digest(p/'best.pt')==result['best_checkpoint_sha256']
        for name in ['best.pt','last.pt']:
            state=torch.load(p/name,map_location='cpu',weights_only=False)
            assert all(torch.isfinite(t).all() for t in state['model_state_dict'].values())
            assert state['epoch']==(best if name=='best.pt' else len(history))
            for key in ['optimizer_state_dict','scheduler_state_dict','scaler_state_dict','torch_rng_state','cuda_rng_state','python_rng_state','numpy_rng_state']:assert key in state
            del state;gc.collect()
        with (p/'validation_predictions.csv').open(newline='') as f:rows=list(csv.DictReader(f))
        assert [(r['image_path'],r['patient_id'],r['class_id']) for r in rows]==identity
        prob=np.array([[float(r['p_'+c]) for c in NAMES] for r in rows]);truth=np.array([int(r['class_id']) for r in rows]);guess=prob.argmax(1)
        assert np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-5)
        assert abs(accuracy_score(truth,guess)-result['validation']['accuracy'])<1e-12
        assert abs(f1_score(truth,guess,average='macro')-result['validation']['macro_f1'])<1e-12
        assert np.array_equal(confusion_matrix(truth,guess),result['validation']['confusion_matrix'])
        report=classification_report(truth,guess,labels=list(range(5)),target_names=NAMES,output_dict=True,zero_division=0)
        for c in NAMES:
            for k in ['precision','recall','f1-score','support']:assert abs(report[c][k]-result['validation']['classification_report'][c][k])<1e-12
        records.append({'seed':seed,'epochs':len(history),'best_epoch':best,'macro_f1':expected,'passed':True,'best_sha256':result['best_checkpoint_sha256']})
    atomic(out/'WITHIN_CLASS_AUDIT.json',{'passed':True,'runs':records,'scope':'CPU-only source/config/initialization/stopping/finite/resume/prediction audit; no reserved pixels','multiset_checks':load(ROOT/'stain_research/runs/hiesd_within_class_v1/protocol.json')['within_class_multiset_checks']})
    print('PASS: all three within-class controls audited.',flush=True)
if __name__=='__main__':main()
