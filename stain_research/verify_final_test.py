"""Independent CPU metric/identity check from saved final-test probabilities."""
import csv,json
from pathlib import Path
import numpy as np
from hiesd_pilot import ROOT,NAMES,digest
from hiesd_resnet18 import atomic
def scores(cm):
    tp=cm.diagonal();support=cm.sum(1);pred=cm.sum(0)
    f1=np.divide(2*tp,support+pred,out=np.zeros(5),where=support+pred>0)
    recall=np.divide(tp,support,out=np.zeros(5),where=support>0)
    return {'macro_f1':float(f1.mean()),'accuracy':float(tp.sum()/cm.sum()),'balanced_accuracy':float(recall.mean())}
def main():
    result_dir=ROOT/'stain_research/results/final_test_hiesd_v1';lock=ROOT/'stain_research/runs/final_test_hiesd_v1'
    r=json.loads((result_dir/'RESULTS.json').read_text());freeze=json.loads((lock/'CHECKPOINT_FREEZE.json').read_text())
    assert r['protocol_sha256']==digest(lock/'PROTOCOL.json') and r['checkpoint_freeze_sha256']==digest(lock/'CHECKPOINT_FREEZE.json')
    for name,h in freeze['implementation_sha256'].items():assert digest(ROOT/'stain_research'/name)==h
    with (ROOT/'data/external/hiesd_final_test_v1/patch_manifest.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    with (ROOT/'data/external/hiesd_metadata/prospective_grouped_manifest.csv').open(newline='') as f:mapping={x['slide_id']:x for x in csv.DictReader(f)}
    assert all(mapping[x['slide_id']]['split']=='reserved_test' and x['patient_id']==mapping[x['slide_id']]['patient_id'] for x in rows)
    assert digest(ROOT/'data/external/hiesd_final_test_v1/patch_manifest.csv')==r['patch_manifest_sha256']
    truth=np.array([int(x['class_id']) for x in rows]);patients=np.array([x['patient_id'] for x in rows]);unique=sorted(set(patients))
    saved_metrics={(x['seed'],x['arm']):x for x in r['per_seed']};by_patient={};checked=0;maximum=0.
    for model in freeze['checkpoints']:
        seed,arm=model['seed'],model['arm'];data=np.load(result_dir/f'seed{seed}_{arm}_predictions.npz')
        assert np.array_equal(data['labels'],truth) and str(data['checkpoint_sha256'])==model['sha256']
        assert str(data['patch_manifest_sha256'])==r['patch_manifest_sha256']
        probability=data['probabilities'];assert probability.shape==(len(rows),5) and np.isfinite(probability).all() and np.allclose(probability.sum(1),1,atol=1e-5)
        guess=probability.argmax(1);cm=np.bincount(truth*5+guess,minlength=25).reshape(5,5)
        actual=saved_metrics[(seed,arm)];assert np.array_equal(cm,actual['confusion_matrix'])
        for key,value in scores(cm).items():maximum=max(maximum,abs(value-actual[key]));assert abs(value-actual[key])<1e-12
        confidence=probability.max(1);bins=np.minimum((confidence*15).astype(int),14);ece=0.
        for b in range(15):
            selected=bins==b
            if selected.any():ece+=selected.mean()*abs((guess[selected]==truth[selected]).mean()-confidence[selected].mean())
        brier=float((np.square(probability).sum(1)-2*probability[np.arange(len(truth)),truth]+1).mean())
        assert abs(ece-actual['ece_15'])<1e-6 and abs(brier-actual['brier'])<1e-6
        for i,c in enumerate(NAMES):
            tp=cm[i,i];support=cm[i].sum();predicted=cm[:,i].sum()
            vals={'support':support,'precision':tp/predicted if predicted else 0.,'recall':tp/support if support else 0.,'f1-score':2*tp/(support+predicted) if support+predicted else 0.}
            for key,value in vals.items():assert abs(value-actual['per_class'][c][key])<1e-12
        if arm in ['adaptive','within_class_shuffled']:
            by_patient[(seed,arm)]=[np.bincount(truth[patients==p]*5+guess[patients==p],minlength=25).reshape(5,5) for p in unique]
        checked+=1
    # Independently sum per-patient confusion matrices with bootstrap multiplicity
    # instead of rebuilding sklearn input arrays for each resample.
    rng=np.random.default_rng(20261117);deltas=[];missing=0
    for _ in range(1000):
        selected=rng.choice(len(unique),len(unique),replace=True);paired=[]
        for seed in [42,43,44]:
            a=np.stack(by_patient[(seed,'adaptive')])[selected].sum(0);b=np.stack(by_patient[(seed,'within_class_shuffled')])[selected].sum(0)
            paired.append(scores(a)['macro_f1']-scores(b)['macro_f1'])
        missing+=int((a.sum(1)==0).any());deltas.append(float(np.mean(paired)))
    assert missing==r['bootstrap_draws_missing_classes']
    assert np.allclose(np.quantile(deltas,[.025,.975]),r['descriptive_patient_cluster_interval'],rtol=0,atol=1e-12)
    receipt={'passed':True,'models':checked,'patches':len(rows),'patients':len(unique),'maximum_metric_error':maximum,
       'identity_and_probability_checks':True,'all_class_metrics_verified':True,'independent_patient_confusion_bootstrap_verified':True,
       'scope':'No new inference; small-patient interval is descriptive and conditions on the three trained seeds.'}
    atomic(result_dir/'INDEPENDENT_VERIFICATION.json',receipt);print('PASS: frozen final-test metrics and patient bootstrap independently verified.',flush=True)
if __name__=='__main__':main()
