"""Recompute frozen test metrics/bootstrap with NumPy only; no GPU/data pixels.

Run this copy inside evaluation_package/publication_results, or pass that folder
as the first argument. Model weights are not distributed; embedded hashes bind
these prediction files to the recorded model/manifest identities.
"""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
def metric(cm):
    tp=cm.diagonal();support=cm.sum(1);pred=cm.sum(0)
    return float(np.divide(2*tp,support+pred,out=np.zeros(5),where=support+pred>0).mean())
def main():
    root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
    result=json.loads((root/'final_test_RESULTS.json').read_text())
    freeze=json.loads((root/'final_test_CHECKPOINT_FREEZE.json').read_text())
    manifest=root/'final_test_patch_manifest.csv'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==result['patch_manifest_sha256']
    with manifest.open(newline='') as f:rows=list(csv.DictReader(f))
    truth=np.array([int(r['class_id']) for r in rows]);patients=np.array([r['patient_id'] for r in rows]);unique=sorted(set(patients))
    by={(r['seed'],r['arm']):r for r in result['per_seed']};clusters={}
    for model in freeze['checkpoints']:
        seed,arm=model['seed'],model['arm']
        with np.load(root/'test_predictions'/f'seed{seed}_{arm}_predictions.npz') as file:
            assert np.array_equal(file['labels'],truth)
            assert str(file['checkpoint_sha256'])==model['sha256'] and str(file['patch_manifest_sha256'])==result['patch_manifest_sha256']
            p=file['probabilities'];assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5);guess=p.argmax(1)
        cm=np.bincount(truth*5+guess,minlength=25).reshape(5,5);r=by[(seed,arm)]
        assert np.array_equal(cm,r['confusion_matrix']) and abs(metric(cm)-r['macro_f1'])<1e-12
        assert abs(cm.diagonal().sum()/cm.sum()-r['accuracy'])<1e-12
        if arm in ['adaptive','within_class_shuffled']:
            clusters[(seed,arm)]=np.stack([np.bincount(truth[patients==u]*5+guess[patients==u],minlength=25).reshape(5,5) for u in unique])
    rng=np.random.default_rng(20261117);values=[];missing=0
    for _ in range(1000):
        selected=rng.choice(len(unique),len(unique),replace=True);paired=[]
        for seed in [42,43,44]:
            a=clusters[(seed,'adaptive')][selected].sum(0);b=clusters[(seed,'within_class_shuffled')][selected].sum(0)
            paired.append(metric(a)-metric(b))
        values.append(float(np.mean(paired)));missing+=int((a.sum(1)==0).any())
    assert missing==result['bootstrap_draws_missing_classes'] and np.allclose(np.quantile(values,[.025,.975]),result['descriptive_patient_cluster_interval'],rtol=0,atol=1e-12)
    print(json.dumps({'passed':True,'models':18,'patches':len(rows),'patients':len(unique),'missing_class_draws':missing,
       'descriptive_patient_interval':np.quantile(values,[.025,.975]).tolist(),'no_pixels_or_gpu_required':True},indent=2))
if __name__=='__main__':main()
