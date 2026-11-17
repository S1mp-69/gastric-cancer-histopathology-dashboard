"""Three bounded missing controls, then a frozen final-test workflow if prepared."""
import json,subprocess,sys,time,shutil
from pathlib import Path
import numpy as np
from hiesd_pilot import ROOT,digest
from hiesd_resnet18 import atomic
from hiesd_ablation_data import Images,configure
from publication_controls import assigned_strengths
import csv
CAM=ROOT/'stain_research/runs/hiesd_within_class_v1'

def main():
    CAM.mkdir(parents=True,exist_ok=True)
    assert json.loads((ROOT/'evaluation_package/publication_readiness/publication_audit_20261117/AUDIT.json').read_text())['passed']
    with (ROOT/'data/external/hiesd_expanded_v3/patch_manifest.csv').open() as f:rows=list(csv.DictReader(f))
    c=json.loads((ROOT/'stain_research/runs/hiesd_resnet18_seed42_v3/config.json').read_text())
    configure('shuffled',42);d=Images([r for r in rows if r['split']=='train'],c['normalization_mean'],c['normalization_std'],True)
    for seed in [42,43,44]:
        for epoch in range(1,41):
            a=assigned_strengths(d.alpha,d.valid,d.codes,seed,epoch,True)
            for cls in set(d.codes):
                ids=(d.codes==cls)&d.valid;assert np.array_equal(np.sort(a[ids]),np.sort(d.alpha[ids]))
    atomic(CAM/'protocol.json',{'seeds':[42,43,44],'max_epochs':40,'early_stopping_patience':7,'within_class_multiset_checks':120,
          'same_initial_weights_and_standard_pipeline':True,'validation_unmodified':True,'test_accessed':False,
          'scope':'Post-hoc falsification control; no parameters tuned and no repeated GPU check.'})
    for seed in [42,43,44]:
        run=CAM/f'within_seed{seed}';run.mkdir(exist_ok=True)
        if (run/'RESULTS.json').exists():continue
        atomic(run/'benchmark.json',{'scope':'Reuse verified environment and previous measured stain-arm timing; no new GPU benchmark.',
                                      'reference':str(ROOT/'stain_research/runs/hiesd_resnet_ablation_v1/seed42_shuffled/history.json')})
        atomic(CAM/'STATUS.json',{'stage':'training','seed':seed,'run':str(run),'completed':seed-42})
        command=[sys.executable,'-u',str(ROOT/'stain_research/hiesd_within_class_train.py'),'--seed',str(seed),'--run',str(run)]
        if (run/'last.pt').exists():command+=['--resume']
        with (run/'training.log').open('a') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
    atomic(CAM/'STATUS.json',{'stage':'complete','completed':3})
    # This is an explicitly bounded publication workflow, not an experiment search.
    driver=ROOT/'stain_research/publication_finalize_pipeline.py'
    if driver.exists():
        with (CAM/'publication_pipeline.log').open('a') as f:subprocess.run([sys.executable,'-u',str(driver)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,check=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:atomic(CAM/'FAILURE.json',{'error':str(exc)});raise
