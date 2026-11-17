"""Three bounded missing controls, then a frozen final-test workflow if prepared."""
import json,subprocess,sys,time,shutil,os,ctypes
from pathlib import Path
import numpy as np
from hiesd_pilot import ROOT,digest
from hiesd_resnet18 import atomic
from hiesd_ablation_data import Images,configure
from publication_controls import assigned_strengths
import csv
CAM=ROOT/'stain_research/runs/hiesd_within_class_v1'

def alive(pid):
    if not pid:return False
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_bool,ctypes.c_uint32];kernel.OpenProcess.restype=ctypes.c_void_p
    handle=kernel.OpenProcess(0x00100000,False,int(pid))
    if not handle:return False
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32];kernel.WaitForSingleObject.restype=ctypes.c_uint32
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    try:return kernel.WaitForSingleObject(handle,0)==258
    finally:kernel.CloseHandle(handle)

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
        process_file=run/'worker_process.json'
        prior=json.loads(process_file.read_text()) if process_file.exists() else {}
        while alive(prior.get('pid')):
            atomic(CAM/'STATUS.json',{'stage':'monitoring_existing_worker','seed':seed,'pid':prior['pid']})
            time.sleep(15)
        if (run/'RESULTS.json').exists():continue
        atomic(run/'benchmark.json',{'scope':'Reuse verified environment and previous measured stain-arm timing; no new GPU benchmark.',
                                      'reference':str(ROOT/'stain_research/runs/hiesd_resnet_ablation_v1/seed42_shuffled/history.json')})
        atomic(CAM/'STATUS.json',{'stage':'training','seed':seed,'run':str(run),'completed':seed-42})
        command=[sys.executable,'-u',str(ROOT/'stain_research/hiesd_within_class_train.py'),'--seed',str(seed),'--run',str(run)]
        if (run/'last.pt').exists():command+=['--resume']
        with (run/'training.log').open('a') as f:
            child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
            atomic(process_file,{'pid':child.pid,'seed':seed,'started':time.time()})
            if child.wait()!=0:raise RuntimeError(f'Seed {seed} failed; saved logs and checkpoints retained')
    atomic(CAM/'STATUS.json',{'stage':'complete','completed':3})
    # This is an explicitly bounded publication workflow, not an experiment search.
    driver=ROOT/'stain_research/publication_finalize_pipeline.py'
    if driver.exists():
        final=ROOT/'stain_research/results/final_test_hiesd_v1/RESULTS.json'
        target=ROOT/'stain_research/refresh_publication_package.py' if final.exists() and json.loads(final.read_text()).get('stage')=='complete' else driver
        with (CAM/'publication_pipeline.log').open('a') as f:subprocess.run([sys.executable,'-u',str(target)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,check=True)

if __name__=='__main__':
    CAM.mkdir(parents=True,exist_ok=True)
    lock=CAM/'queue_process.json';old=json.loads(lock.read_text()) if lock.exists() else {}
    if alive(old.get('pid')):print('Existing publication queue remains active; no duplicate launched.',flush=True);sys.exit(0)
    atomic(lock,{'pid':os.getpid(),'started':time.time()})
    try:main()
    except Exception as exc:atomic(CAM/'FAILURE.json',{'error':str(exc)});raise
    finally:
        if lock.exists() and json.loads(lock.read_text()).get('pid')==os.getpid():lock.unlink()
