"""Matched shuffled-strength follow-up. Reuses the audited trainer unchanged.
Only training diagnostic assignment changes. Never edits the source cache.
"""
import argparse,copy,hashlib,json,multiprocessing as mp,os,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import overnight_trainer as trainer
from controller import strength
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'stain_research/runs/overnight_20261113'
CAM=ROOT/'stain_research/runs/shuffled_20261116'
SMOKE=False

class ShuffledImages(trainer.CachedImages):
    def __init__(self,directory,split,arm,seed,epoch,subset=None):
        if arm!='shuffled':raise ValueError('This isolated runner accepts shuffled only')
        super().__init__(directory,split,'adaptive',seed,epoch,subset=(256 if split=='train' else 128) if SMOKE else subset)
        self._original_diagnostics=None;self._assignment_epoch=-1
    def __getstate__(self):
        result=super().__getstate__();result['_original_diagnostics']=None;result['_assignment_epoch']=-1;return result
    def open(self):
        super().open()
        if self._original_diagnostics is None:
            self._original_diagnostics=np.load(Path(self.directory)/'diagnostics.npy',mmap_mode='r')
            self._assignment_epoch=-1
        epoch=int(self.epoch.value)
        if self.split=='train' and epoch!=self._assignment_epoch:
            valid=self.maps['valid'][self.indices];indices=self.indices[valid]
            rng=np.random.default_rng(np.random.SeedSequence([self.seed,epoch,710301]))
            # Source basis/concentrations/RGB stay attached to their image.
            # Permuting only diagnostics permutes the resulting alpha exactly.
            features=np.array(self._original_diagnostics,copy=True)
            features[indices]=self._original_diagnostics[rng.permutation(indices)]
            self.maps['diagnostics']=features;self._assignment_epoch=epoch

def install():trainer.CachedImages=ShuffledImages
def prepare():
    CAM.mkdir(parents=True,exist_ok=True)
    config=json.loads((BASE/'campaign_config.json').read_text())
    config.update({'created_at':trainer.now(),'arms':['shuffled'],'followup':'Post-hoc global epoch-wise diagnostic permutation; seeds42/43/44; no retuning',
       'source_code_sha256':{p.name:trainer.sha(p) for p in [Path(__file__),ROOT/'stain_research/overnight_trainer.py',ROOT/'stain_research/efficient_cache.py']}})
    if (CAM/'campaign_config.json').exists():
        old=json.loads((CAM/'campaign_config.json').read_text())
        if old['source_code_sha256']!=config['source_code_sha256']:raise RuntimeError('Campaign code changed; refuse silent resume')
    else:trainer.atomic_json(CAM/'campaign_config.json',config)
    for name in ['cache_reference.json','initial_seed42.pt','initial_seed43.pt','initial_seed44.pt']:
        target=CAM/name
        if target.exists():assert trainer.sha(target)==trainer.sha(BASE/name)
        else:shutil.copy2(BASE/name,target)

def preflight():
    prepare();install();directory=Path(json.loads((CAM/'cache_reference.json').read_text())['directory'])
    epoch=mp.Value('i',0,lock=False);d=ShuffledImages(directory,'train','shuffled',42,epoch);d.open()
    ids=d.indices[d.maps['valid'][d.indices]];p=d.params['controller']
    expected=np.array([strength(v,p) for v in d._original_diagnostics[ids]])
    actual=np.array([strength(v,p) for v in d.maps['diagnostics'][ids]])
    assert np.array_equal(np.sort(expected),np.sort(actual)) and not np.array_equal(expected,actual)
    e0=d.maps['diagnostics'][ids].copy();d._assignment_epoch=-1;d.open();assert np.array_equal(e0,d.maps['diagnostics'][ids])
    epoch.value=1;d.open();assert not np.array_equal(e0,d.maps['diagnostics'][ids])
    val=ShuffledImages(directory,'val','shuffled',42,epoch);val.open();assert np.array_equal(val.maps['diagnostics'],val._original_diagnostics)
    # Compare persistent multi-worker loading with direct loading for exact inputs.
    epoch.value=0;d.open();loader=trainer.make_loader(d,64,4)
    first=next(iter(loader));reference=[d[i] for i in range(64)]
    assert torch.equal(first[0],torch.stack([r[0] for r in reference]))
    assert np.array_equal(first[3].numpy(),np.array([r[3] for r in reference]))
    del loader,d,val
    global SMOKE
    SMOKE=True
    smoke=CAM/'smoke';smoke.mkdir(exist_ok=True)
    cfg=json.loads((CAM/'campaign_config.json').read_text());cfg.update(max_epochs=2,workers=2,checkpoint_seconds=120)
    trainer.atomic_json(smoke/'campaign_config.json',cfg)
    for name in ['cache_reference.json','initial_seed42.pt']:shutil.copy2(CAM/name,smoke/name)
    saved=trainer.atomic_checkpoint
    def interrupt(path,state):
        saved(path,state)
        if Path(path).name=='last.pt' and state['epoch']==1:raise RuntimeError('PLANNED_SMOKE_INTERRUPTION')
    trainer.atomic_checkpoint=interrupt
    try:trainer.run_training(smoke,'shuffled',42)
    except RuntimeError as exc:
        if str(exc)!='PLANNED_SMOKE_INTERRUPTION':raise
    finally:trainer.atomic_checkpoint=saved
    torch.cuda.empty_cache();trainer.run_training(smoke,'shuffled',42)
    result=json.loads((smoke/'seed42_shuffled/metrics.json').read_text());assert result['completed_epochs']==2 and result['checkpoint_reload_verified']
    SMOKE=False
    seconds=result['mean_epoch_seconds'];estimated=23761/256*seconds
    receipt={'exact_valid_strength_multiset':True,'repeatability':True,'assignment_changes_by_epoch':True,
      'validation_diagnostics_unchanged':True,'four_worker_tensor_equivalence':True,'smoke_epochs':2,
      'checkpoint_interruption_and_resume_passed':True,'best_reload_verified':True,'test_accessed':False,
      'gpu':torch.cuda.get_device_name(),'telemetry':trainer.telemetry(),
      'prior_measured_full_epoch_seconds':119.3,'expected_full_campaign_hours_at_ceiling':3*40*119.3/3600,
      'smoke_epoch_seconds_including_startup':seconds,'note':'Full-data epoch measurement will refine prior estimate; smoke startup makes linear extrapolation unsuitable'}
    trainer.atomic_json(CAM/'preflight.json',receipt);print(json.dumps(receipt),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['preflight','train']);p.add_argument('--seed',type=int,choices=[42,43,44],default=42);args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    with threadpool_limits(limits=2):
        if args.mode=='preflight':preflight()
        else:
            prepare()
            if not (CAM/'preflight.json').exists():raise RuntimeError('Pass preflight first')
            install();trainer.run_training(CAM,'shuffled',args.seed)
if __name__=='__main__':main()
