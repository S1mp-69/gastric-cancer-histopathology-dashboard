"""Audit all15 completed runs without training or held-out image decoding."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import hashlib,json,statistics
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation_package/publication_readiness/audit_20261116'
CLASSES=['ADI','DEB','LYM','MUC','MUS','NOR','STR','TUM']
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 OUT.mkdir(exist_ok=True);base=ROOT/'stain_research/runs/overnight_20261113';follow=ROOT/'stain_research/runs/shuffled_20261116';runs=[];reference=None
 cfg=read(base/'campaign_config.json');fcfg=read(follow/'campaign_config.json')
 keys=['max_epochs','patience','min_delta','batch_size','workers','amp','learning_rate','weight_decay','checkpoint_seconds','scheduler','evaluation']
 assert all(cfg[k]==fcfg[k] for k in keys)
 for name,expected in fcfg['source_code_sha256'].items():assert sha(ROOT/'stain_research'/name)==expected
 cache_ref=read(base/'cache_reference.json');assert cache_ref==read(follow/'cache_reference.json')
 directory=Path(cache_ref['directory']);index=pd.read_csv(directory/'index.csv');manifest=ROOT/'stain_research/runs/split_audit_20261113/manifest_exact_dedup.csv'
 assert sha(manifest)==cache_ref['source']['manifest_sha256'];split=pd.read_csv(manifest)
 assert set(index.split)=={'train','val'} and len(index)==26820
 expected=split[split.split.isin(['train','val'])].reset_index(drop=True)
 assert index[['image_path','class_code','split','sha256']].equals(expected[['image_path','class_code','split','sha256']])
 assert not set(index[index.split=='train'].sha256)&set(index[index.split=='val'].sha256)
 params=read(directory/'train_parameters.json');feat=np.load(directory/'diagnostics.npy',mmap_mode='r');valid=np.load(directory/'valid.npy',mmap_mode='r');train=(index.split=='train').to_numpy()
 q=np.quantile(feat[train&valid],[.1,.9],axis=0);assert np.array_equal(q,np.array([params['controller']['low'],params['controller']['high']]))
 for seed in [42,43,44]:
  for arm in ['standard','residual_fixed','uniform_weaker','adaptive','shuffled']:
   campaign=follow if arm=='shuffled' else base;p=campaign/f'seed{seed}_{arm}';m=read(p/'metrics.json');c=read(p/'config.json');hist=read(p/'history.json')
   assert all(c[k]==cfg[k] for k in keys)
   assert m['completed_epochs']==len(hist)<=40 and m['checkpoint_reload_verified'] and not m['test_evaluated']
   assert m['initial_sha256']==sha(base/f'initial_seed{seed}.pt')==sha(campaign/f'initial_seed{seed}.pt')
   assert sha(p/'best.pt')==m['best_checkpoint_sha256']
   state=torch.load(p/'best.pt',map_location='cpu',weights_only=False)
   weights=state['model_state_dict'];assert weights and all(torch.is_tensor(v) and torch.isfinite(v).all() for v in weights.values())
   last=torch.load(p/'last.pt',map_location='cpu',weights_only=False)
   assert all(k in last for k in ['model','optimizer','scheduler','scaler','rng','epoch','offset'])
   selected=None;stale=0
   for h in hist:
    if selected is None or h['val']['macro_f1']>selected['val']['macro_f1']+.0001:selected=h;stale=0
    else:stale+=1
   assert selected['epoch']==m['best_epoch']==state['epoch']
   assert m['stop_reason']==('early_stopping' if stale>=7 else 'maximum_epoch_budget')
   f=pd.read_csv(p/'validation_predictions.csv');ids=f[['image_path','class_code']].to_csv(index=False)
   if reference is None:reference=ids
   assert reference==ids and len(f)==3059
   probabilities=f[['p_'+c for c in CLASSES]].to_numpy();pred=np.array(CLASSES)[probabilities.argmax(1)]
   assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(1),1,atol=1e-5)
   assert np.array_equal(pred,f.predicted_class.to_numpy())
   acc=accuracy_score(f.class_code,pred);f1=f1_score(f.class_code,pred,average='macro')
   assert abs(acc-m['validation']['accuracy'])<1e-12 and abs(f1-m['validation']['macro_f1'])<1e-12
   report=classification_report(f.class_code,pred,labels=CLASSES,output_dict=True,zero_division=0)
   assert np.array_equal(confusion_matrix(f.class_code,pred,labels=CLASSES),m['validation']['confusion_matrix'])
   for code in CLASSES:
    for metric in ['precision','recall','f1-score','support']:assert abs(report[code][metric]-m['validation']['per_class'][code][metric])<1e-12
   correct=(pred==f.class_code.to_numpy());conf=probabilities.max(1);ece=0
   for low,high in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
    mask=(conf>low)&(conf<=high)
    if mask.any():ece+=mask.mean()*abs(correct[mask].mean()-conf[mask].mean())
   y=np.array([CLASSES.index(v) for v in f.class_code]);brier=np.square(probabilities-np.eye(8)[y]).sum(1).mean()
   assert abs(ece-m['validation']['ece_10_bins'])<1e-6 and abs(brier-m['validation']['multiclass_brier'])<1e-6
   before=pd.read_csv(p/'best_validation_before_reload.csv');assert np.allclose(before[['p_'+c for c in CLASSES]],probabilities,rtol=1e-4,atol=1e-6)
   log=(p/'worker.log').read_text(encoding='utf-8',errors='replace')
   runs.append({'seed':seed,'arm':arm,'accuracy':acc,'macro_f1':f1,'epochs':len(hist),'tum_recall':report['TUM']['recall'],'checkpoint_tensor_count':len(weights),'all_checks_passed':True,'restarts_logged':log.count('Resuming '),'tracebacks_logged':log.count('Traceback'),'amp_backoffs_logged':log.count('AMP overflow handled')})
 summary=[]
 for arm in ['standard','residual_fixed','uniform_weaker','adaptive','shuffled']:
  r=[v for v in runs if v['arm']==arm];summary.append({'arm':arm,'accuracy_mean':statistics.mean(v['accuracy'] for v in r),'accuracy_sd':statistics.stdev(v['accuracy'] for v in r),'f1_mean':statistics.mean(v['macro_f1'] for v in r),'f1_sd':statistics.stdev(v['macro_f1'] for v in r),'tum_recall_mean':statistics.mean(v['tum_recall'] for v in r)})
 # Metadata-only audit of legacy demo prototype selection; no image decoding.
 tests=set(split[split.split=='test'].image_path.map(lambda x:str(Path(x).resolve())))
 proto=[]
 for code in CLASSES:proto+=sorted((ROOT/'data/raw/HMU-GC-HE-30K/all_image'/code).glob('*.png'))[:60]
 overlaps=[str(p.relative_to(ROOT)) for p in proto if str(p.resolve()) in tests]
 result={'runs_checked':15,'checks_passed':True,'strict_finite_tensor_check':'Accessed model_state_dict explicitly; an earlier audit skipped nested state dicts and was insufficient on this check','runs':runs,'summary':summary,
  'split_and_cache_verified':True,'training_fitted_diagnostic_bounds_verified':True,'test_pixels_read_by_this_audit':False,
  'legacy_dashboard_prototype_selection':{'first_60_per_class':len(proto),'retained_test_path_overlap':len(overlaps),'overlap_paths':overlaps,'scope':'Static source and filename comparison, not a new inference run; scientific classifier runner does not use these prototypes'},
  'documents_status':'Report/PPT/IEEE and Drive copies last generated13Nov, before shuffled follow-up; they are stale for15-run conclusions',
  'publication_status':'Not submission-ready: full-budget diagnostics superiority unsupported; patient independence unverified; no verified Overleaf project or Turnitin report'}
 (OUT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 pd.DataFrame(summary).to_csv(OUT/'five_arm_summary.csv',index=False)
 print(json.dumps({'runs_checked':15,'checks_passed':True,'summary':summary,'legacy_prototype_test_overlap':len(overlaps)},indent=2))
if __name__=='__main__':main()
