"""Audit completed campaign files and compute descriptive matched-seed results.
Never loads test images or changes a checkpoint.
"""
import hashlib,itertools,json,statistics,csv
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score,f1_score
ROOT=Path(__file__).resolve().parents[1]
CAM=ROOT/'stain_research/runs/overnight_20261113'
OUT=ROOT/'stain_research/results/final_campaign_20261113'
ARMS=['standard','residual_fixed','uniform_weaker','adaptive']
CLASSES=['ADI','DEB','LYM','MUC','MUS','NOR','STR','TUM']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=True);config=json.loads((CAM/'campaign_config.json').read_text());records=[];checks=[]
 for seed in [42,43,44]:
  initial=sha(CAM/f'initial_seed{seed}.pt');reference=None
  for arm in ARMS:
   folder=CAM/f'seed{seed}_{arm}';m=json.loads((folder/'metrics.json').read_text());c=json.loads((folder/'config.json').read_text())
   assert all(c[k]==v for k,v in config.items()),folder
   assert m['initial_sha256']==initial and m['checkpoint_reload_verified'] and not m['test_evaluated']
   assert sha(folder/'best.pt')==m['best_checkpoint_sha256']
   frame=pd.read_csv(folder/'validation_predictions.csv');identity=frame[['image_path','class_code']].to_csv(index=False)
   if reference is None:reference=identity
   assert reference==identity
   assert len(frame)==3059 and set(frame.class_code)==set(CLASSES)
   assert abs(accuracy_score(frame.class_code,frame.predicted_class)-m['validation']['accuracy'])<1e-12
   assert abs(f1_score(frame.class_code,frame.predicted_class,average='macro')-m['validation']['macro_f1'])<1e-12
   probabilities=frame[['p_'+x for x in CLASSES]].to_numpy();assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(1),1,atol=1e-5)
   state=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
   model=state.get('model',state);assert all(torch.isfinite(v).all() for v in model.values() if torch.is_tensor(v))
   last=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
   assert all(k in last for k in ['optimizer','scheduler','scaler','rng','epoch','offset'])
   checks.append({'run':folder.name,'config_matches':True,'initial_hash_matches':True,'prediction_metrics_recomputed':True,'best_hash_matches':True,'best_tensors_finite':True,'resumable_state_present':True,'validation_identity_sha256':hashlib.sha256(identity.encode()).hexdigest()})
   records.append({'arm':arm,'seed':seed,'epochs':m['completed_epochs'],'best_epoch':m['best_epoch'],'stop_reason':m['stop_reason'],**{k:m['validation'][k] for k in ['accuracy','macro_f1','ece_10_bins','multiclass_brier']},'per_class':m['validation']['per_class'],'confusion_matrix':m['validation']['confusion_matrix'],'mean_epoch_seconds':m['mean_epoch_seconds']})
 summary=[]
 for arm in ARMS:
  rows=[r for r in records if r['arm']==arm]
  summary.append({'arm':arm,'n':3,**{k+'_'+suffix:fun(r[k] for r in rows) for k in ['accuracy','macro_f1','ece_10_bins','multiclass_brier'] for suffix,fun in [('mean',statistics.mean),('sd',statistics.stdev)]},
    'recall_mean':{c:statistics.mean(r['per_class'][c]['recall'] for r in rows) for c in CLASSES},
    'confusion_matrix_sum':np.sum([r['confusion_matrix'] for r in rows],axis=0).tolist()})
 paired=[]
 for control in ARMS[:-1]:
  delta=[next(r['macro_f1'] for r in records if r['arm']=='adaptive' and r['seed']==s)-next(r['macro_f1'] for r in records if r['arm']==control and r['seed']==s) for s in [42,43,44]]
  mean=statistics.mean(delta);sd=statistics.stdev(delta);half=4.302652729911275*sd/3**.5
  paired.append({'adaptive_minus':control,'seeds':[42,43,44],'macro_f1_differences':delta,'mean':mean,'sd':sd,'exploratory_t_interval_95':[mean-half,mean+half],'interval_assumption':'Independent approximately normal seed differences; n=3, very uncertain; not patient-level uncertainty or multiplicity-adjusted evidence'})
 result={'complete_runs':12,'summary':summary,'paired':paired,'runs':records,'integrity':checks,'scope':'Validation only; exact-file-controlled image split; no patient independence; checkpoints selected on validation','inference':'Descriptive three-seed comparison; no superiority, clinical benefit or novelty established'}
 (OUT/'analysis.json').write_text(json.dumps(result,indent=2))
 with (OUT/'summary.csv').open('w',newline='') as f:
  fields=['arm','n','accuracy_mean','accuracy_sd','macro_f1_mean','macro_f1_sd'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(summary)
 lines=['# Completed campaign analysis','', 'All 12 runs completed. Three matched seeds per arm. Validation only.','', '| Arm | Accuracy mean ± SD | Macro F1 mean ± SD | TUM recall mean |','|---|---:|---:|---:|']
 for r in summary:lines.append(f"| {r['arm']} | {100*r['accuracy_mean']:.2f} ± {100*r['accuracy_sd']:.2f}% | {r['macro_f1_mean']:.4f} ± {r['macro_f1_sd']:.4f} | {100*r['recall_mean']['TUM']:.2f}% |")
 lines+=['','Adaptive minus control, paired macro-F1 differences:']
 for r in paired:lines.append(f"- {r['adaptive_minus']}: mean {r['mean']:+.4f}; seeds {[round(x,4) for x in r['macro_f1_differences']]}; exploratory 95% t interval {[round(x,4) for x in r['exploratory_t_interval_95']]}. n=3 and strong assumptions; not patient-level uncertainty.")
 lines+=['','All completed-run best hashes, initialization hashes, shared configurations, ordered validation identities, finite best model tensors, prediction metric recomputation and resumable checkpoint keys passed audit. These checks do not independently re-run all inference or prove bit-identical interrupted training.','', 'Inspect class recall and learning curves; epoch ceilings do not prove convergence. Historical numerical preprocessing results and the earlier negative three-epoch pilot remain separate. The test set remains unexamined. Patient/slide mapping and near-duplicate control are unresolved. Shuffled diagnostics were only evaluated in the short pilot, so causal benefit from diagnostic assignment is unconfirmed.','', 'Recovery logs retain AMP-handler and Windows file-lock failures. No new training was launched for this analysis.']
 (OUT/'FINAL_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps({'runs_verified':12,'summary':summary,'paired':paired}))
if __name__=='__main__':main()
