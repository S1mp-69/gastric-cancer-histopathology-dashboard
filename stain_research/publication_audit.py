"""Post-hoc diagnostics and prospective metadata audit; no test image access."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from publication_controls import assigned_strengths
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation_package/publication_readiness'
def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 cam=ROOT/'stain_research/runs/overnight_20261113'
 directory=Path(json.loads((cam/'cache_reference.json').read_text())['directory'])
 index=pd.read_csv(directory/'index.csv');assert set(index.split)<={'train','val'}
 features=np.load(directory/'diagnostics.npy',mmap_mode='r');valid=np.load(directory/'valid.npy',mmap_mode='r')
 params=json.loads((directory/'train_parameters.json').read_text())['controller']
 scaled=np.clip((features-np.array(params['low']))/np.maximum(np.array(params['high'])-params['low'],1e-8),0,1)
 alpha=1-.9*scaled.mean(1);frame=index.copy();frame['valid']=valid;frame['alpha']=alpha;frame['risk']=scaled.mean(1)
 summaries=[]
 for split in ['train','val']:
  f=frame[(frame.split==split)&frame.valid];grand=f.alpha.mean();total=float(((f.alpha-grand)**2).sum())
  between=sum(len(g)*(g.alpha.mean()-grand)**2 for _,g in f.groupby('class_code'))
  for cls,g in f.groupby('class_code'):summaries.append({'split':split,'class':cls,'n':len(g),'alpha_mean':float(g.alpha.mean()),'alpha_sd':float(g.alpha.std()),'alpha_p10':float(g.alpha.quantile(.1)),'alpha_p90':float(g.alpha.quantile(.9))})
  save(OUT/f'{split}_diagnostic_summary.json',{'valid_n':len(f),'invalid_n':int(((frame.split==split)&~frame.valid).sum()),'alpha_variance_fraction_between_classes':between/total if total else None,'feature_spearman':float(spearmanr(features[f.index,0],features[f.index,1]).statistic)})
 pd.DataFrame(summaries).to_csv(OUT/'diagnostic_strength_by_class.csv',index=False)
 cut=np.quantile(frame[(frame.split=='train')&frame.valid].risk,[.25,.5,.75])
 rows=[]
 for seed in [42,43,44]:
  a=pd.read_csv(cam/f'seed{seed}_adaptive/validation_predictions.csv');b=pd.read_csv(cam/f'seed{seed}_standard/validation_predictions.csv')
  assert a.image_path.equals(b.image_path)
  m=a.merge(frame[['image_path','class_code','risk','valid']],on=['image_path','class_code'],validate='one_to_one')
  assert len(m)==3059
  m['standard_error']=(b.predicted_class.to_numpy()!=b.class_code.to_numpy()).astype(float)
  m['adaptive_error']=(m.predicted_class!=m.class_code).astype(float)
  m=m[m.valid].copy();m['risk_bin']=np.searchsorted(cut,m.risk,side='right')
  for bucket,g in m.groupby('risk_bin'):rows.append({'seed':seed,'train_defined_risk_quartile':int(bucket)+1,'n':len(g),'standard_accuracy':float(1-g.standard_error.mean()),'adaptive_accuracy':float(1-g.adaptive_error.mean())})
  # Within-class centered ranks remove class-average differences descriptively.
  m['risk_rank']=m.groupby('class_code').risk.rank(pct=True)
  m['error_centered']=m.standard_error-m.groupby('class_code').standard_error.transform('mean')
  save(OUT/f'seed{seed}_diagnostic_error_association.json',{'within_class_rank_risk_vs_standard_error_pearson':float(np.corrcoef(m.risk_rank,m.error_centered)[0,1]),'scope':'Post-hoc descriptive association, not independence-adjusted or causal evidence; no tuning performed'})
 pd.DataFrame(rows).to_csv(OUT/'risk_quartile_classification.csv',index=False)
 train=frame[frame.split=='train'];strengths=train.alpha.to_numpy();ok=train.valid.to_numpy();codes=train.class_code.to_numpy()
 for within in [False,True]:
  x=assigned_strengths(strengths,ok,codes,20261116,1,within)
  assert np.array_equal(x,assigned_strengths(strengths,ok,codes,20261116,1,within))
  for c in sorted(set(codes)) if within else [None]:
   mask=ok & (codes==c) if c is not None else ok
   assert np.array_equal(np.sort(x[mask]),np.sort(strengths[mask]))
  assert np.all(x[~ok]==0)
 save(OUT/'control_preflight.json',{'valid_training_n':int(ok.sum()),'global_and_within_class_permutations_preserve_exact_strength_multisets':True,'repeatability':True,'invalid_decomposition_fallback':True,'training_launched':False,'test_accessed':False})
 prepare_hiesd()
 print('Publication diagnostics, control preflight and HiESD grouping audit complete')

def prepare_hiesd():
 local=ROOT/'data/external/hiesd_metadata';m=pd.read_csv(local/'patient_slide_mapping.csv');s=pd.read_csv(local/'ESD104-Summary-new.csv')
 assert len(m)==104 and m.slide_id.nunique()==104 and m.patient_id.nunique()==44
 s['slide_id']=s['SVS(svs_files)'].str.replace('.svs','',regex=False)
 merged=s.merge(m,on='slide_id',how='left',validate='many_to_one');assert merged.patient_id.notna().all() and len(merged)==308
 patients=np.array(sorted(m.patient_id.unique()));np.random.default_rng(20261116).shuffle(patients)
 assignments={p:('train' if i<30 else 'validation' if i<37 else 'reserved_test') for i,p in enumerate(patients)}
 merged['split']=merged.patient_id.map(assignments)
 target=local/'prospective_grouped_manifest.csv'
 encoded=merged.to_csv(index=False).encode()
 if target.exists() and target.read_bytes()!=encoded:raise RuntimeError('Refusing to alter an existing frozen prospective split')
 target.write_bytes(encoded)
 fs=json.loads((OUT/'evidence/hiesd_figshare.json').read_text(encoding='utf-8-sig'))['files']
 original=[f for f in fs if f['name'].endswith('.svs')]
 report={'dataset':'HiESD','patient_mapping_verified_against_local_release':True,'patient_count':44,'slide_count':104,'tissue_strips':308,
  'patients_by_split':{k:sum(v==k for v in assignments.values()) for k in set(assignments.values())},
  'slides_by_split':merged.groupby('split').slide_id.nunique().to_dict(),'patient_intersections':0,
  'manifest_sha256':digest(target),'mapping_sha256':digest(local/'patient_slide_mapping.csv'),
  'source_license':'CC BY 4.0 in Figshare metadata','raw_slides_count':len(original),'raw_slides_bytes':sum(f['size'] for f in original),
  'actual_image_files_downloaded':False,'new_test_pixels_decoded':False,'label_mapping_to_HMU_8_classes':'Not established; distinct annotated task requires separate protocol and classifier head',
  'scope':'Prospective split frozen before image/model examination; class coverage and mapping consistency still require train-only acquisition checks. Metadata grouping is verified, not image extraction or external model validation.'}
 save(OUT/'HIESD_DATA_READINESS.json',report)
if __name__=='__main__':main()
