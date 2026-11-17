"""Independent sklearn verification and class decomposition of fixed predictions."""
import csv,json,hashlib
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'stain_research/results/ranking_sensitivity_20261117_051818'
AUDIT=ROOT/'stain_research/results/construction_audit_20261117_051530'
OUT=ROOT/'evaluation_package/publication_readiness/ranking_audit_20261117'
NAMES=['tumor','mucosa_inflammation','intestinal_metaplasia','lymphoid','other']
ARMS=['standard','residual_fixed','uniform_weaker','shuffled','adaptive']

def read(p):
    with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,data):p.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
def csvout(p,data):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plan=json.loads((SOURCE/'PLAN.json').read_text());old=json.loads((SOURCE/'RESULTS.json').read_text())
    for path,h in plan['source_prediction_sha256'].items():assert sha(Path(path))==h
    counts=defaultdict(Counter)
    for row in read(AUDIT/'memberships.csv'):counts[row['policy']][int(row['class_id'])]+=1
    mixtures={policy:np.array([c[i] for i in range(5)],float)/sum(c.values()) for policy,c in counts.items()}
    # Form weights independently from explicit per-record denominators.
    baseline=ROOT/'stain_research/runs/hiesd_resnet18_seed42_v3'
    sample=read(baseline/'validation_predictions.csv');n=len(sample)
    labels=np.array([int(r['class_id']) for r in sample]);patients=[r['patient_id'] for r in sample]
    identities=[tuple(r[k] for k in ['image_path','patient_id','slide_id','class_id']) for r in sample]
    nc=Counter(labels);np_all=Counter(patients);npc=Counter(zip(patients,labels))
    pc={c:len({p for p,y in zip(patients,labels) if y==c}) for c in range(5)}
    all_weights={};max_error=0.
    for spec in plan['specifications']:
        name=spec['name'];mode=spec['within']
        if name.startswith('trainmix_'):
            policy=name[len('trainmix_'):].rsplit('__',1)[0];q=mixtures[policy]
            assert np.allclose(q,spec['mixture'],atol=1e-15)
        else:q=np.array([.2]*5)
        w=[]
        for p,c in zip(patients,labels):
            if mode=='observed':value=1/n
            elif mode=='patient_global':value=1/(len(np_all)*np_all[p])
            elif mode=='patch':value=q[c]/nc[c]
            else:value=q[c]/(pc[c]*npc[(p,c)])
            w.append(value)
        w=np.array(w);assert abs(w.sum()-1)<1e-12
        assert np.allclose(np.bincount(labels,weights=w,minlength=5),old['weight_details'][name]['class_masses'])
        all_weights[name]=w
    rows=[];classrows=[];metrics={};source_hashes={}
    for seed in [42,43,44]:
        for arm in ARMS:
            run=baseline if seed==42 and arm=='standard' else ROOT/f'stain_research/runs/hiesd_resnet_ablation_v1/seed{seed}_{arm}'
            path=run/'validation_predictions.csv';source_hashes[str(path)]=sha(path);records=read(path)
            assert identities==[tuple(r[k] for k in ['image_path','patient_id','slide_id','class_id']) for r in records]
            probs=np.array([[float(r[f'p_{c}']) for c in NAMES] for r in records]);pred=probs.argmax(1)
            assert np.isfinite(probs).all() and np.allclose(probs.sum(1),1,atol=1e-5)
            for name,w in all_weights.items():
                cm=confusion_matrix(labels,pred,labels=list(range(5)),sample_weight=w)
                precision,recall,f1,support=precision_recall_fscore_support(labels,pred,labels=list(range(5)),sample_weight=w,zero_division=0)
                score=float(f1.mean());acc=float(accuracy_score(labels,pred,sample_weight=w));balanced=float(recall.mean())
                prior=old['detailed_metrics'][f'{seed}_{arm}_{name}']
                errors=[abs(score-prior['macro_f1']),abs(acc-prior['accuracy']),float(np.max(abs(cm-np.array(prior['weighted_confusion_matrix']))))]
                max_error=max(max_error,*errors);assert max(errors)<1e-10
                metrics[(seed,arm,name)]={'macro_f1':score,'accuracy':acc,'balanced_accuracy':balanced,'precision':precision,'recall':recall,'f1':f1}
                rows.append({'seed':seed,'arm':arm,'scheme':name,'macro_f1':score,'accuracy':acc,'balanced_accuracy':balanced})
                for i,c in enumerate(NAMES):classrows.append({'seed':seed,'arm':arm,'scheme':name,'class':c,'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i]),'weighted_support':float(support[i])})
                # Changing class masses alone preserves class-conditional recall.
                if name!='observed_validation' and (name=='uniform_classes' or name.endswith('__patch')):
                    assert np.max(abs(recall-metrics[(seed,arm,'observed_validation')]['recall']))<1e-10
    summaries=[]
    for name in all_weights:
        means={a:float(np.mean([metrics[(s,a,name)]['macro_f1'] for s in [42,43,44]])) for a in ARMS}
        balanced={a:float(np.mean([metrics[(s,a,name)]['balanced_accuracy'] for s in [42,43,44]])) for a in ARMS}
        rank=sorted(ARMS,key=lambda a:(-means[a],a));d=means['adaptive']-means['shuffled']
        prior=next(s for s in old['summaries'] if s['scheme']==name)
        assert rank==prior['rank'] and abs(d-prior['mean_adaptive_minus_shuffled'])<1e-10
        summaries.append({'scheme':name,'macro_f1_winner':rank[0],'macro_f1_rank':' > '.join(rank),'adaptive_minus_shuffled':d,
                          'balanced_accuracy_winner':max(balanced,key=balanced.get),**{f'f1_{a}':means[a] for a in ARMS},**{f'balanced_{a}':balanced[a] for a in ARMS}})
    drivers=[]
    for c,name in enumerate(NAMES):
        contributions={}
        for scheme in ['observed_validation','uniform_patients','uniform_classes','uniform_classes_and_patients']:
            delta=np.mean([metrics[(s,'adaptive',scheme)]['f1'][c]-metrics[(s,'shuffled',scheme)]['f1'][c] for s in [42,43,44]])/5
            contributions[scheme]=float(delta)
        drivers.append({'class':name,**contributions,'contribution_to_reversal_shift':contributions['uniform_classes']-contributions['observed_validation']})
    for scheme in ['observed_validation','uniform_patients','uniform_classes','uniform_classes_and_patients']:
        assert abs(sum(d[scheme] for d in drivers)-next(r['adaptive_minus_shuffled'] for r in summaries if r['scheme']==scheme))<1e-10
    csvout(OUT/'all_24_weightings.csv',summaries);csvout(OUT/'verified_seed_metrics.csv',rows)
    csvout(OUT/'verified_class_metrics.csv',classrows);csvout(OUT/'class_reversal_contributions.csv',drivers)
    save(OUT/'VERIFICATION.json',{'passed':True,'independent_method':'sklearn sample_weight confusion_matrix, precision_recall_fscore_support and accuracy_score; weights regenerated per record, training mixtures recomputed from raw membership CSV.',
         'runs':15,'schemes':len(all_weights),'weighted_confusion_matrices_verified':len(rows),'class_metric_rows':len(classrows),
         'maximum_numeric_difference':max_error,'all_rankings_match':True,'class_only_reweighting_recall_invariance_verified':True,
         'decomposition_sums_verified':True,'source_hashes':source_hashes,'patient_count':len(np_all),'class_patient_counts':{NAMES[c]:pc[c] for c in range(5)},
         'pixels_or_models_accessed':False,'gpu_checks_or_training':False})
    matrix=np.array([[r[f'f1_{a}'] for a in ARMS] for r in summaries])
    fig,ax=plt.subplots(figsize=(9,10));im=ax.imshow(matrix,cmap='viridis',aspect='auto')
    ax.set_xticks(range(5),ARMS,rotation=25,ha='right');ax.set_yticks(range(len(summaries)),[r['scheme'] for r in summaries],fontsize=8)
    for i in range(len(summaries)):
        for j in range(5):ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',fontsize=7,color='white' if matrix[i,j]<.70 else 'black')
    ax.set_title('Mean macro-F1 across three seeds: fixed validation predictions')
    fig.colorbar(im,ax=ax,label='Macro-F1');fig.tight_layout();fig.savefig(OUT/'all_weightings.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar(NAMES,[d['contribution_to_reversal_shift']*100 for d in drivers]);ax.axhline(0,color='black',lw=.8)
    ax.set(ylabel='Contribution to contrast shift (macro-F1 points)',title='Observed to equal-class weighting: adaptive minus shuffled')
    ax.tick_params(axis='x',rotation=20);fig.tight_layout();fig.savefig(OUT/'class_drivers.png',dpi=180);plt.close(fig)
    for path,h in source_hashes.items():assert sha(Path(path))==h
    report=['# Independent ranking audit','',f'PASS: all{len(rows)} weighted confusion matrices and scores reproduced with sklearn. Maximum numeric difference: {max_error:.3g}. All24 scheme rankings agree.','',
            '| Class | Original contribution | Equal-class contribution | Shift contribution |','|---|---:|---:|---:|']
    for d in sorted(drivers,key=lambda r:abs(r['contribution_to_reversal_shift']),reverse=True):report.append(f"|{d['class']}|{d['observed_validation']:+.5f}|{d['uniform_classes']:+.5f}|{d['contribution_to_reversal_shift']:+.5f}|")
    report+=['','Contributions are (adaptive F1 minus shuffled F1)/5, averaged over three seeds. They add exactly to the macro-F1 contrast. Pure class reweighting preserves every class recall and balanced accuracy; it changes precision through the relative mass of false-positive source classes.',
             '', 'All24 weightings are reported in all_24_weightings.csv, including unfavorable results. Three seeds and24 scenarios are not72 independent cohorts. These are hypothetical reweightings of a validation set already used for stopping, not new test results or retraining interventions.']
    (OUT/'AUDIT_REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps({'verified_matrices':len(rows),'max_error':max_error,'drivers':drivers,'patient_count':len(np_all),'report':str(OUT)},indent=2))

if __name__=='__main__':main()
