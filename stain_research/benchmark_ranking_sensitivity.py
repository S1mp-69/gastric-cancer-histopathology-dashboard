"""Fixed-prediction class/patient reweighting; no fitting or pixel access."""
import csv,hashlib,json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ARMS=['standard','residual_fixed','uniform_weaker','shuffled','adaptive']
NAMES=['tumor','mucosa_inflammation','intestinal_metaplasia','lymphoid','other']

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):
    with path.open(newline='',encoding='utf-8') as stream:return list(csv.DictReader(stream))
def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def metrics(truth,pred,weights):
    cm=np.zeros((5,5));np.add.at(cm,(truth,pred),weights)
    tp=np.diag(cm);support=cm.sum(1);predicted=cm.sum(0)
    precision=np.divide(tp,predicted,out=np.zeros(5),where=predicted>0)
    recall=np.divide(tp,support,out=np.zeros(5),where=support>0)
    f1=np.divide(2*tp,support+predicted,out=np.zeros(5),where=support+predicted>0)
    return {'accuracy':float(tp.sum()/cm.sum()),'macro_f1':float(f1.mean()),
            'class_precision':precision.tolist(),'class_recall':recall.tolist(),'class_f1':f1.tolist(),'weighted_confusion_matrix':cm.tolist()}

def main():
    audit=ROOT/'stain_research/results/construction_audit_20261117_051530'
    earlier=json.loads((audit/'RESULTS.json').read_text())
    target_mixtures={}
    for row in earlier['class_patient_composition']:
        target_mixtures.setdefault(row['policy'],np.zeros(5))[NAMES.index(row['class'])]=row['share']
    specifications=[{'name':'observed_validation','within':'observed','mixture':None},
                    {'name':'uniform_patients','within':'patient_global','mixture':None},
                    {'name':'uniform_classes','within':'patch','mixture':[.2]*5},
                    {'name':'uniform_classes_and_patients','within':'patient_class','mixture':[.2]*5}]
    for name,mix in target_mixtures.items():
        for within in ['patch','patient_class']:
            specifications.append({'name':f'trainmix_{name}__{within}','within':within,'mixture':mix.tolist()})
    out=ROOT/'stain_research/results'/('ranking_sensitivity_'+datetime.now(ZoneInfo('Asia/Calcutta')).strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True,exist_ok=False)
    base=ROOT/'stain_research/runs/hiesd_resnet18_seed42_v3'
    campaign=ROOT/'stain_research/runs/hiesd_resnet_ablation_v1'
    run_paths={(s,a):(base if s==42 and a=='standard' else campaign/f'seed{s}_{a}') for s in [42,43,44] for a in ARMS}
    protected={str(p/'validation_predictions.csv'):sha(p/'validation_predictions.csv') for p in run_paths.values()}
    save(out/'PLAN.json',{'source_audit_sha256':sha(audit/'RESULTS.json'),'source_prediction_sha256':protected,'specifications':specifications,
         'primary_contrast':'Adaptive minus shuffled macro-F1; inspect all specified mixtures, no favorable selection.',
         'primary_summary':'Mean across the three matched seeds; individual-seed results retained.',
         'interpretation':'Reweight fixed validation cases by specified class/patient distributions. Does NOT apply new purity rules to validation images, retrain on alternative training sets, change labels, or identify causal effects.',
         'population_assumption':'Class-only reweighting treats observed within-class predictions as fixed. It is a sensitivity scenario, not an estimated real target population.',
         'known_limitations':['Validation already used for model selection','Annotation-selected small cohort','Only seven lymphoid validation cases','No independent test confirmation','No patient-level inferential claims'],
         'pixel_access':False,'gpu_use':False,'training':False})
    first=read(base/'validation_predictions.csv');identity=[(r['image_path'],r['patient_id'],r['slide_id'],r['class_id']) for r in first]
    truth=np.array([int(r['class_id']) for r in first]);patients=np.array([r['patient_id'] for r in first]);n=len(first)
    class_counts=np.bincount(truth,minlength=5);assert (class_counts>0).all()
    weights={};weight_details={}
    for spec in specifications:
        mode=spec['within'];w=np.zeros(n)
        if mode=='observed':w[:]=1/n
        elif mode=='patient_global':
            for p in set(patients):
                ids=patients==p;w[ids]=1/(len(set(patients))*ids.sum())
        else:
            for c in range(5):
                ids=truth==c;mass=spec['mixture'][c]
                if mode=='patch':w[ids]=mass/ids.sum()
                else:
                    ps=set(patients[ids])
                    for p in ps:
                        group=ids&(patients==p);w[group]=mass/(len(ps)*group.sum())
        assert np.isclose(w.sum(),1) and (w>=0).all();weights[spec['name']]=w
        weight_details[spec['name']]={'class_masses':np.bincount(truth,weights=w,minlength=5).tolist(),
             'patient_masses':{p:float(w[patients==p].sum()) for p in sorted(set(patients))},
             'weight_effective_patch_count':float(1/(w*w).sum()),'note':'Effective patch count measures weight concentration, not independent sample size.'}
    results=[];full={}
    for (seed,arm),path in run_paths.items():
        rows=read(path/'validation_predictions.csv');assert identity==[(r['image_path'],r['patient_id'],r['slide_id'],r['class_id']) for r in rows]
        probability=np.array([[float(r[f'p_{name}']) for name in NAMES] for r in rows]);assert np.isfinite(probability).all() and np.allclose(probability.sum(1),1,atol=1e-5)
        pred=probability.argmax(1);assert np.array_equal(pred,np.array([int(r['predicted_class_id']) for r in rows]))
        original=json.loads((path/'RESULTS.json').read_text())['validation'];unweighted=metrics(truth,pred,weights['observed_validation'])
        assert abs(original['macro_f1']-unweighted['macro_f1'])<1e-12 and abs(original['accuracy']-unweighted['accuracy'])<1e-12
        for name,w in weights.items():
            m=metrics(truth,pred,w);full[f'{seed}_{arm}_{name}']=m
            results.append({'scheme':name,'seed':seed,'arm':arm,'macro_f1':m['macro_f1'],'accuracy':m['accuracy']})
    summaries=[]
    for spec in specifications:
        name=spec['name'];means={a:float(np.mean([r['macro_f1'] for r in results if r['scheme']==name and r['arm']==a])) for a in ARMS}
        rank=sorted(ARMS,key=lambda a:(-means[a],a))
        delta=[next(r['macro_f1'] for r in results if r['scheme']==name and r['seed']==s and r['arm']=='adaptive')-
               next(r['macro_f1'] for r in results if r['scheme']==name and r['seed']==s and r['arm']=='shuffled') for s in [42,43,44]]
        summaries.append({'scheme':name,'rank':rank,'mean_macro_f1':means,'adaptive_minus_shuffled_by_seed':delta,'mean_adaptive_minus_shuffled':float(np.mean(delta))})
    reference=summaries[0];winner_changes=[s['scheme'] for s in summaries[1:] if s['rank'][0]!=reference['rank'][0]]
    reversals=[s['scheme'] for s in summaries[1:] if s['mean_adaptive_minus_shuffled']*reference['mean_adaptive_minus_shuffled']<0]
    assert all(sha(Path(p))==h for p,h in protected.items())
    save(out/'RESULTS.json',{'summaries':summaries,'weight_details':weight_details,'detailed_metrics':full,
         'winner_changes':winner_changes,'adaptive_shuffled_mean_sign_reversals':reversals,'source_predictions_unchanged':True,
         'observed_validation_metrics_recomputed_all15':True,'new_pixel_access':False,'training_or_gpu_use':False,
         'scope':'Descriptive reweighting only. Not actual dataset-policy intervention, causal attribution, clinical validation, statistical significance or confirmed novelty.'})
    with (out/'seed_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    lines=['# Fixed-prediction benchmark sensitivity','',
           'All15 original validation accuracy/macro-F1 values recomputed from saved probabilities. Same1251 cases and ordered identities throughout; all predictions preserved.',
           '', '| Scheme | Mean-F1 winner | Adaptive minus shuffled |','|---|---|---:|']
    for s in summaries:lines.append(f"|{s['scheme']}|{s['rank'][0]}|{s['mean_adaptive_minus_shuffled']:+.5f}|")
    lines+=['',f'Winner changes relative to observed validation: {len(winner_changes)}/{len(summaries)-1}.',
            f'Adaptive/shuffled mean sign reversals: {len(reversals)}/{len(summaries)-1}.',
            '', 'Training-derived class masses are hypothetical evaluation distributions, not validation purity selections. Within-class patient balancing gives equal mass to each represented patient within a class. Global patient balancing is a separate scenario. Class-mixture changes can change precision/F1 without changing class-conditional recall; this is a metric/population effect, not evidence of a better classifier.',
            '', 'No new validation/test images or masks were opened and no model was trained or run. Original validation was already used for early stopping and the cohort was annotation-selected. Only seven lymphoid cases are present, clustered within very few patients. Heavy weights may concentrate on a small number of patches: see weight_details in RESULTS.json. No independent patient confidence interval or clinical benefit is claimed.',
            '', 'Actual effects of retraining on alternative patch policies and performance on newly retained boundary cases remain untested. Do not describe this analysis as proving those effects, correcting annotation errors, or validating a novel method.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(out),'schemes':len(summaries),'winner_changes':winner_changes,'adaptive_shuffled_sign_reversals':reversals,
                     'observed_ranking':reference['rank'],'contrast_range':[min(s['mean_adaptive_minus_shuffled'] for s in summaries),max(s['mean_adaptive_minus_shuffled'] for s in summaries)]},indent=2))

if __name__=='__main__':main()
