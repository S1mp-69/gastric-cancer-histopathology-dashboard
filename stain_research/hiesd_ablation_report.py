"""Descriptive matched-arm summaries, including calibration and class failures."""
import csv,json
from pathlib import Path
import numpy as np
from hiesd_pilot import ROOT,NAMES,digest
from hiesd_resnet18 import atomic

CAM=ROOT/'stain_research/runs/hiesd_resnet_ablation_v1'
BASE=ROOT/'stain_research/runs/hiesd_resnet18_seed42_v3'
ARMS=['standard','residual_fixed','uniform_weaker','shuffled','adaptive']

def summarize():
    out=ROOT/'stain_research/results/hiesd_resnet_ablation_v1';out.mkdir(parents=True,exist_ok=True)
    rows=[];reference=None;details={}
    for seed in [42,43,44]:
        for arm in ARMS:
            run=BASE if seed==42 and arm=='standard' else CAM/f'seed{seed}_{arm}'
            if not (run/'RESULTS.json').exists():continue
            result=json.loads((run/'RESULTS.json').read_text())
            if result['stage']!='complete' or not result['reload_predictions_match']:raise RuntimeError('Incomplete/unverified result')
            with (run/'validation_predictions.csv').open() as stream:predictions=list(csv.DictReader(stream))
            identities=[(r['image_path'],r['patient_id'],r['slide_id'],r['class_id']) for r in predictions]
            if reference is None:reference=identities
            else:assert reference==identities
            probability=np.array([[float(r[f'p_{n}']) for n in NAMES] for r in predictions]);truth=np.array([int(r['class_id']) for r in predictions])
            assert np.isfinite(probability).all() and np.allclose(probability.sum(1),1,atol=1e-5)
            confidence=probability.max(1);correct=probability.argmax(1)==truth;ece=0.
            for b in range(15):
                mask=(confidence>=b/15)&((confidence<(b+1)/15) if b<14 else (confidence<=1))
                if mask.any():ece+=mask.mean()*abs(correct[mask].mean()-confidence[mask].mean())
            brier=float(((probability-np.eye(5)[truth])**2).sum(1).mean())
            m=result['validation'];record={'seed':seed,'arm':arm,'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],
                    'tumor_recall':m['classification_report']['tumor']['recall'],'other_recall':m['classification_report']['other']['recall'],
                    'ece_15bins':float(ece),'brier':brier,'epochs':result['completed_epochs'],'best_epoch':result['best_epoch'],
                    'elapsed_seconds':result['elapsed_this_process_seconds']}
            rows.append(record);details[f'{seed}_{arm}']=m
    preserved=json.loads((CAM/'baseline_preservation.json').read_text());assert all(digest(BASE/n)==h for n,h in preserved.items())
    grouped={}
    for arm in ARMS:
        chosen=[r for r in rows if r['arm']==arm]
        if chosen:grouped[arm]={'seeds':[r['seed'] for r in chosen],'macro_f1_mean':float(np.mean([r['macro_f1'] for r in chosen])),
                               'macro_f1_sd':float(np.std([r['macro_f1'] for r in chosen],ddof=1)) if len(chosen)>1 else None}
    paired=[]
    for seed in [42,43,44]:
        lookup={r['arm']:r for r in rows if r['seed']==seed}
        if 'adaptive' in lookup and 'shuffled' in lookup:
            paired.append({'seed':seed,'adaptive_minus_shuffled_macro_f1':lookup['adaptive']['macro_f1']-lookup['shuffled']['macro_f1']})
    atomic(out/'summary.json',{'runs':rows,'arm_summaries':grouped,'paired_assignment_comparison':paired,
           'per_class_details':details,'baseline_preserved':True,'same_validation_order':True,'reserved_test_used':False,
           'limitations':['Annotation-selected validation cohort','Few patients and only seven validation lymphoid patches','Single-seed results until replication completes','Stain arms share empirical strength distribution but not actual pixel/OD change','No pathology expert review; no clinical/novelty claim']})
    if rows:
        with (out/'summary.csv').open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    text=['# HiESD matched ResNet-18 augmentation comparison','',
          'Exploratory selected-cohort experiment. Common split, initialization within each seed, standard RGB/geometric augmentation, normalization and training budgets. Stain arms add residual-preserving augmentation before the shared standard pipeline. Reserved test data excluded.',
          '', '| Seed | Arm | Accuracy | Macro-F1 | Tumor recall | Other recall | ECE | Epochs |','|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:text.append(f"|{r['seed']}|{r['arm']}|{r['accuracy']:.4f}|{r['macro_f1']:.4f}|{r['tumor_recall']:.4f}|{r['other_recall']:.4f}|{r['ece_15bins']:.4f}|{r['epochs']}|")
    text+=['','Adaptive-minus-shuffled macro-F1 differences: '+json.dumps(paired)+'.',
           '', 'Seed variation does not provide independent patient samples. Calibration values are descriptive; the validation cohort was used for early stopping. The study follows earlier negative/mixed experiments and is not preregistered confirmation. Exact duplicate control and patient grouping do not establish clinical transportability.',
           '', 'All original baseline artifacts remain hash-verified. Latest/best research checkpoints remain isolated from the dashboard. No claims of novelty, clinical benefit or method superiority are implied by execution or individual rankings.']
    (out/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return rows

if __name__=='__main__':summarize()
