"""Summarize one post-hoc class-weighted follow-up; no fitting or test access."""
import csv,json
from pathlib import Path
import numpy as np
from hiesd_pilot import ROOT,NAMES,digest,save
from hiesd_baseline_report import main as render_report

def main():
    base=ROOT/'stain_research/runs/hiesd_baseline_seed42_20261116'
    weighted=ROOT/'stain_research/runs/hiesd_weighted_seed42_20261116'
    out=ROOT/'stain_research/results/hiesd_weighted_diagnostic_20261116'
    a=json.loads((base/'RESULTS.json').read_text());b=json.loads((weighted/'RESULTS.json').read_text())
    ca=json.loads((base/'config.json').read_text());cb=json.loads((weighted/'config.json').read_text())
    assert a['stage']==b['stage']=='complete' and a['reload_predictions_match'] and b['reload_predictions_match']
    for k in ca:
        if k not in ['arm','source_sha256']:assert ca[k]==cb[k],k
    assert digest(base/'initial.pt')==digest(weighted/'initial.pt')
    preserved=json.loads((out/'baseline_preservation.json').read_text())
    assert all(digest(base/name)==sha for name,sha in preserved.items())
    with (base/'validation_predictions.csv').open() as stream:pa=list(csv.DictReader(stream))
    with (weighted/'validation_predictions.csv').open() as stream:pb=list(csv.DictReader(stream))
    assert len(pa)==len(pb)==1042
    assert all(all(x[k]==y[k] for k in ['image_path','patient_id','slide_id','class_id']) for x,y in zip(pa,pb))
    train_counts=np.array(cb['class_counts_train']);expected=sum(train_counts)/(5*train_counts)
    assert np.array_equal(expected,np.array(cb['class_weights']))
    render_report(weighted)
    ma=a['validation'];mb=b['validation']
    comparison={'scope':'One post-hoc seed pair on a selected, annotation-guided validation subset; descriptive only.',
                'baseline':ma,'weighted':mb,'delta_macro_f1':mb['macro_f1']-ma['macro_f1'],'delta_accuracy':mb['accuracy']-ma['accuracy'],
                'baseline_completed_epochs':a['completed_epochs'],'weighted_completed_epochs':b['completed_epochs'],
                'same_initial_checkpoint':True,'same_common_config':True,'same_ordered_validation_cases':True,
                'original_baseline_artifacts_preserved':True,'weights_fit_training_only':True,'class_weights':cb['class_weights'],
                'reserved_test_used':False,'inferential_statistics':'None; single seed, clustered and selected validation cases.',
                'automatic_additional_runs':False}
    save(out/'COMPARISON.json',comparison)
    text=['# HiESD class-weighted diagnostic follow-up','',
          'One post-hoc comparison, seed 42. Same initial weights, split, geometry, normalization, optimizer, scheduler rule and maximum budget. Only the training loss receives inverse-frequency class weights. The validation NLL and evaluation rules remain unweighted.',
          '', '| Metric | Standard | Class-weighted |','|---|---:|---:|',
          f"|Validation accuracy|{ma['accuracy']:.4f}|{mb['accuracy']:.4f}|",
          f"|Validation macro-F1|{ma['macro_f1']:.4f}|{mb['macro_f1']:.4f}|",
          f"|Selected epoch|{a['best_epoch']}|{b['best_epoch']}|",
          f"|Completed epochs|{a['completed_epochs']}|{b['completed_epochs']}|",'',
          '| Class | Validation support | Standard recall | Weighted recall | Standard F1 | Weighted F1 |',
          '|---|---:|---:|---:|---:|---:|']
    for name in NAMES:
        x=ma['classification_report'][name];y=mb['classification_report'][name]
        text.append(f"|{name}|{int(x['support'])}|{x['recall']:.4f}|{y['recall']:.4f}|{x['f1-score']:.4f}|{y['f1-score']:.4f}|")
    text+=['',f"Macro-F1 difference: {comparison['delta_macro_f1']:+.4f}; accuracy difference: {comparison['delta_accuracy']:+.4f}. These are descriptive differences, not significance or generalization claims.",
           '', 'Weights use N_train/(5*n_class) and were fixed from training counts without tuning: '+', '.join(f'{n}={w:.6f}' for n,w in zip(NAMES,cb['class_weights']))+'.',
           '', 'Audit: all 1811 coarse annotation regions recomputed; 71 stratified source RGB reconstructions matched exactly. No manifest label-index inconsistency, exact train/validation patch overlap or patient overlap was detected. This is not expert validation of pathology labels or landmark registration. Some reviewed validation "other" patches appeared blurred; they were retained without post-hoc relabeling or filtering.',
           '', 'The subset is small, annotation-selected and strongly clustered: 82/83 validation tumor patches and 242/256 validation metaplasia patches belong to Patient30; 104/110 training lymphoid patches belong to Patient13. Only twelve validation lymphoid patches are present. Neither arm is a clinical model or an unbiased benchmark result.',
           '', 'Class weighting is an imbalance control, not a novel method and not evidence for the stain controller. Do not automatically search more weight formulas, change the validation split, or access reserved test data in response to this comparison.',
           '', 'Original baseline checkpoint, configuration, history and predictions hashes were checked unchanged after the follow-up. Latest/best weighted checkpoints stay isolated from the dashboard.']
    (out/'COMPARISON.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(f'Comparison saved: {out}',flush=True)

if __name__=='__main__':main()
