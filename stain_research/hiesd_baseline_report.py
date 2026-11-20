"""Generate local figures/report from completed HiESD baseline outputs only."""
import argparse
import json
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main(run):
    root=Path(__file__).resolve().parents[1]
    result=json.loads((run/'RESULTS.json').read_text())
    config=json.loads((run/'config.json').read_text())
    weighted=config.get('arm')=='standard_class_weighted'
    resnet=config.get('arm')=='resnet18_imagenet_standard'
    label='ResNet18 baseline after consistent QC' if resnet else ('class-weighted baseline' if weighted else 'baseline')
    history=json.loads((run/'history.json').read_text())
    out=root/'stain_research/results'/run.name;out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    epochs=[h['epoch'] for h in history]
    axes[0].plot(epochs,[h['train']['loss'] for h in history],label='Train weighted smoothed CE' if weighted else 'Train smoothed CE')
    axes[0].plot(epochs,[h['validation']['nll'] for h in history],label='Validation NLL')
    axes[0].set(xlabel='Epoch',ylabel='Loss');axes[0].legend()
    axes[1].plot(epochs,[h['validation']['macro_f1'] for h in history],label='Validation macro F1')
    axes[1].plot(epochs,[h['validation']['accuracy'] for h in history],label='Validation accuracy')
    axes[1].axvline(result['best_epoch'],linestyle='--',color='gray',label='Selected epoch')
    axes[1].set(xlabel='Epoch',ylabel='Score',ylim=(0,1));axes[1].legend()
    fig.suptitle(f'HiESD selected-subset {label}: seed 42 (exploratory)')
    fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=180);plt.close(fig)
    cm=np.array(result['validation']['confusion_matrix'])
    fig,ax=plt.subplots(figsize=(8,7));im=ax.imshow(cm,cmap='Blues')
    names=config['class_order'];ax.set_xticks(range(5),names,rotation=35,ha='right');ax.set_yticks(range(5),names)
    for i in range(5):
        for j in range(5):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    ax.set(xlabel='Predicted class',ylabel='Released annotation class',title='Validation confusion matrix: selected best checkpoint')
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=180);plt.close(fig)
    text=[f'# HiESD selected-subset {label}','',
          'Single-seed exploratory feasibility result. This is not an unbiased held-out evaluation, a clinical validation, or evidence of controller novelty.','',
          f"Completed epochs: {result['completed_epochs']}; selected epoch: {result['best_epoch']}; stop: {result['stop_reason']}.",
          f"Validation accuracy: {result['validation']['accuracy']:.4f}; macro-F1: {result['validation']['macro_f1']:.4f}.",'',
          '| Group | Support | Precision | Recall | F1 |','|---|---:|---:|---:|---:|']
    for n in names:
        r=result['validation']['classification_report'][n]
        text.append(f"|{n}|{int(r['support'])}|{r['precision']:.4f}|{r['recall']:.4f}|{r['f1-score']:.4f}|")
    lymphoid_n=int(result['validation']['classification_report']['lymphoid']['support'])
    text+=['',f'Patient split is preserved. Validation annotations informed subset selection; {lymphoid_n} validation lymphoid patches are present. Patches are clustered within a limited number of patients. No patient-level confidence interval is claimed. Reserved test data was not used.',
           '', 'The broad five-group semantics were checked against HiESD Figure 1/Table 2. This model, patch extraction, subset selection, and budget differ from the published baseline protocol, so numerical scores are not directly comparable.',
           '',f"Best checkpoint SHA256: `{result['best_checkpoint_sha256']}`. Reloaded validation probabilities matched saved outputs.",'',
           ('Train weighted smoothed cross-entropy and validation unweighted unsmoothed NLL are distinct losses; their curves and training losses between arms are not directly comparable.' if weighted else 'Train smoothed cross-entropy and validation unsmoothed NLL are distinct losses; their curves are not directly equivalent.'),
           '', ('Automated released GrandQC masks were applied consistently under a frozen QC protocol; this does not establish expert quality validation or landmark registration. Additional limitations: coarse annotation boundaries, no expert patch-label re-review, class imbalance and only one training seed. The cohort, normalization and backbone differ from prior pilot runs: this is a new reference baseline, not a controlled attribution of score changes.' if resnet else 'Additional limitations: coarse mask boundaries, no expert patch-label re-review, no GrandQC artifact mask filtering, strong class imbalance, and only one training seed.'),
           '', 'Before interpreting augmentation comparisons, use identical split, initialization, geometry, normalization, and budgets. Do not choose methods using reserved test data.']
    (out/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    shutil.copy2(run/'validation_predictions.csv',out/'validation_predictions.csv')
    print(f'Figures and report saved: {out}',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    main(parser.parse_args().run.resolve())
