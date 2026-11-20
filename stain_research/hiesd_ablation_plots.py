"""Draw saved validation curves; no inference or fitting."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hiesd_ablation_report import CAM,BASE,ARMS,ROOT,summarize

def main():
    rows=summarize();out=ROOT/'stain_research/results/hiesd_resnet_ablation_v1'
    fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True)
    for seed,ax in zip([42,43,44],axes):
        for arm in ARMS:
            run=BASE if seed==42 and arm=='standard' else CAM/f'seed{seed}_{arm}'
            if not (run/'history.json').exists():continue
            h=json.loads((run/'history.json').read_text())
            ax.plot([r['epoch'] for r in h],[r['validation']['macro_f1'] for r in h],label=arm)
        ax.set(title=f'Seed {seed}',xlabel='Epoch',ylim=(0,1));ax.legend(fontsize=8)
    axes[0].set_ylabel('Validation macro-F1')
    fig.suptitle('HiESD selected-cohort matched augmentation comparison (exploratory)')
    fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=180);plt.close(fig)
    print(f'Saved {len(rows)} completed-run summaries and learning curves: {out}')

if __name__=='__main__':main()
