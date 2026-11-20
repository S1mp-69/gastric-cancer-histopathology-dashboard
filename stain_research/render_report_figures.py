"""Render scientific report figures in the local research plotting environment."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rows=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2])
labels={'standard':'Standard','residual_fixed':'Residual fixed','uniform_weaker':'Uniform weaker','adaptive':'Adaptive'}
fig,ax=plt.subplots(figsize=(8,3.6));ax.set_axis_off()
boxes=[(.02,.66,'Upload → FastAPI\noriginal predictor'),(.42,.66,'Eight probabilities\nSQLAlchemy → SQLite'),
       (.02,.14,'Train / validation\nversioned stain cache'),(.39,.14,'Random augmentation\nCNN → BiLSTM → attention'),(.80,.14,'Best / last\ncheckpoints')]
for x,y,t in boxes:ax.text(x,y,t,ha='left',va='center',fontsize=9,bbox=dict(boxstyle='square,pad=.6',facecolor='white',edgecolor='black'),transform=ax.transAxes)
for start,end in [((.31,.66),(.41,.66)),((.30,.14),(.38,.14)),((.73,.14),(.79,.14))]:ax.annotate('',xy=end,xytext=start,xycoords='axes fraction',arrowprops=dict(arrowstyle='->'))
ax.text(.02,.94,'Application path',weight='bold',transform=ax.transAxes)
ax.text(.02,.39,'Isolated research path — no test data or dashboard activation',weight='bold',transform=ax.transAxes,fontsize=10)
fig.savefig(out/'architecture.png',dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig)
fig,ax=plt.subplots(figsize=(7.4,3.3));ys=[r['macro_f1'] or 0 for r in rows]
analysis_path=Path(__file__).resolve().parents[1]/'stain_research/results/final_campaign_20261113/analysis.json'
errors=None
if analysis_path.exists():
    aggregate=json.loads(analysis_path.read_text());rows=aggregate['summary'];ys=[r['macro_f1_mean'] for r in rows];errors=[r['macro_f1_sd'] for r in rows]
bars=ax.bar([labels[r['arm']] for r in rows],ys,yerr=errors,capsize=4,color=['#bfbfbf','#dddddd','#999999','#555555'],edgecolor='black')
ax.set_ylim(0,1);ax.set_ylabel('Validation macro F1');ax.spines[['top','right']].set_visible(False)
for bar,y in zip(bars,ys):ax.text(bar.get_x()+bar.get_width()/2,y+.02,f'{y:.4f}',ha='center',fontsize=10)
fig.tight_layout();fig.savefig(out/'validation_comparison.png',dpi=180,facecolor='white');plt.close(fig)
if analysis_path.exists():
    evidence_dir=analysis_path.parent;campaign=evidence_dir.parents[1]/'runs/overnight_20261113'
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    for ax,r in zip(axes.flat,aggregate['summary']):
        import numpy as np
        matrix=np.array(r['confusion_matrix_sum'],dtype=float);matrix/=matrix.sum(axis=1,keepdims=True)
        ax.imshow(matrix,vmin=0,vmax=1,cmap='Greys');ax.set_title(labels[r['arm']]);ax.set_xticks(range(8),['ADI','DEB','LYM','MUC','MUS','NOR','STR','TUM'],rotation=45);ax.set_yticks(range(8),['ADI','DEB','LYM','MUC','MUS','NOR','STR','TUM'])
        ax.set_xlabel('Predicted');ax.set_ylabel('True')
        for i in range(8):
            for j in range(8):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if matrix[i,j]>.5 else 'black')
    fig.suptitle('Row-normalized confusion matrices summed across three seeds\nSame validation images repeated; not independent patient observations')
    fig.tight_layout();fig.savefig(evidence_dir/'confusion_matrices.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
    for ax,arm in zip(axes.flat,labels):
        for seed in [42,43,44]:
            hist=json.loads((campaign/f'seed{seed}_{arm}/history.json').read_text())
            ax.plot([v['epoch'] for v in hist],[v['val']['macro_f1'] for v in hist],label=str(seed))
        ax.set_title(labels[arm]);ax.set_xlabel('Epoch');ax.set_ylabel('Validation macro F1');ax.legend(title='Seed');ax.grid(alpha=.2)
    fig.suptitle('Learning curves — stopping at 40 epochs does not prove convergence');fig.tight_layout();fig.savefig(evidence_dir/'learning_curves.png',dpi=180);plt.close(fig)
