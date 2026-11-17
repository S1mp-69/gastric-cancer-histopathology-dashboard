"""Refresh the single evaluation package from saved evidence, without inference.

Called once now and after the bounded final-test pipeline. No API credits,
research checkpoint activation, or modification of completed experiments.
"""
import os
os.environ['PYTHONUTF8']='1'
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import csv,hashlib,json,shutil,statistics,subprocess,sys,time,zipfile
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]
RUNTIME=Path('C:/Users/gxhyp/.cache/codex-runtimes/codex-primary-runtime/dependencies')
BUILD=ROOT/'.sync-backups/publication_build_20261117'
PACKAGE=ROOT/'evaluation_package'
LABELS={'standard':'Standard','residual_fixed':'Residual fixed','uniform_weaker':'Uniform weaker','shuffled':'Global shuffled','adaptive':'Adaptive','within_class_shuffled':'Within-class shuffled'}
REFS=[
 '[1] S. Lou et al., A large histological images dataset of gastric cancer with tumour microenvironment annotation for AI, Scientific Data, 12, 138, 2025. doi:10.1038/s41597-025-04489-9.',
 '[2] C. Wang et al., A fully annotated pathology slide dataset for early gastric cancer and precancerous lesions, Scientific Data, 2025. doi:10.1038/s41597-025-05679-1.',
 '[3] T. Kim et al., Stain SAN: simultaneous augmentation and normalization for histopathology images, Journal of Medical Imaging, 11(4), 044006, 2024. doi:10.1117/1.JMI.11.4.044006.',
 '[4] N. Marini et al., Data-driven color augmentation for H&E stained images in computational pathology, Journal of Pathology Informatics, 14, 100183, 2023. doi:10.1016/j.jpi.2022.100183.',
 '[5] J. Brabec, T. Komarek, V. Franc and L. Machlica, On Model Evaluation under Non-constant Class Imbalance, ICCS, 2020; arXiv:2001.05571.',
 '[6] J. Opitz, A Closer Look at Classification Evaluation Metrics and a Critical Reflection of Common Evaluation Practice, Transactions of the ACL, 12, pp.820–836, 2024. doi:10.1162/tacl_a_00675.',
 '[7] Local evidence: stain_research/results/hiesd_resnet_ablation_v1; results/shuffled_20261116; publication_readiness/publication_audit_20261117; ranking_audit_20261117; frozen final-test protocol and checkpoint freeze. Snapshot '+datetime.now(ZoneInfo('Asia/Calcutta')).isoformat(timespec='minutes')+'.',
 '[8] KIIT School of Computer Science and Engineering, Project_Report-format..docx, institutional template supplied by the student. Sample template results are not project evidence.'
]
def read(path,default=None):
    try:return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:return default
def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def snapshot():
    original=read(ROOT/'stain_research/results/hiesd_resnet_ablation_v1/summary.json')
    within=ROOT/'stain_research/runs/hiesd_within_class_v1'
    runs=list(original['runs'])
    for seed in [42,43,44]:
        r=read(within/f'within_seed{seed}/RESULTS.json')
        if r and r['stage']=='complete':
            runs.append({'seed':seed,'arm':'within_class_shuffled','macro_f1':r['validation']['macro_f1'],'accuracy':r['validation']['accuracy'],
                         'epochs':r['completed_epochs'],'best_epoch':r['best_epoch'],'tumor_recall':r['validation']['classification_report']['tumor']['recall'],
                         'other_recall':r['validation']['classification_report']['other']['recall']})
    summaries=[]
    for arm,label in LABELS.items():
        items=[r for r in runs if r['arm']==arm];v=[r['macro_f1'] for r in items]
        summaries.append({'arm':arm,'label':label,'n':len(items),'macro_f1_mean':statistics.mean(v) if v else None,'macro_f1_sd':statistics.stdev(v) if len(v)>1 else 0.})
    complete=sum(r['n'] for r in summaries)
    status=read(within/'STATUS.json',{})
    wstatus=f"{summaries[-1]['n']} of three seeds complete"
    if status.get('stage')=='training':wstatus+=f"; seed {status.get('seed')} active"
    test=read(ROOT/'stain_research/results/final_test_hiesd_v1/RESULTS.json')
    final_status=read(ROOT/'stain_research/runs/final_test_hiesd_v1/STATUS.json',{})
    failure=read(ROOT/'stain_research/runs/final_test_hiesd_v1/FAILURE.json')
    if test:
        ci=test['descriptive_patient_cluster_interval']
        test_text=(f"The frozen final test completed on {test['patches']} patches from {test['patients']} reserved patients. "
        f"The prespecified adaptive-minus-within-class-shuffled mean macro-F1 difference was {test['primary_adaptive_minus_within_class']:+.4f}, "
        f"with descriptive patient-cluster interval [{ci[0]:+.4f}, {ci[1]:+.4f}]. "
        f"Of 1,000 cluster draws, {test['bootstrap_draws_missing_classes']} lacked at least one class; fixed five-class zero-division-zero scoring was retained. "
        "Only seven patient clusters limit precision. All models were frozen before acquisition; no tuning follows test access. This is held-out public-release testing, not independent clinical external validation.")
    elif failure:test_text='The frozen final-test workflow encountered a blocker: '+failure['error']+'. No complete final-test claim is made. Consult FAILURE.json and preserved logs; resume the same protocol rather than changing the cohort or models.'
    else:test_text=('The final-test protocol was frozen before reserved-pixel acquisition. It evaluates all six arms and seeds 42, 43 and 44 using validation-selected best checkpoints, with adaptive-minus-within-class-shuffled patch macro-F1 as the primary contrast. The seven reserved patients cover 13 source slides (about 7.53 GiB). Model hashes are frozen before acquisition. No tuning follows test access. Current stage: '+final_status.get('stage','waiting for within-class controls')+'. No final-test scores are available in this snapshot.')
    abstract=('We evaluate whether source-image decomposition diagnostics usefully assign residual-preserving stain-augmentation strength. '
    'An eight-class HMU study uses an exact-duplicate-controlled image split without verified patient mapping; a five-group HiESD study uses a quality-screened patient-grouped subset with ImageNet-pretrained ResNet18. '
    'Standard, fixed, uniformly weaker, global shuffled and adaptive augmentation share seed-specific initial weights, geometry and bounded training budgets. '
    'Across three seeds, HMU adaptive mean validation macro-F1 is 0.6836 versus 0.6890 for shuffled; HiESD adaptive is 0.7131 versus 0.7350 for shuffled and 0.7270 for standard. '
    'These completed results do not establish adaptive superiority. A within-class permutation control isolates source-specific assignment while preserving each class strength bank. '
    + ('The frozen reserved-patient test is reported separately. ' if test else 'The frozen reserved-patient test remains pending or active. ')
    +'Exact-duplicate auditing and independent metric verification improve reproducibility. Reweighting saved predictions illustrates known prevalence sensitivity rather than a novel metric. Rare-class support, coarse annotations, selected patient coverage and development reuse limit inference; clinical benefit and novelty are unconfirmed.')
    return {'as_of':datetime.now(ZoneInfo('Asia/Calcutta')).isoformat(timespec='seconds'),'as_of_display':datetime.now(ZoneInfo('Asia/Calcutta')).strftime('%d November %Y, %H:%M IST'),
      'abstract':abstract,'hmu':[{'label':LABELS[a],'mean':m} for a,m in [('standard',.6775761972821549),('residual_fixed',.682201),('uniform_weaker',.680422),('shuffled',.689041),('adaptive',.683598)]],
      'hiesd':summaries,'hiesd_runs':runs,'completed_hiesd_runs':complete,'within_status':wstatus,'test_complete':bool(test),'test':test,'test_text':test_text,
      'labels':LABELS,'references':REFS,'final_status':final_status,'failure':failure}
def figures(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.4,3.4));ax.axis('off')
    boxes=[(.02,.72,'Browser → FastAPI\noriginal CNN/LSTM predictor'),(.56,.72,'Eight-class probabilities\nSQLAlchemy → isolated SQLite'),
      (.02,.30,'Pinned data + split + QC\nversioned train-fitted cache'),(.37,.30,'Random augmentation\nmatched ResNet18 arms'),(.75,.30,'Best / last states\nsaved predictions')]
    for x,y,t in boxes:ax.text(x,y,t,fontsize=9.5,va='center',transform=ax.transAxes,bbox=dict(boxstyle='square,pad=.5',fc='white',ec='#163F48'))
    for x1,x2,y in [(.37,.54,.72),(.30,.35,.30),(.66,.73,.30)]:ax.annotate('',(x2,y),(x1,y),xycoords='axes fraction',arrowprops=dict(arrowstyle='->'))
    ax.text(.02,.05,'Frozen checkpoint set → reserved-patient test → auditable reports (no tuning after access)',fontsize=10,transform=ax.transAxes)
    fig.savefig(BUILD/'publication_architecture.png',dpi=180,bbox_inches='tight');plt.close(fig)
    rows=[r for r in data['hiesd'] if r['n']==3]
    fig,ax=plt.subplots(figsize=(8,3.4));ax.bar(range(len(rows)),[r['macro_f1_mean'] for r in rows],yerr=[r['macro_f1_sd'] for r in rows],capsize=4,color='#417D83')
    ax.set_xticks(range(len(rows)),[r['label'].replace(' ','\n') for r in rows]);ax.set_ylabel('Validation macro-F1');ax.set_ylim(0,.85);ax.spines[['top','right']].set_visible(False)
    for i,r in enumerate(rows):ax.text(i,r['macro_f1_mean']+r['macro_f1_sd']+.015,f"{r['macro_f1_mean']:.4f}",ha='center',fontsize=9)
    fig.tight_layout();fig.savefig(BUILD/'publication_validation.png',dpi=180);plt.close(fig)
def run(command,timeout=180):
    completed=subprocess.run([str(x) for x in command],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8',errors='replace',timeout=timeout,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    with (BUILD/'build.log').open('a',encoding='utf-8') as f:f.write('\n'+str(command)+'\n'+completed.stdout)
    if completed.returncode:raise RuntimeError('Artifact command failed; see '+str(BUILD/'build.log'))
def main():
    BUILD.mkdir(parents=True,exist_ok=True)
    backup=ROOT/'.sync-backups/evaluation_package_prepublication_20261117'
    if not backup.exists():
        backup.mkdir()
        for p in PACKAGE.iterdir():
            if p.is_file():shutil.copy2(p,backup/p.name)
        shutil.copytree(PACKAGE/'overleaf',backup/'overleaf')
    data=snapshot();write(BUILD/'publication_snapshot.json',data)
    from export_publication_evidence import main as export_evidence
    export_evidence()
    figures(data)
    # Native document template construction uses the bundled authoring runtime.
    run([RUNTIME/'python/python.exe','-c',"import sys,json;from pathlib import Path;sys.path.insert(0,'stain_research');from build_publication_report import build;p=Path('.sync-backups/publication_build_20261117');build(json.loads((p/'publication_snapshot.json').read_text(encoding='utf-8')),p)"])
    from publication_text_materials import build_text
    build_text(data,BUILD,PACKAGE)
    run([RUNTIME/'node/bin/node.exe',ROOT/'stain_research/build_publication_deck.mjs'],timeout=300)
    run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',ROOT/'stain_research/office_export.ps1','-InputDocx',BUILD/'Progress_Report.docx','-PaginationJson',BUILD/'pagination.json','-OutputPdf',BUILD/'Progress_Report.pdf'])
    run([RUNTIME/'python/python.exe','-c',"import sys;sys.path.insert(0,'stain_research');from build_teacher_report import cache_pagination;cache_pagination('.sync-backups/publication_build_20261117/Progress_Report.docx','.sync-backups/publication_build_20261117/pagination.json')"])
    run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',ROOT/'stain_research/office_export.ps1','-InputPptx',BUILD/'Progress_Presentation.pptx','-OutputPdf',BUILD/'Progress_Presentation.pdf'])
    run([ROOT/'.sync-backups/evaluation_build/tectonic/tectonic.exe','--keep-logs','--outdir',BUILD,PACKAGE/'overleaf/main.tex'],timeout=180)
    shutil.copy2(BUILD/'main.pdf',BUILD/'IEEE_Paper_Draft.pdf')
    portable=PACKAGE/'publication_results';portable.mkdir(exist_ok=True)
    write(portable/'PUBLICATION_SNAPSHOT.json',data)
    for source in [ROOT/'stain_research/results/hiesd_resnet_ablation_v1/summary.json',ROOT/'evaluation_package/publication_readiness/publication_audit_20261117/AUDIT.json',ROOT/'stain_research/runs/final_test_hiesd_v1/PROTOCOL.json']:
        shutil.copy2(source,portable/(source.parent.name+'_'+source.name))
    final=ROOT/'stain_research/results/final_test_hiesd_v1'
    if final.exists():
        for p in final.glob('*.json'):shutil.copy2(p,portable/('final_test_'+p.name))
    # Keep previous safe files until every essential export succeeds.
    for name in ['Progress_Report.docx','Progress_Report.pdf','Progress_Presentation.pptx','Progress_Presentation.pdf','IEEE_Paper_Draft.pdf']:
        shutil.copy2(BUILD/name,PACKAGE/name)
    with zipfile.ZipFile(PACKAGE/'Overleaf_Paper_Source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in (PACKAGE/'overleaf').rglob('*'):
            if p.is_file() and p.suffix not in ['.aux','.log','.pdf']:z.write(p,p.relative_to(PACKAGE/'overleaf'))
    receipt={'snapshot':data['as_of'],'completed_hiesd_runs':data['completed_hiesd_runs'],'test_complete':data['test_complete'],
      'files':{n:digest(PACKAGE/n) for n in ['Progress_Report.docx','Progress_Report.pdf','Progress_Presentation.pptx','Progress_Presentation.pdf','IEEE_Paper_Draft.pdf','Overleaf_Paper_Source.zip']},
      'build_log':str(BUILD/'build.log'),'status':'generated; final visual inspection required after each changed export','overleaf_authenticated_project':'pending','turnitin':'pending genuine institutional reports'}
    write(PACKAGE/'PUBLICATION_BUILD.json',receipt)
    print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
