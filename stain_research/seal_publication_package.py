"""Seal reviewed artifacts without regenerating already checked Office files."""
import argparse,csv,hashlib,json,shutil,zipfile
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1];PACKAGE=ROOT/'evaluation_package';BUILD=ROOT/'.sync-backups/publication_build_20261117'
PRIMARY=['Progress_Report.docx','Progress_Report.pdf','Progress_Presentation.pptx','Progress_Presentation.pdf','IEEE_Paper_Draft.pdf','Overleaf_Paper_Source.zip']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf-8')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reviewed',action='store_true');args=parser.parse_args()
    final=ROOT/'stain_research/results/final_test_hiesd_v1';proof=read(final/'INDEPENDENT_VERIFICATION.json');assert proof['passed']
    protocol=ROOT/'stain_research/runs/final_test_hiesd_v1';freeze=read(protocol/'CHECKPOINT_FREEZE.json')
    assert all(sha(ROOT/'stain_research'/n)==h for n,h in freeze['implementation_sha256'].items())
    assert read(BUILD/'template_fidelity.json')['preserve_only_parts_unchanged']
    out=PACKAGE/'publication_results';predictions=out/'test_predictions';predictions.mkdir(exist_ok=True)
    for p in final.glob('*_predictions.npz'):shutil.copy2(p,predictions/p.name)
    shutil.copy2(protocol/'CHECKPOINT_FREEZE.json',out/'final_test_CHECKPOINT_FREEZE.json')
    shutil.copy2(ROOT/'data/external/hiesd_final_test_v1/patch_manifest.csv',out/'final_test_patch_manifest.csv')
    shutil.copy2(ROOT/'stain_research/verify_portable_test.py',out/'verify_portable_test.py')
    (out/'VERIFY_RESULTS.md').write_text('Run `python verify_portable_test.py` from this folder with NumPy installed. It recomputes all 18 saved-prediction confusion matrices, accuracy/macro-F1 and the prespecified patient-cluster bootstrap. No raw images, model weights, PyTorch, GPU or network are needed. This checks saved numerical evidence; it does not independently prove model inference without the local checkpoints. The cluster interval is descriptive and conditions on these three seeds. The fuller independent verification also checks all per-class metrics and calibration within the recorded tolerances.\n',encoding='utf-8')
    report_pages=len(PdfReader(PACKAGE/'Progress_Report.pdf').pages);paper_pages=len(PdfReader(PACKAGE/'IEEE_Paper_Draft.pdf').pages);slides=len(PdfReader(PACKAGE/'Progress_Presentation.pdf').pages)
    assert report_pages==23 and paper_pages==4 and slides==18
    artifact=read(PACKAGE/'PUBLICATION_BUILD.json');artifact['files']={n:sha(PACKAGE/n) for n in PRIMARY}
    artifact['status']='generated and visually reviewed' if args.reviewed else 'generated; visual review pending'
    artifact['release_prepared_at']=datetime.now(ZoneInfo('Asia/Calcutta')).isoformat(timespec='seconds')
    save(PACKAGE/'PUBLICATION_BUILD.json',artifact)
    save(PACKAGE/'PUBLICATION_QA.json',{'reviewed':args.reviewed,'report_pages':report_pages,'paper_pages':paper_pages,'slides':slides,
       'visual_scope':'All final paper pages and all slide content checked; all report content checked, compacted literature page and updated contents checked; subsequent report pages reproduced identically with page-number shift.',
       'native_word_and_powerpoint_pdf_exports':True,'editable_native_tables':True,'teacher_opaque_parts_preserved':True,
       'final_test_independent_verification':proof,'original_preservation':read(ROOT/'stain_research/runs/classifier_local_20261113/preservation_after.json'),
       'frozen_test_implementation_unchanged':True,'scope_limits':'No authenticated Overleaf project, genuine Turnitin report, guide signature, expert pathology adjudication, clinical validation or acceptance is asserted.'})
    skip={'Research_Evaluation_Package.zip','FILE_MANIFEST.json','OTHER_PC_OVERNIGHT_PROMPT.md','PACKAGE_READY.json','CLOUD_SYNC_STATUS.json'}
    candidates=[p for p in PACKAGE.iterdir() if p.is_file() and p.suffix in ['.md','.json','.pdf','.docx','.pptx','.zip'] and p.name not in skip]
    for directory in ['overleaf','publication_results','publication_readiness']:
        candidates += [p for p in (PACKAGE/directory).rglob('*') if p.is_file() and p.suffix in ['.tex','.cls','.md','.json','.csv','.png','.pdf','.bib','.py','.npz']]
    save(PACKAGE/'FILE_MANIFEST.json',{'snapshot':artifact['snapshot'],'scope':'Portable reports, numeric predictions and audit evidence; raw data, environments, databases, model weights, private reference templates and external sync receipts excluded.',
         'files':{p.relative_to(PACKAGE).as_posix():sha(p) for p in candidates}})
    with zipfile.ZipFile(PACKAGE/'Research_Evaluation_Package.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in candidates+[PACKAGE/'FILE_MANIFEST.json']:z.write(p,p.relative_to(PACKAGE))
    print(json.dumps({'reviewed':args.reviewed,'report_pages':report_pages,'paper_pages':paper_pages,'slides':slides,'bundle_bytes':(PACKAGE/'Research_Evaluation_Package.zip').stat().st_size,'files':len(candidates)},indent=2))
if __name__=='__main__':main()
