"""Refresh the ONE transferred evaluation package from actual campaign files.

Uses the bundled artifact runtime for DOCX/PPTX work, native Word for PDF export,
and preserves the transferred source privately. No authors, scores or links are
fabricated. The supervisor can run this after Codex is unavailable.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import zipfile
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=Path('C:/Users/gxhyp/.cache/codex-runtimes/codex-primary-runtime/dependencies')
TZ=ZoneInfo('Asia/Calcutta')
ARMS=['standard','residual_fixed','uniform_weaker','adaptive']
LABELS={'standard':'Standard','residual_fixed':'Residual fixed','uniform_weaker':'Uniform weaker','adaptive':'Adaptive'}
CLASSES=['ADI','DEB','LYM','MUC','MUS','NOR','STR','TUM']


def read(path,default=None):
    try:return json.loads(Path(path).read_text(encoding='utf-8'))
    except (FileNotFoundError,json.JSONDecodeError):return default


def save(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');os.replace(temporary,path)


def snapshot(campaign):
    config=read(campaign/'campaign_config.json',{})
    rows=[]
    for seed in [42,43,44]:
        for arm in ARMS:
            folder=campaign/f'seed{seed}_{arm}'
            history=read(folder/'history.json',[]);metrics=read(folder/'metrics.json')
            progress=read(folder/'progress.json',{})
            accepted=None
            for entry in history:
                if accepted is None or entry['val']['macro_f1']>accepted['val']['macro_f1']+.0001:
                    accepted=entry
            val=metrics['validation'] if metrics else (accepted['val'] if accepted else None)
            rows.append({'arm':arm,'seed':seed,'status':'Complete' if metrics else ('Active' if progress else 'Queued'),
                         'epoch':metrics['completed_epochs'] if metrics else progress.get('epoch',0),
                         'best_epoch':metrics['best_epoch'] if metrics else (accepted['epoch'] if accepted else None),
                         'accuracy':val['accuracy'] if val else None,'macro_f1':val['macro_f1'] if val else None,
                         'per_class':val.get('per_class',{}) if val else {},
                         'failures_by_class':metrics['failures_by_class'] if metrics else progress.get('failures_by_class',{}),
                         'checkpoint_reload_verified':bool(metrics and metrics.get('checkpoint_reload_verified')),
                         'stop_reason':metrics.get('stop_reason') if metrics else None,
                         'mean_epoch_seconds':metrics.get('mean_epoch_seconds') if metrics else (statistics.mean(e['epoch_seconds'] for e in history) if history else None)})
    t=datetime.now(TZ)
    benchmark=read(campaign/'throughput_benchmark.json',{})
    return {'final_analysis':read(ROOT/'stain_research/results/final_campaign_20261113/analysis.json'),
            'as_of':t.isoformat(timespec='seconds'),'as_of_display':t.strftime('%d %b %Y %H:%M IST'),
            'campaign':'overnight_20261113','first_seed':[r for r in rows if r['seed']==42],
            'all_runs':rows,'config':config,
            'throughput':benchmark.get('selected',{}).get('images_per_second',0),
            'benchmark':benchmark,'completed_runs':sum(r['status']=='Complete' for r in rows),
            'patient_independence':False,'test_evaluated':False,
            'author':read(ROOT/'stain_research/report_identity.json',{}).get('name','Amlan'),
            'report_template_status':('Supplied KIIT template; certificate unsigned and genuine plagiarism report pending'
                 if (ROOT/'evaluation_package/templates/Project_Report_Format.docx').exists() else 'Provisional; prescribed institutional template unavailable'),
            'overleaf_status':'Unauthenticated login page observed; no project imported; uploadable ZIP retained'}


def table_style(table,widths):
    from docx.shared import Inches,Pt
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
    table.alignment=WD_TABLE_ALIGNMENT.CENTER;table.autofit=False
    for i,row in enumerate(table.rows):
        for j,cell in enumerate(row.cells):
            cell.width=Inches(widths[j]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            properties=cell._tc.get_or_add_tcPr()
            shading=OxmlElement('w:shd');shading.set(qn('w:fill'),'163F48' if i==0 else ('EDF2F1' if i%2==0 else 'FFFFFF'));properties.append(shading)
            margins=OxmlElement('w:tcMar')
            for edge in ['top','left','bottom','right']:
                e=OxmlElement('w:'+edge);e.set(qn('w:w'),'100');e.set(qn('w:type'),'dxa');margins.append(e)
            properties.append(margins)
            for p in cell.paragraphs:
                p.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_after=Pt(2);p.paragraph_format.space_before=Pt(2)
                for run in p.runs:
                    run.font.size=Pt(10.5);run.font.name='Calibri';run.bold=i==0
                    from docx.shared import RGBColor
                    run.font.color.rgb=RGBColor.from_string('FFFFFF' if i==0 else '000000')
    borders=OxmlElement('w:tblBorders')
    for edge in ['top','left','bottom','right','insideH','insideV']:
        e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');borders.append(e)
    table._tbl.tblPr.append(borders)
    header=OxmlElement('w:tblHeader');table.rows[0]._tr.get_or_add_trPr().append(header)


def document(data,build):
    template=ROOT/'evaluation_package/templates/Project_Report_Format.docx'
    if template.exists() and (ROOT/'stain_research/report_identity.json').exists():
        from build_teacher_report import build as build_teacher_report
        return build_teacher_report(data,build,template)
    from docx import Document
    from docx.shared import RGBColor
    source=ROOT/'.sync-backups/evaluation_package_original_20261113/Progress_Report.docx'
    d=Document(source)
    d.paragraphs[2].text='Provisional progress report'
    for name in ['Title','Subtitle','Heading 1','Heading 2']:
        d.styles[name].font.color.rgb=RGBColor(0,0,0)
    # The retained generic report controls the typography and existing sections.
    # Append new sections; keep all historical evidence and reference tables.
    d.add_page_break();d.add_heading('8 Overnight image level campaign',level=1)
    d.add_paragraph('Status snapshot '+data['as_of_display']+'. This campaign uses the provisional exact-file-controlled manifest, with 23,761 training and 3,059 validation images. The 3,090 test images remain outside fitting, tuning and evaluation. Patient and slide independence remain unverified.')
    d.add_paragraph('The four arms are standard augmentation, residual-preserving fixed augmentation, uniformly weaker augmentation and the full adaptive controller. Each seed uses identical initial weights, split, geometry stream, batch order and optimizer settings. Training uses AdamW, a maximum of 40 epochs, early stopping after seven epochs without a macro-F1 improvement exceeding 0.0001, and the same validation-loss scheduler. Three epochs are not treated as convergence.')
    t=d.add_table(rows=1,cols=5)
    for c,v in zip(t.rows[0].cells,['Arm','State','Epoch','Accuracy','Macro F1']):c.text=v
    for row in data['first_seed']:
        cells=t.add_row().cells
        values=[LABELS[row['arm']],row['status'],str(row['epoch']),
                '-' if row['accuracy'] is None else f"{row['accuracy']:.4f}",
                '-' if row['macro_f1'] is None else f"{row['macro_f1']:.4f}"]
        for c,v in zip(cells,values):c.text=v
    width=(d.sections[0].page_width-d.sections[0].left_margin-d.sections[0].right_margin)/914400
    table_style(t,[width*.30,width*.18,width*.12,width*.20,width*.20])
    d.add_paragraph('Table 3. Seed 42 best-so-far validation results. Values for active runs are provisional and may change. A completed run must reload its saved best checkpoint before its final validation report. This table is separate from the earlier 512/128-image, three-epoch pilot.')
    d.add_paragraph(f"The selected cached pipeline benchmark achieved {data['throughput']:.1f} training images per second with batch 64 and four loader workers. Batch 96 was rejected for insufficient VRAM headroom. Warmup, worker startup and full-epoch timings are recorded separately. The benchmark changes several settings together, so its throughput difference is not an isolated estimate of cache benefit.")
    d.add_paragraph('The deterministic cache stores float64 stain matrices, concentrations and diagnostics. It passed 64 exact pixel-equivalence checks against uncached controller output. Two ADI extraction failures remain as unchanged-image fallbacks. Stochastic target stains, intensity factors, color adjustments and geometry are not cached.')
    d.add_page_break();d.add_heading('9 Class specific results and operational status',level=1)
    d.add_paragraph('The table below reports recall for each class at each arm\'s selected checkpoint or current accepted best validation epoch. A dash denotes an arm without validation results; it is not a measured zero.')
    recall=d.add_table(rows=1,cols=5)
    for c,v in zip(recall.rows[0].cells,['Class','Standard','Fixed','Weaker','Adaptive']):c.text=v
    for code in CLASSES:
        cells=recall.add_row().cells;cells[0].text=code
        for i,row in enumerate(data['first_seed'],start=1):
            value=row['per_class'].get(code,{}).get('recall')
            cells[i].text='-' if value is None else f'{value:.3f}'
    table_style(recall,[width*.16]+[width*.21]*4)
    d.add_paragraph(f"Completed overnight runs: {data['completed_runs']} of a bounded maximum of 12. Remaining replications depend on the campaign time budget. Extraction-fallback counts are logged by class, and validation predictions retain all eight probabilities. Calibration summaries, confusion matrices and learning histories are stored with each completed run.")
    d.add_paragraph('A separate supervisor writes current status and the 09:00 Asia/Calcutta snapshot while training continues. Resumable checkpoints contain model and optimizer states, the learning-rate scheduler, mixed-precision scaler, random states and the completed position within the epoch. The original application, SQLite database, data, model pointer and checkpoints remain protected.')
    d.add_paragraph('The prescribed institutional report format and complete author details were not supplied. This document uses a provisional report format and the verified short author name Amlan. The Overleaf session required sign-in, so no private online project or real project link has been created. The IEEE conference ZIP remains ready for upload.')
    d.add_paragraph('No observed validation difference establishes clinical benefit, patient-independent generalization or research novelty. The historical notebook figures are separately sourced and must not be substituted for this campaign\'s results. Similarity and AI-detection scores require the institution\'s own checks and are not promised.')
    target=build/'Progress_Report.docx';d.save(target)
    return target


def latex(data,package,original):
    source=(original/'overleaf/main.tex').read_text(encoding='utf-8')
    rows=[]
    for r in data['first_seed']:
        accuracy='--' if r['accuracy'] is None else f"{100*r['accuracy']:.2f}"
        macro='--' if r['macro_f1'] is None else f"{r['macro_f1']:.4f}"
        rows.append(f"{LABELS[r['arm']]} & {r['epoch']} & {accuracy} & {macro}\\\\")
    section=r'''
\section{Extended Image-Level Campaign}
The subsequent campaign uses the provisional exact-file-controlled manifest with
23,761 training and 3,059 validation images. The 3,090 test images remain outside
tuning and inference. Patient linkage is unavailable, so these image-level
validation results cannot establish patient-independent generalization.

Standard augmentation, residual-preserving fixed augmentation, uniformly weaker
augmentation and the full controller share initialization within each seed,
geometry streams, ordering, optimizer, a 40-epoch ceiling and early stopping
with patience seven on validation macro-F1. A validation-loss scheduler halves
the learning rate after two non-improving epochs. Training-only cached diagnostics
fit the controller and stain parameters. Three epochs are not a convergence claim.

Float64 cached decompositions reproduced uncached controller output exactly in
64 checks. The cache covers training and validation only. Two ADI extractions
failed and retain unchanged-image fallback. Targets, intensities and geometry
remain stochastic. Main CPU threads are limited to two and each loader worker
to one. A training-only warm-cache benchmark selected batch 64 and four workers,
achieving THROUGHPUT images/s, while batch 96 failed the VRAM-headroom rule.
This joint configuration comparison does not isolate cache speedup.

\begin{table}[t]\caption{Extended campaign snapshot at SNAPSHOT. Seed 42 values
are best-so-far validation selections. Active runs are incomplete. Dashes denote
unavailable measurements.}\label{tab:overnight}\centering
\begin{tabular}{lrrr}\toprule
Arm & Epoch & Accuracy (\%) & Macro-F1\\\midrule
ROWS
\bottomrule\end{tabular}\end{table}

Completed-run metrics reload the saved best checkpoint. Frequent resumable
checkpoints include optimizer, scheduler, scaler, random states and mid-epoch
position. No research checkpoint is activated in the application. The 09:00
reporting snapshot does not stop the bounded campaign. Novelty and clinical
benefit remain unconfirmed, and any preliminary between-arm differences require
replication and a verified grouped or external evaluation design.

'''.replace('THROUGHPUT',f"{data['throughput']:.1f}").replace('SNAPSHOT',data['as_of_display']).replace('ROWS','\n'.join(rows))
    source=source.replace(r'\section{Discussion and Limitations}',section+r'\section{Discussion and Limitations}')
    source=source.replace('This later manifest was not used for the classifier pilot below.','This later manifest was not used for the small classifier pilot below; it is used for the separate extended image-level campaign.')
    # Distinguish the update from the original pilot without inventing a paper outcome.
    source=source.replace('Improved classification was therefore not established.','Improved classification was therefore not established. A subsequent bounded campaign on a provisional exact-file-controlled image split is reported separately as an active progress snapshot.')
    (package/'overleaf/main.tex').write_text(source,encoding='utf-8')
    instructions='''# IEEE source and Overleaf status

This folder contains the updated IEEE conference paper source and IEEEtran.cls.
The class file and its license notices are unchanged from the transferred package.

Overleaf status: no authenticated session was available. The inspected project URL
redirected to the login page. No private project or real project link was created.

To import: sign in to Overleaf, choose New Project, Upload Project, and select
Overleaf_Paper_Source.zip. Select main.tex as the main file and pdfLaTeX as the
compiler. Keep the project private. Recompile and review the output before sharing
an actual project URL. No URL or completed online import is claimed here.

The supplied author name is Amlan. Add a verified full name, affiliation and email
only when provided. The paper is an exploratory progress draft, not an accepted
publication. Numerical preprocessing stability, classifier validation and notebook
metrics are distinct evidence sources. No Turnitin or AI-score guarantee is made.
'''
    (package/'overleaf/README.md').write_text(instructions,encoding='utf-8')
    (package/'OVERLEAF_STATUS.md').write_text(instructions,encoding='utf-8')
    temporary=package/'Overleaf_Paper_Source.tmp.zip'
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ['main.tex','IEEEtran.cls','README.md']:
            z.write(package/'overleaf'/name,name)
    os.replace(temporary,package/'Overleaf_Paper_Source.zip')


def supporting(data,package,original):
    text=(original/'Presentation_Script_and_Viva.md').read_text(encoding='utf-8')
    text=text.replace('## Slide 13: Demonstration and conclusion','## Slide 15: Demonstration and conclusion')
    insert='''## Slide 13: Overnight image-level validation

This is a separate, larger campaign using 23,761 training and 3,059 validation
images from the provisional exact-file-controlled split. Read the snapshot time
and each arm's state. Active runs show best-so-far validation values. Patient
independence is still unverified, and the final test set has not been evaluated.

## Slide 14: Measured throughput and restartable training

The selected real pipeline benchmark uses batch 64 and four workers. A larger
batch was slightly faster but failed the VRAM-headroom requirement. Deterministic
decompositions are cached, while random augmentation remains random. Checkpoints
save every two minutes and each epoch. The 9 AM snapshot does not stop training.

'''
    text=text.replace('## Slide 15: Demonstration and conclusion',insert+'## Slide 15: Demonstration and conclusion')
    text+='''
## Updated overnight viva answers

Q: Why train without patient mapping now?
A: The current task explicitly permits an exploratory image-level comparison.
Exact-file deduplication reduces one known leak, but the results cannot establish
patient independence or clinical generalization. Verified mapping remains needed.

Q: Does caching freeze augmentation?
A: No. Only deterministic source stain decompositions and diagnostics are cached.
Each image and epoch still samples target stains, intensity and geometry from
separate reproducible random streams.

Q: Why reject the larger batch?
A: Batch 96 left too little free VRAM in the actual mixed-precision benchmark.
Batch 64 retained headroom for normal display and application use.

Q: What survives a Codex interruption?
A: The independent supervisor, training logs, best checkpoints and full resumable
last checkpoints remain local. The checkpoint includes the completed epoch
position, optimizer, scheduler, scaler and random states.

Q: Has Overleaf import been completed?
A: No. An authenticated session was unavailable. The ZIP is ready for a private
upload, but an actual project link still requires sign-in and verified import.

Q: Is this the teacher's exact report format?
A: No prescribed template was provided. The editable report is explicitly
provisional and must be transferred into the prescribed template when supplied.

'''
    text=text.replace('.\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000',
                      '.\\.venv-research\\Scripts\\python.exe stain_research/demo_serve.py')
    (package/'Presentation_Script_and_Viva.md').write_text(text,encoding='utf-8')
    checklist='''# Solo demo checklist

Progress evaluation: 13 November 2026, 11:00 AM, B-305, Asia/Calcutta.
Final evaluation: 19 November 2026, 11:00 AM; confirm the final venue separately.

1. Read MORNING_STATUS.md and check which runs are complete versus active.
2. Open the PPTX and PDF backup. Read the actual snapshot time on the new table.
3. Rehearse the 10-minute solo script. Keep pilot, overnight and notebook metrics separate.
4. Use `.venv-research/Scripts/python.exe stain_research/demo_serve.py` for an
   isolated demonstration copy of the database and upload folder. The original
   database and active checkpoint remain unchanged.
5. Open http://127.0.0.1:8000. Show the model source before interpreting predictions.
6. Upload a public TRAIN example, not a held-out test image. Show eight probabilities
   and the stored history in the isolated demo database. Never call the weighted
   dashboard percentage a calibrated clinical risk or measured tumor area.
7. Explain the 16 spatial subpatches, CNN, two-layer BiLSTM and additive attention.
8. Show the controller formula and the mandatory residual-fixed and weaker controls.
9. State the duplicate audit and unverified patient mapping. Do not claim clinical
   validity, established novelty, publication acceptance or a guaranteed similarity score.
10. If the live demonstration fails, use the saved implementation and PDF honestly.
    Leave the original application and model activation intact. Bring the provisional
    report and apply the prescribed institutional template when it becomes available.

The training supervisor continues after 9 AM. Stop a run only through a deliberate,
checkpoint-aware action; the reporting milestone itself is not a stop command.
'''
    (package/'DEMO_CHECKLIST.md').write_text(checklist,encoding='utf-8')
    start=f'''# Progress evaluation package

Progress evaluation: 13 November 2026, 11:00 AM, B-305, Asia/Calcutta.
Final evaluation: 19 November 2026, 11:00 AM; final venue unverified.
Evidence snapshot: {data['as_of_display']}. Read evaluation_snapshot.json for
the exact completed and active runs represented in these documents. The live
MORNING_STATUS.md is in the project root; the supervisor saves a separate morning
snapshot by 09:00 and continues the bounded campaign afterward.

- Progress_Presentation.pptx: 15 editable slides with speaker notes.
- Progress_Presentation.pdf: offline slide backup.
- Progress_Report.docx and .pdf: provisional report; exact teacher template unavailable.
- Presentation_Script_and_Viva.md: solo script and research questions.
- DEMO_CHECKLIST.md: isolated application demonstration and rehearsal instructions.
- Overleaf_Paper_Source.zip and overleaf/: IEEE conference LaTeX source.
- OVERLEAF_STATUS.md: authentication and upload status; no invented project link.

The original three-epoch pilot and the larger overnight campaign are separate
experiments. Adaptive pilot macro-F1 0.2651 did not beat uniformly weaker 0.2802.
The larger campaign uses a provisional exact-file-duplicate-controlled split;
patient independence is unverified. Report best validation results as exploratory,
not final test performance. No test inference or dashboard research-model activation
has occurred. Classification superiority, novelty and clinical benefit remain unconfirmed.

The prescribed institutional template, full author identity, registration number,
department and guide details have not been supplied. Only verified author details
are used. Apply the real template and verified identity before claiming compliance.
No similarity or AI-detection score is promised.

To upload the paper: sign in to Overleaf, choose New Project > Upload Project,
select Overleaf_Paper_Source.zip, set main.tex as the main document, use pdfLaTeX,
and compile. A private authenticated project has not been created. Check
OVERLEAF_STATUS.md for the actual status. Preserve IEEEtran's license notice.

Practice the slide notes and solo script, bring both PDFs offline, and rehearse
the isolated demo before 11 AM. The dashboard remains on its original model path;
prototype fallback predictions must be identified honestly. Do not activate the
research checkpoints. Local Python automation refreshes these materials without
ChatGPT calls; numerical updates still require presenter review.
'''
    (package/'START_HERE.md').write_text(start,encoding='utf-8')
    identity=read(ROOT/'stain_research/report_identity.json',{})
    if identity:
        start=start.replace('provisional report; exact teacher template unavailable.','report follows the supplied KIIT template; certificate unsigned.')
        start=start.replace('The prescribed institutional template, full author identity, registration number,\ndepartment and guide details have not been supplied. Only verified author details\nare used. Apply the real template and verified identity before claiming compliance.',
                            'The supplied KIIT template is used for the report. Student: '+identity['name']+', roll '+identity['roll_number']+'. Guide: '+identity['guide']+'. The certificate is unsigned and a genuine plagiarism report remains pending.')
        (package/'START_HERE.md').write_text(start,encoding='utf-8')


def run(command,log,timeout=180):
    with log.open('a',encoding='utf-8') as stream:
        result=subprocess.run([str(v) for v in command],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                              timeout=timeout,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if result.returncode:raise RuntimeError('Artifact command failed; see '+str(log))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    # All document authoring runs in the authoritative bundled runtime.
    if not args.worker:
        result=subprocess.run([str(RUNTIME/'python/python.exe'),str(Path(__file__)),'--campaign',str(args.campaign.resolve()),'--worker'],cwd=ROOT)
        raise SystemExit(result.returncode)
    campaign=args.campaign.resolve();package=ROOT/'evaluation_package';build=ROOT/'.sync-backups/evaluation_build'
    original=ROOT/'.sync-backups/evaluation_package_original_20261113'
    lock=package/'.refresh.lock'
    try: descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:print('Another package refresh is active; leaving existing deliverables intact.');return
    os.close(descriptor)
    try:
        data=snapshot(campaign);save(build/'evaluation_snapshot.json',data)
        document_path=document(data,build)
        run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',ROOT/'stain_research/office_export.ps1',
             '-InputDocx',document_path,'-OutputPdf',build/'Progress_Report.pdf',
             '-PaginationJson',build/'report_pagination.json'],build/'word_export.log',120)
        if (ROOT/'evaluation_package/templates/Project_Report_Format.docx').exists():
            from build_teacher_report import cache_pagination
            cache_pagination(document_path,build/'report_pagination.json')
        # PDF raster images provide actual visual QA. Native Word is used because
        # the packaged LibreOffice renderer reported soffice.exe unavailable.
        poppler=RUNTIME/'native/poppler/Library/bin/pdftoppm.exe'
        qa=package/'.qa';qa.mkdir(exist_ok=True)
        run([poppler,'-r','110','-png',build/'Progress_Report.pdf',qa/'report'],build/'pdf_render.log',120)
        run([RUNTIME/'node/bin/node.exe',ROOT/'stain_research/build_evaluation_deck.mjs'],build/'deck_build.log',180)
        run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',ROOT/'stain_research/office_export.ps1',
             '-InputPptx',build/'Progress_Presentation.pptx','-OutputPdf',build/'Progress_Presentation.pdf'],build/'powerpoint_export.log',120)
        stage=build/'package_stage';stage.mkdir(exist_ok=True);(stage/'overleaf').mkdir(exist_ok=True)
        shutil.copy2(original/'overleaf/IEEEtran.cls',stage/'overleaf/IEEEtran.cls')
        for name in ['Progress_Report.docx','Progress_Report.pdf','Progress_Presentation.pptx','Progress_Presentation.pdf']:
            shutil.copy2(build/name,stage/name)
        save(stage/'evaluation_snapshot.json',data)
        latex(data,stage,original);supporting(data,stage,original)
        if data.get('final_analysis'):
            from final_materials import finish_package
            finish_package(data,stage)
        compiler=build/'tectonic/tectonic.exe'
        if compiler.exists():
            compiled=build/'latex_checked';compiled.mkdir(exist_ok=True)
            run([compiler,'--keep-logs','--outdir',compiled,stage/'overleaf/main.tex'],build/'latex_build.log',180)
            shutil.copy2(compiled/'main.pdf',stage/'IEEE_Paper_Draft.pdf')
            save(stage/'IEEE_LOCAL_BUILD.json',{'compiler':'Tectonic 0.17.0',
                 'source_sha256':hashlib.sha256((stage/'overleaf/main.tex').read_bytes()).hexdigest(),
                 'local_compile_passed':True,'overleaf_import_complete':False})
        from pypdf import PdfReader
        page_counts={'report':len(PdfReader(stage/'Progress_Report.pdf').pages),
                     'presentation':len(PdfReader(stage/'Progress_Presentation.pdf').pages)}
        if page_counts['presentation']!=15:raise RuntimeError('Unexpected presentation page count')
        # Guard the negative findings and the new experimental limitations.
        assert '0.2651' in (stage/'overleaf/main.tex').read_text(encoding='utf-8')
        assert 'Patient' in (stage/'overleaf/main.tex').read_text(encoding='utf-8')
        manifest={p.relative_to(stage).as_posix():{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                  for p in stage.rglob('*') if p.is_file() and not any(x.startswith('.') for x in p.relative_to(stage).parts)
                  and p.name not in ['FILE_MANIFEST.json','PACKAGE_READY.json']}
        save(stage/'FILE_MANIFEST.json',manifest)
        save(stage/'last_refresh.json',{'at':data['as_of'],'page_counts':page_counts,'automated_checks_passed':True,
                                         'word_renderer':'Microsoft Word PDF export; Poppler rasterization',
                                         'presentation_renderer':'Artifact Tool import/edit/export with finalizer',
                                         'human_visual_review':'Initial reviewed template; refreshed numeric fields need normal presenter review',
                                         'overleaf_import_complete':False})
        for source in stage.rglob('*'):
            if source.is_file():
                target=package/source.relative_to(stage);target.parent.mkdir(parents=True,exist_ok=True)
                temp=target.with_name(target.name+'.refresh-tmp');shutil.copy2(source,temp);os.replace(temp,target)
        print(json.dumps({'as_of':data['as_of'],'page_counts':page_counts,'completed_runs':data['completed_runs']}))
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
