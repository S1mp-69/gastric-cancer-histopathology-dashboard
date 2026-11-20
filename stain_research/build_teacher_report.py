"""Build the project report from the retained teacher template and verified evidence.

No training, inference, database writes or model activation. Unedited template
package parts are retained byte-for-byte; source placeholders are replaced.
"""
from copy import deepcopy
import hashlib
import io
import json
import subprocess
from pathlib import Path
import zipfile
import re

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
TITLE = 'Gastric Histopathology Tissue Classification'
SHORT = 'Gastric Histopathology Tissue Classification'
LABELS = {'standard':'Standard', 'residual_fixed':'Residual fixed',
          'uniform_weaker':'Uniform weaker', 'adaptive':'Adaptive'}


def field(paragraph, instruction, cached='Update field'):
    r=paragraph.add_run(); begin=OxmlElement('w:fldChar');begin.set(qn('w:fldCharType'),'begin');r._r.append(begin)
    r=paragraph.add_run(); code=OxmlElement('w:instrText');code.set(qn('xml:space'),'preserve');code.text=' '+instruction+' ';r._r.append(code)
    r=paragraph.add_run();sep=OxmlElement('w:fldChar');sep.set(qn('w:fldCharType'),'separate');r._r.append(sep)
    paragraph.add_run(cached)
    r=paragraph.add_run();end=OxmlElement('w:fldChar');end.set(qn('w:fldCharType'),'end');r._r.append(end)


def bookmark(p, name, number):
    a=OxmlElement('w:bookmarkStart');a.set(qn('w:id'),str(number));a.set(qn('w:name'),name)
    b=OxmlElement('w:bookmarkEnd');b.set(qn('w:id'),str(number));p._p.insert(0,a);p._p.append(b)


def cache_pagination(docx_path, pagination_path):
    """Cache Word's actual page references without rewriting template parts."""
    from lxml import etree
    pages=json.loads(Path(pagination_path).read_text(encoding='utf-8-sig'))
    path=Path(docx_path)
    with zipfile.ZipFile(path) as z:parts={n:z.read(n) for n in z.namelist()}
    root=etree.fromstring(parts['word/document.xml']);ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    for p in root.findall('.//w:p',ns):
        for instruction in p.findall('.//w:instrText',ns):
            match=re.search(r'PAGEREF\s+(\w+)',instruction.text or '')
            if match and match.group(1) in pages:
                run=instruction.getparent();following=run.getnext();separated=False
                while following is not None:
                    marker=following.find('w:fldChar',ns)
                    if marker is not None:
                        value=marker.get(qn('w:fldCharType'))
                        if value=='end':break
                        if value=='separate':separated=True
                    if separated:
                        text=following.find('w:t',ns)
                        if text is not None:text.text=str(pages[match.group(1)]);break
                    following=following.getnext()
    parts['word/document.xml']=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
    tmp=path.with_suffix('.cached.docx')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for name,content in parts.items():z.writestr(name,content)
    tmp.replace(path)


def font(p, size=12, bold=False):
    for r in p.runs:
        r.font.name='Times New Roman';r.font.size=Pt(size);r.bold=bold;r.font.color.rgb=RGBColor(0,0,0)


def build(data, build_dir, template=None):
    build_dir=Path(build_dir);build_dir.mkdir(parents=True,exist_ok=True)
    (build_dir/'report_snapshot.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    template=Path(template or ROOT/'evaluation_package/templates/Project_Report_Format.docx')
    identity=json.loads((ROOT/'stain_research/report_identity.json').read_text())
    digest=hashlib.sha256(template.read_bytes()).hexdigest()
    if digest!=identity['template_sha256']:raise RuntimeError('Retained template hash changed')
    source_zip=zipfile.ZipFile(template);d=Document(template)
    final=data.get('final_analysis')
    figure_data=build_dir/'figure_data.json';figure_data.write_text(json.dumps(data['first_seed']))
    subprocess.run([str(ROOT/'.venv-research/Scripts/python.exe'),str(ROOT/'stain_research/render_report_figures.py'),
                    str(figure_data),str(build_dir)],check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    # The copied template supplies styles, fonts, relationships and logo assets.
    for element in list(d._element.body):d._element.body.remove(element)
    first=deepcopy(Document(template).sections[0]._sectPr)
    for tag in ['headerReference','footerReference','pgNumType','type']:
        for item in first.findall(qn('w:'+tag)):first.remove(item)
    d._element.body.append(first)
    markers=[];figures=[];tables=[];number=100;pending_page=False

    def para(text='',size=12,bold=False,align=None,before=0,after=7,keep=False):
        nonlocal pending_page
        p=d.add_paragraph(text);p.paragraph_format.space_before=Pt(before);p.paragraph_format.space_after=Pt(after)
        if pending_page:p.paragraph_format.page_break_before=True;pending_page=False
        p.paragraph_format.line_spacing=1.15;p.paragraph_format.keep_with_next=keep
        p.paragraph_format.widow_control=True
        p.alignment=align if align is not None else WD_ALIGN_PARAGRAPH.JUSTIFY
        font(p,size,bold);return p

    def center(text,size=13.5,bold=False,before=0,after=10):
        return para(text,size,bold,WD_ALIGN_PARAGRAPH.CENTER,before,after)

    def heading(text,level=1,mark=None):
        nonlocal number
        p=para(text,23.5 if level==1 else 14,False,WD_ALIGN_PARAGRAPH.LEFT,12,12,True)
        outline=OxmlElement('w:outlineLvl');outline.set(qn('w:val'),str(level-1));p._p.get_or_add_pPr().append(outline)
        if mark:
            bookmark(p,mark,number);number+=1;markers.append((text,mark))
        return p

    def page():
        nonlocal pending_page
        pending_page=True

    def caption(text,kind,mark):
        nonlocal number
        p=para(text,10.5,False,WD_ALIGN_PARAGRAPH.CENTER,4,10)
        bookmark(p,mark,number);number+=1
        (figures if kind=='figure' else tables).append((text,mark))
        return p

    def table(headers,rows,widths=None):
        t=d.add_table(rows=1,cols=len(headers));t.autofit=False
        for c,s in zip(t.rows[0].cells,headers):c.text=str(s)
        for row in rows:
            for c,s in zip(t.add_row().cells,row):c.text=str(s)
        widths=widths or [6.3/len(headers)]*len(headers)
        for row_i,row in enumerate(t.rows):
            prop=row._tr.get_or_add_trPr();cant=OxmlElement('w:cantSplit');prop.append(cant)
            for i,c in enumerate(row.cells):
                c.width=Inches(widths[i]);m=OxmlElement('w:tcMar')
                for edge in ['top','left','bottom','right']:
                    v=OxmlElement('w:'+edge);v.set(qn('w:w'),'80');v.set(qn('w:type'),'dxa');m.append(v)
                c._tc.get_or_add_tcPr().append(m)
                for p in c.paragraphs:
                    p.paragraph_format.space_after=Pt(3);p.paragraph_format.line_spacing=1.05
                    font(p,11,row_i==0)
        borders=OxmlElement('w:tblBorders')
        for edge in ['top','left','bottom','right','insideH','insideV']:
            v=OxmlElement('w:'+edge);v.set(qn('w:val'),'single');v.set(qn('w:sz'),'4');v.set(qn('w:color'),'000000');borders.append(v)
        t._tbl.tblPr.append(borders);repeat=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(repeat)
        para('',after=2);return t

    def geometry(s,front=False):
        s.page_width=11900*635;s.page_height=16838*635
        s.left_margin=1440*635 if front else 1400*635;s.right_margin=1440*635 if front else 1386*635
        s.top_margin=1292*635 if front else 1440*635;s.bottom_margin=807*635 if front else 1440*635
        s.header_distance=Inches(.5);s.footer_distance=Inches(.5)
        cols=s._sectPr.find(qn('w:cols'))
        if cols is not None:cols.attrib.clear();cols.set(qn('w:space'),'720')
        borders=OxmlElement('w:pgBorders');borders.set(qn('w:offsetFrom'),'page')
        for edge in ['top','left','bottom','right']:
            e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'6');e.set(qn('w:space'),'24');e.set(qn('w:color'),'000000');borders.append(e)
        old=s._sectPr.find(qn('w:pgBorders'))
        if old is not None:s._sectPr.remove(old)
        s._sectPr.append(borders)

    geometry(d.sections[0],True)
    for cover in range(2):
        if cover:page()
        center('A PROJECT REPORT',16.5,True,10,12);center('on',13.5,True)
        p=center(TITLE,19.5,True,after=14)
        # Title paragraph role remains directly formatted from the reference.
        center('Submitted to',16.5,True);center('KIIT Deemed to be University',19.5,True)
        center('In Partial Fulfilment of the Requirement for the Award of',13.5,False,after=12)
        center('BACHELOR OF TECHNOLOGY IN',16.5,True)
        center('COMPUTER SCIENCE AND ENGINEERING',16.5,True)
        center('BY',16.5,True,before=5)
        center(identity['name'].upper(),14.5,True,after=3);center('Roll Number '+identity['roll_number'],14.5,after=12)
        center('UNDER THE GUIDANCE OF',13.5,True);center(identity['guide'],13.5,True)
        p=center('',after=8);p.add_run().add_picture(io.BytesIO(source_zip.read('word/media/image2.jpeg')),width=Inches(.8))
        center('SCHOOL OF COMPUTER SCIENCE AND ENGINEERING',12.5,True,after=5)
        center('KALINGA INSTITUTE OF INDUSTRIAL TECHNOLOGY',15.5,True,after=5)
        center('BHUBANESWAR, ODISHA 751024',12.5,True,after=6)
        center('November 2026',12.5,True,after=3)
        center('Programme year as supplied: '+identity['year_as_supplied'],11,after=3)
        center('Progress evaluation report',11,after=0)

    page();center('KIIT Deemed to be University',19.5)
    center('School of Computer Science and Engineering',13.5);center('Bhubaneswar, Odisha 751024',13.5)
    p=center('');p.add_run().add_picture(io.BytesIO(source_zip.read('word/media/image2.jpeg')),width=Inches(.8))
    center('CERTIFICATE',23.5,before=12)
    center('DRAFT FOR GUIDE REVIEW — UNSIGNED',12,True)
    para('The certificate page is retained in the prescribed report format. Certification of the work, academic-year range and signatures must be confirmed by the project guide; this draft does not assert that certification has occurred.')
    center(TITLE,16.5);center('Submitted by',13.5)
    center(identity['name']+' — Roll Number '+identity['roll_number'],14.5,True)
    para('Programme: '+identity['programme']+'. Project guide: '+identity['guide']+'. Year supplied by the student: '+identity['year_as_supplied']+'. Report prepared for the progress evaluation on 13 November 2026.')
    para('Date of certification: ____________________',before=20)
    para('Signature of project guide: ____________________',before=30)
    para(identity['guide']+'\nProject Guide',align=WD_ALIGN_PARAGRAPH.RIGHT)

    page();center('Acknowledgements',19.5,True,before=18,after=24)
    para('I acknowledge '+identity['guide']+' as my project guide and the School of Computer Science and Engineering, KIIT Deemed to be University, as the academic setting for this project. I thank the authors who released the gastric histopathology dataset and the stain augmentation research used as the basis of the implementation.')
    para('The report distinguishes my project implementation, externally sourced algorithms and datasets, and experiments assisted by software tools. Stain SAN-derived extraction is attributed in the source notices. I remain responsible for reviewing the experimental claims, references and final submission, including any disclosure required by the institution.')
    para('Student signature: ____________________\n'+identity['name']+'\nRoll Number '+identity['roll_number'],align=WD_ALIGN_PARAGRAPH.RIGHT,before=24)

    page();center('ABSTRACT',19.5,True,before=18,after=24)
    para('This project investigates eight-class tissue classification in gastric histopathology and whether stain-decomposition diagnostics can guide augmentation strength. A CNN, bidirectional LSTM and attention classifier is implemented alongside a FastAPI dashboard with SQLite prediction storage. A residual-preserving adaptive stain controller was developed and compared with standard augmentation, fixed stain augmentation and uniformly weaker augmentation.')
    para('An exact-file audit found duplicate leakage in the historical image splits. A provisional manifest retains 29,910 unique files; patient and slide independence remain unverified. In the completed seed-42 comparison on 23,761 training and 3,059 validation images, adaptive augmentation reached accuracy 69.01% and macro-F1 0.6888, versus 68.65% and 0.6867 for standard augmentation. The small lead is unreplicated and accompanied by lower tumor recall. The earlier three-epoch pilot did not establish adaptive benefit. Preprocessing numerical stability, classifier validation and historical notebook metrics are reported separately.')
    para('A CUDA pipeline with deterministic caches, stochastic augmentation, mixed precision, early stopping and resumable checkpoints supports the bounded campaign. Replication continues. Final test images remain outside tuning and inference, and research checkpoints are not activated in the dashboard. Classification superiority, patient-level generalization, research novelty and clinical benefit are not established.')
    para('Keywords: gastric histopathology; tissue classification; stain augmentation; CNN BiLSTM attention; duplicate audit; adaptive preprocessing.',bold=True)
    para('Evidence snapshot: '+data['as_of_display']+'. Completed runs: '+str(data['completed_runs'])+' of a maximum of 12. Active values in later tables are provisional.',size=11)

    page();heading('Contents');contents_anchor=d.paragraphs[-1]
    page();heading('List of Figures');fig_anchor=d.paragraphs[-1]
    page();heading('List of Tables');table_anchor=d.paragraphs[-1]

    s=d.add_section(WD_SECTION_START.NEW_PAGE);geometry(s)
    s.header.is_linked_to_previous=False;s.footer.is_linked_to_previous=False
    h=s.header.paragraphs[0];h.text=SHORT;h.alignment=WD_ALIGN_PARAGRAPH.RIGHT;font(h,10)
    f=s.footer.paragraphs[0];f.text='School of Computer Science and Engineering, KIIT, BBSR\t';font(f,9)
    f.paragraph_format.tab_stops.add_tab_stop(Inches(6.2),2);field(f,'PAGE','1');font(f,9)
    pn=OxmlElement('w:pgNumType');pn.set(qn('w:start'),'1');s._sectPr.append(pn)

    def chapter(n,title,mark):
        if n>1:page()
        para('Chapter '+str(n),23.5,False,WD_ALIGN_PARAGRAPH.LEFT,8,10,True)
        heading(title,1,mark)

    chapter(1,'Introduction','chapter1')
    para('Histopathology images contain tumor epithelium, normal glands, stroma and other tissue components whose appearance depends on both morphology and hematoxylin and eosin staining. Variation in stain concentration, preparation and imaging can alter image appearance. An augmentation method must preserve diagnostically useful structure while producing training variation; better numerical reconstruction alone does not establish better classification.')
    para('This project combines an implemented tissue-classification dashboard with an isolated research study of decomposition-aware stain augmentation. The classifier predicts ADI, DEB, LYM, MUC, MUS, NOR, STR and TUM. The research question is whether reconstruction and conditioning diagnostics can guide perturbation strength more effectively than fixed or uniformly weaker augmentation under matched classifier conditions.')
    para('The current result is mixed. The full seed-42 adaptive run has a small macro-F1 lead over standard augmentation, but reduced tumor recall and no completed matched replication. The previous short pilot was negative relative to weaker and shuffled controls. The project therefore contributes an implemented mechanism, reproducible comparisons and a transparent audit; it does not yet establish a novel or clinically useful classifier improvement.')
    para('The following chapters describe relevant concepts and prior work, requirements and system design, implementation and verification, adopted development practices and resource costs, and conclusions with remaining work. The research pipeline protects the original application, dataset, database and checkpoints, while reserving final test data for a later locked evaluation.')

    chapter(2,'Basic Concepts and Literature Review','chapter2')
    heading('2.1 Dataset and tissue classes',2,'sec21')
    para('The HMU-GC-HE-30K release described by Lou et al. [1] contains 31,096 PNG patches, with 3,887 images in each of eight classes. Patches are 224 by 224 pixels. The eight labels represent adipose (ADI), debris (DEB), lymphocytes (LYM), mucus (MUC), muscle (MUS), normal gastric mucosa (NOR), stroma (STR) and tumor epithelium (TUM). Separate clinical tables support dashboard analytics but have no verified image-to-patient mapping in this local release.')
    heading('2.2 Stain decomposition and augmentation',2,'sec22')
    para('Stain decomposition represents optical-density image values using stain vectors and concentration coefficients. The research implementation measures reconstruction error and matrix conditioning. Residual-preserving augmentation reconstructs the stain-modified component while retaining the original reconstruction residual. Failed extraction returns the unchanged image rather than silently removing an observation.')
    para('Stain SAN [2] provides training-distribution-based stain sampling. Marini et al. [3] study data-driven color augmentation and motivate checking whether color perturbations remain admissible. These studies show why augmentation must be validated on downstream classification. The controller in this project uses training-fitted diagnostic ranges to reduce augmentation strength for higher-risk decompositions. Its novelty requires a fuller comparison with existing adaptive and quality-aware methods.')
    heading('2.3 CNN BiLSTM and attention',2,'sec23')
    para('The implemented architecture first encodes sixteen spatial 56 by 56 patches extracted from each resized image. Three convolutional stages use 32, 64 and 128 channels and project each patch to 128 features. A two-layer bidirectional LSTM uses 128 hidden units per direction. Additive attention weights the sixteen outputs, followed by a 128-unit classifier head, dropout 0.2 and eight logits. Spatial patch order is a constructed sequence and must not be described as patient time-series data.')
    heading('2.4 Validation and leakage',2,'sec24')
    para('Accuracy summarizes all correct predictions, while macro-F1 gives equal weight to each tissue class. Per-class recall identifies failures hidden by overall accuracy. Exact duplicate control prevents identical retained file bytes from appearing across splits, but does not demonstrate patient independence or eliminate near-duplicate morphology. Validation-based checkpoint selection also means the same validation result is not an untouched final estimate.')

    chapter(3,'Problem Statement and Requirement Specifications','chapter3')
    para('The problem is to evaluate whether diagnostic-guided stain perturbation improves an eight-class classifier under an auditable, matched experimental protocol, while preserving a working demonstration application. The requirements below describe this research prototype; no formal IEEE SRS conformity or clinical certification is asserted.')
    heading('3.1 Project Planning',2,'sec31')
    para('The work proceeds through provenance and duplicate auditing, deterministic preprocessing verification, a standard baseline, matched controller comparisons, replicated seeds where the time budget permits, and preparation of evaluation materials. The 09:00 status snapshot is a reporting milestone and does not stop training. Progress evaluation is scheduled for 13 November 2026 at 11:00 AM in B-305; final evaluation is listed for 19 November at 11:00 AM, with the final venue unverified.')
    heading('3.2 Project Analysis',2,'sec32')
    table(['Requirement','Implementation and acceptance condition'],[
        ['Image prediction','Upload a tissue patch; return eight probabilities and identify the active model source.'],
        ['Persistent history','Store a prediction record in isolated SQLite during rehearsal; leave the original database unchanged.'],
        ['Matched research arms','Same retained split, initial model per seed, geometry stream, batch order, optimizer and stopping rules.'],
        ['Data isolation','Train and validation only; prohibit test-image loading in the research dataset class.'],
        ['Restartability','Save last and best checkpoints; restore optimizer, scaler, scheduler, RNG and epoch position.']],[1.4,4.8])
    caption('Table 3.1 Functional research requirements','table','tab31')
    para('The original split had 24,876 training, 3,110 validation and 3,110 test files. Auditing found 932 duplicate groups containing 2,118 images; 334 groups spanned historical splits and 165 linked training and test. No cross-class exact duplicate label conflicts were found. The provisional manifest excludes 1,186 redundant copies, retaining 23,761 training, 3,059 validation and 3,090 test files. Retained assignments are not moved, and historical test copies are preferred over validation and training representatives.')
    heading('3.3 System Design',2,'sec33')
    heading('3.3.1 Design Constraints',2,'sec331')
    para('The PC has an NVIDIA RTX 3050 Laptop GPU with 4 GB VRAM and a Ryzen 5 5600H CPU. A fresh local Python 3.12 environment uses PyTorch 2.7.0 with CUDA 12.8. Batch size 64 and four loader workers were selected with mixed precision, two main CPU threads and one thread per loader worker. A larger batch of 96 was rejected because it left insufficient VRAM headroom. No system clock-rate or thermal-protection changes were made.')
    heading('3.3.2 System Architecture and Block Diagram',2,'sec332')
    para('The dashboard accepts an upload through FastAPI, preprocesses the image, invokes its configured predictor, returns probabilities and records the result through SQLAlchemy into SQLite. The isolated research runner imports the original classifier architecture without importing application database services. Cached train/validation images and diagnostics feed a stochastic augmentation layer, then the same classifier and validation routine for each arm. Research checkpoints remain outside the dashboard activation path.')
    p=para('',align=WD_ALIGN_PARAGRAPH.CENTER);p.add_run().add_picture(str(build_dir/'architecture.png'),width=Inches(6.15))
    caption('Figure 3.1 Application and isolated research architecture','figure','fig31')

    chapter(4,'Implementation','chapter4')
    heading('4.1 Methodology and Algorithms',2,'sec41')
    para('The deterministic cache verifies source image hashes and stores resized RGB, float64 stain bases, concentrations, diagnostics and validity flags using versioned keys. The key includes the manifest, numerical source versions and representation choices. It covers train and validation files only; fitted stain distributions and diagnostic scaling use training data. Cache lookup never fixes target stain sampling, intensity changes or geometric randomness.')
    para('Controller procedure: (1) decode and inspect the source decomposition; (2) normalize diagnostic features using training-fitted limits; (3) compute the controller strength from decomposition risk; (4) sample target stain and intensity using the current epoch and image random stream; (5) perturb the stain component and add the original residual; (6) clip to the image range and apply matched random geometry. Invalid decompositions follow an unchanged-image fallback. Standard augmentation uses its separate color-jitter stream; the research stain arms differ in their strength assignment.')
    para('The four main arms are standard augmentation, residual-preserving fixed augmentation, uniformly weaker augmentation and the full adaptive controller. Each seed shares the initial checkpoint and deterministic order/geometry streams across arms. AdamW uses learning rate 0.001, weight decay 0.0001 and label smoothing 0.05. Runs have a ceiling of 40 epochs and early stopping after seven epochs without a macro-F1 gain greater than 0.0001. ReduceLROnPlateau uses validation loss, patience two and factor 0.5. Gradient clipping and FP16 gradient scaling are enabled.')
    para('Last checkpoints are saved every 120 seconds and at epoch boundaries; best checkpoints are stored separately. Resume includes the model, optimizer, scheduler, gradient scaler, random states and completed image offset. Normal mixed-precision overflow skips the unsafe optimizer step and backs off the scale. Finite-loss checks remain strict. Transient Windows progress-file locks caused recovered restarts; a retry was added for subsequent workers. These are operational repairs, not evidence of classification improvement.')
    heading('4.2 Testing and Verification Plan',2,'sec42')
    table(['ID','Check','Observed evidence'],[
        ['T01','GPU execution','CUDA smoke and throughput checks passed; active status recorded GPU use.'],
        ['T02','Cached numerical equivalence','64 of 64 cached and uncached output comparisons were pixel-identical.'],
        ['T03','Model persistence','Completed runs reloaded best checkpoints and reproduced validation probabilities.'],
        ['T04','Application and SQL','Isolated upload returned eight probabilities and added one history row; original database hash unchanged.'],
        ['T05','Artifact preservation','Last audit checked 69 protected artifacts and 31,096 dataset files with no changes.'],
        ['T06','Split restrictions','Exact-file-controlled manifest and train/validation-only dataset guard; patient grouping unverified.']],[.45,1.6,4.15])
    caption('Table 4.1 Verification checks and observed outcomes','table','tab41')
    para('The preservation audit preceded completion of the overnight campaign. The demonstration used an isolated database and the prototype-baseline fallback, which must be identified honestly. It did not demonstrate the new research checkpoints. No research model activation or final test inference occurred.')
    heading('4.3 Result Analysis',2,'sec43')
    para('Preprocessing follow-up on 80 new validation images measured synthetic-noise amplification. At noise sigma 0.5, adaptive mean gain was 1.3049 versus 1.4249 for the matched uniform control; at sigma 2.0, values were 1.1646 versus 1.2532. Image-level descriptive intervals included no improvement. These numerical stability measurements do not establish classification robustness or accuracy.')
    table(['Arm','Accuracy','Macro-F1'],[
        ['Standard','33.59%','0.2438'],['Original SAN','32.81%','0.2177'],
        ['Residual fixed','30.47%','0.2323'],['Uniform weaker','33.59%','0.2802'],
        ['Shuffled','35.94%','0.2768'],['Adaptive','35.16%','0.2651']],[2.8,1.7,1.7])
    caption('Table 4.2 Historical three epoch classifier pilot','table','tab42')
    para('The historical pilot used 512 training and 128 validation images, one seed and three epochs. Adaptive macro-F1 did not beat weaker or shuffled controls, and TUM and NOR recall were zero for its selected checkpoint. This pilot is not convergence evidence and is separate from the larger provisional-split campaign. Historical notebook metrics (validation accuracy 0.5810; test accuracy 0.5682) are separately sourced legacy outputs and are not current campaign performance.')
    first=data['first_seed']
    table(['Arm','Epochs','State','Accuracy','Macro-F1'],[[LABELS[r['arm']],r['epoch'],r['status'],
          '-' if r['accuracy'] is None else f"{100*r['accuracy']:.2f}%",
          '-' if r['macro_f1'] is None else f"{r['macro_f1']:.4f}"] for r in first],[1.5,.6,1.1,1.5,1.5])
    caption('Table 4.3 Extended campaign seed 42 validation results','table','tab43')
    para('Snapshot: '+data['as_of_display']+'. Completed runs: '+str(data['completed_runs'])+' of a maximum of 12. Completed-run results are measured after reloading each selected best checkpoint; active rows are best-so-far values. The first completed four-arm seed gives adaptive accuracy 69.01% and macro-F1 0.6888, compared with standard 68.65% and 0.6867. The accuracy difference is 11 images out of 3,059; the macro-F1 difference is 0.0021. Neither is a replicated superiority result.')
    table(['Class','Standard','Fixed','Weaker','Adaptive'],[[c]+[
          '-' if r['per_class'].get(c,{}).get('recall') is None else f"{100*r['per_class'][c]['recall']:.2f}%" for r in first]
          for c in ['ADI','DEB','LYM','MUC','MUS','NOR','STR','TUM']],[.6,1.4,1.4,1.4,1.4])
    caption('Table 4.4 Per class recall in the seed 42 comparison','table','tab44')
    para('Adaptive TUM recall decreased from 79.27% to 74.87%, while ADI, MUS and STR recall improved. NOR recall remained below 50% in both standard and adaptive runs. Adaptive DEB recall also decreased. These tradeoffs prevent a claim of uniform benefit, even though overall macro-F1 is slightly higher. Complete paired seed results and variability are needed; unequal completed seed counts must not be pooled into a treatment-effect claim.')
    if final:
        heading('Completed three seed analysis',2)
        para('All 12 runs are complete. The preceding tables retain the historical pilot and seed-42 illustration. The following matched three-seed summaries are the campaign-level result. SD describes between-seed variation on the same validation images, not patient-level uncertainty.')
        table(['Arm','Accuracy mean ± SD','Macro F1 mean ± SD'],[[LABELS[r['arm']],f"{100*r['accuracy_mean']:.2f} ± {100*r['accuracy_sd']:.2f}%",f"{r['macro_f1_mean']:.4f} ± {r['macro_f1_sd']:.4f}"] for r in final['summary']],[1.8,2.2,2.2])
        caption('Table 4.5 Completed three seed validation summary','table','tab45')
        table(['Adaptive minus','Mean F1 difference','Seed differences'],[[LABELS[r['adaptive_minus']],f"{r['mean']:+.4f}",', '.join(f'{v:+.4f}' for v in r['macro_f1_differences'])] for r in final['paired']],[1.5,1.7,3])
        caption('Table 4.6 Paired differences for seeds 42 43 and 44','table','tab46')
        para('Adaptive mean macro-F1 is 0.6836 versus standard 0.6776, fixed 0.6822 and weaker 0.6804. It beats standard in all three seeds, fixed in only one and weaker in two. The fixed comparison changes sign across seeds. Exploratory 95% t intervals for paired F1 differences all include zero; with three seeds and distributional assumptions these are highly uncertain and cannot establish superiority. The full numerical analysis is retained with portable results.')
        table(['Class','Standard','Fixed','Weaker','Adaptive'],[[c]+[f"{100*r['recall_mean'][c]:.2f}%" for r in final['summary']] for c in ['ADI','DEB','LYM','MUC','MUS','NOR','STR','TUM']],[.6,1.4,1.4,1.4,1.4])
        caption('Table 4.7 Mean class recall across three seeds','table','tab47')
        para('Across all three seeds, adaptive TUM recall is 75.56% versus standard 79.97%, and DEB recall is 62.89% versus 65.72%. ADI, MUS, NOR and STR mean recall improves. Standard has lower mean calibration error (0.0157 versus adaptive 0.0213). Summed confusion matrices and learning curves are supplied as separate figures; the same validation cases appear in each seed and must not be treated as 9,177 independent images. Shuffled diagnostic assignments were not included in the full campaign, so the diagnostics-specific causal claim remains untested at this budget.')
    p=para('',align=WD_ALIGN_PARAGRAPH.CENTER);p.add_run().add_picture(str(build_dir/'validation_comparison.png'),width=Inches(6.1))
    caption('Figure 4.1 Validation macro F1 mean and between seed SD' if final else 'Figure 4.1 Seed 42 validation macro F1 without replicated uncertainty','figure','fig41')
    para('Measured warm-cache throughput for the selected batch-64/four-worker pipeline was 238.7 training images per second in the benchmark. Actual completed standard epochs averaged approximately 57–60 seconds, while stain-augmentation epochs averaged about 117–119 seconds. This joint pipeline comparison does not isolate the causal speedup from caching. CPU decode/decomposition benchmarks selected four preprocessing threads; the full cache was built in about 229 seconds. Two ADI extraction failures were retained as unchanged-image fallbacks.')
    heading('4.4 Quality Assurance',2,'sec44')
    para('Experiment outputs retain configuration, learning histories, class-level metrics, calibration summaries, confusion matrices, prediction probabilities and checkpoint hashes. The supervisor writes a live status and saved the requested morning snapshot at 08:59 IST. Local monitoring and document refresh make no ChatGPT API calls. The original application, database and checkpoint pointer are preserved. The evidence supports an exploratory progress report; no external quality certification, clinical validation or guaranteed similarity score is asserted.')
    para('SQLite implementation uses the prediction_records table through SQLAlchemy. It stores an integer record identifier, upload filename, predicted class, confidence, the eight-class probability payload and timestamp. This is application prediction history, not a verified patient registry. A simple query such as SELECT predicted_class, COUNT(*) FROM prediction_records GROUP BY predicted_class summarizes stored predictions; its counts are not disease prevalence. The dashboard weighted score is an uncalibrated heuristic, not a tumor-area estimate or clinical probability.')

    chapter(5,'Standards Adopted','chapter5')
    heading('5.1 Design Standards',2,'sec51')
    para('The project separates the web application, model definition, preprocessing, storage and research runner. The database schema and probability interface are explicit, and the architecture diagram identifies research/application boundaries. The paper source uses the IEEE conference class for formatting. Use of an IEEE document style does not imply that the prototype conforms to an IEEE engineering process or is an accepted publication.')
    heading('5.2 Coding Standards',2,'sec52')
    para('Python modules use named functions, explicit configuration, descriptive identifiers and isolated output directories. Numerical dtype and fallback behavior are recorded. Source-derived algorithms retain attribution. Deterministic preprocessing is separated from stochastic augmentation, with per-seed initialization and per-epoch/image random streams. Credentials, raw data, environments, databases and large checkpoints are excluded from portable research-source backups.')
    heading('5.3 Testing Standards',2,'sec53')
    para('Verification is based on reproducible evidence: duplicate hashes, split guards, loader checks, cached reference comparisons, finite numerical checks, saved-state recovery, best-model reload verification, isolated API/SQL rehearsal and artifact-preservation checks. These are project-level validation practices, not a claim of audited ISO or IEEE testing compliance. The final test set remains reserved, and patient-level generalization requires mapping or an external grouped cohort.')
    heading('5.4 Cost Analysis',2,'sec54')
    para('The study uses an existing PC and local storage; no new hardware purchase or paid cloud training cost is documented. Monetary cost cannot be calculated without an electricity tariff, measured wall energy and equipment-accounting assumptions. Measured elapsed time and VRAM are reported instead of inventing a price. The deterministic cache occupies roughly 25 GB locally, and GPU training retains approximately 2.1 GB VRAM in observed samples, subject to runtime variation.')
    para('A full seed-42 cycle used 40 standard, 34 fixed, 40 weaker and 40 adaptive epochs. Standard epochs averaged 59.7 seconds, and the three stain arms averaged roughly 117–119 seconds. Additional seeds are bounded by the existing 24-hour scheduling budget. CPU thread limits and rejection of batch 96 avoid oversubscription and excessive VRAM pressure. System clock rates and thermal protections are unchanged.')

    chapter(6,'Conclusion and Future Scope','chapter6')
    heading('6.1 Conclusion',2,'sec61')
    para('An eight-class gastric histopathology dashboard, decomposition-aware controller and isolated matched experiment pipeline have been implemented. The duplicate audit identifies a material limitation in the historical split. The short pilot did not establish controller benefit. The first larger matched seed has a small adaptive macro-F1 lead, alongside worse tumor and debris recall. This remains preliminary validation evidence, with no confirmed novelty, clinical benefit or patient-independent generalization.')
    heading('6.2 Future Scope',2,'sec62')
    para('Complete the bounded paired replications and summarize means and variability at equal seed counts. Inspect learning curves and class confusion before claiming convergence, especially for runs ending at the 40-epoch ceiling. Obtain patient or slide mapping, audit near duplicates and evaluate a locked design on a grouped or external cohort. Keep final test data outside all current tuning. Conduct a fuller related-work review before defining novelty, and investigate whether diagnostic control adds value beyond simply weakening augmentation.')
    para('The institutional report template is now supplied and used for this report. The certificate remains unsigned, academic-year interpretation needs guide confirmation, and a genuine Turnitin report must still be attached. A locally compiled IEEE source ZIP is available, but an authenticated Overleaf import and real project link are pending. Research checkpoint activation is not part of this study handoff.')

    page();heading('References',1,'references')
    refs=[
      '[1] S. Lou et al., “A large histological images dataset of gastric cancer with tumour microenvironment annotation for AI,” Scientific Data, vol. 12, art. 138, 2025. doi: 10.1038/s41597-025-04489-9.',
      '[2] T. Kim et al., “Stain SAN: simultaneous augmentation and normalization for histopathology images,” Journal of Medical Imaging, vol. 11, no. 4, art. 044006, 2024. doi: 10.1117/1.JMI.11.4.044006.',
      '[3] N. Marini et al., “Data-driven color augmentation for H&E stained images in computational pathology,” Journal of Pathology Informatics, vol. 14, art. 100183, 2023. doi: 10.1016/j.jpi.2022.100183.',
      '[4] Project source and experiment records: app/ml/model.py; app/db/models.py; database_schema.sql; stain_research/runs/overnight_20261113/*/metrics.json and history.json; split_audit_20261113/summary.json; classifier_local_20261113/pilot/summary.json. Local evidence snapshot '+data['as_of_display']+'.',
      '[5] KIIT School of Computer Science and Engineering, Project_Report-format..docx, report template supplied by the student on 13 November 2026. This reference establishes the report structure; its sample results and citations are not project evidence.'
    ]
    for text in refs:para(text,after=12)

    s=d.add_section(WD_SECTION_START.NEW_PAGE);geometry(s);s.left_margin=s.right_margin=s.top_margin=s.bottom_margin=Inches(1)
    for old in list(s._sectPr.findall(qn('w:pgNumType'))):s._sectPr.remove(old)
    heading('Individual Contribution Report',1,'contribution')
    center(TITLE,14,True);center(identity['name'].upper(),12);center('Roll Number '+identity['roll_number'],12)
    p=para('Abstract: This project implements eight-class gastric histopathology classification and examines diagnostic-guided stain augmentation under matched conditions. A dashboard and isolated training workflow support reproducible experiments. The first complete seed shows a small adaptive macro-F1 lead with worse tumor recall; replication and patient-level validation remain incomplete.')
    heading('Individual contribution and findings',2)
    para('I am presenting this as a solo project. The implementation scope described in the repository includes the dashboard, classifier integration, SQLite prediction history, stain-controller experiments and evaluation materials. I used software assistance for coding, auditing, running experiments and preparing this report; this assistance does not establish my unaided authorship of every line. The project guide should review the contribution statement and the experimental record before it is signed.')
    para('My project planning focuses on separating a working prototype from research claims. The work includes identifying duplicate leakage, retaining a provisional controlled manifest, comparing four augmentation arms with shared initial weights and budgets, keeping final test images outside tuning, and preserving resumable state. The current finding is a small single-seed macro-F1 difference accompanied by class-level tradeoffs. I should explain those limits and avoid presenting numerical stability as proof of classification improvement.')
    heading('Individual contribution to project report preparation',2)
    para('The report covers the literature, dataset audit, architecture, requirements, algorithms, SQL storage, verification, actual positive and negative results, development practices and future work. I am responsible for verifying its content, identity details, citations, institutional formatting and any disclosure required by KIIT. The certificate and plagiarism-report attachment remain pending, and no similarity score has been fabricated.')
    heading('Individual contribution for presentation and demonstration',2)
    para('The solo presentation explains the eight tissue classes, architecture, controller, duplicate audit and matched results. The demo checklist uses an isolated database and identifies the prototype model source honestly. I will distinguish saved research metrics from live application predictions and will not activate a research checkpoint merely to strengthen a demonstration.')
    for p in d.paragraphs:
        # Only contribution paragraphs require the explicit 1.5 spacing in the supplied sample.
        if p._p is s._sectPr:continue
    # Apply 1.5 lines to the contribution block via its section start location.
    contribution_start=next(i for i,p in enumerate(d.paragraphs) if p.text=='Individual Contribution Report')
    for p in d.paragraphs[contribution_start:]:p.paragraph_format.line_spacing=1.5
    para('Full signature of supervisor: ____________________\n\nFull signature of student: ____________________',before=18)

    page();heading('Turnitin Plagiarism Report',1,'plagiarism')
    para('Genuine report pending attachment',bold=True)
    para('No Turnitin submission or score has been produced by this workflow. The template contains an illustrative similarity screenshot; that sample has been removed because it is not evidence for this project. Attach the institution-generated report after submitting the actual final text through the required process.')
    para('The supplied template mentions a similarity threshold below 25%. Earlier transferred evaluation notes mentioned different thresholds. The current applicable threshold must be confirmed with the project guide or evaluation notice; this report does not resolve the discrepancy or guarantee any similarity or AI-detection result.')
    para('Submission identifier: ____________________\nSubmission date: ____________________\nVerified similarity result: ____________________\nGuide review: ____________________')
    if final:
        replacements={
         'Replication continues.':'All three matched seeds per arm have completed.',
         'The first complete seed shows a small adaptive macro-F1 lead with worse tumor recall; replication and patient-level validation remain incomplete.':'Three matched seeds per arm show a modest adaptive mean macro-F1 lead with worse mean tumor recall than standard; patient-level validation remains incomplete.',
         'The current result is mixed. The full seed-42 adaptive run has a small macro-F1 lead over standard augmentation, but reduced tumor recall and no completed matched replication.':'The completed three-seed result is mixed. Adaptive mean macro-F1 is 0.6836 versus standard 0.6776 and residual-fixed 0.6822, with lower tumor recall than standard and substantial uncertainty.',
         'Complete paired seed results and variability are needed; unequal completed seed counts must not be pooled into a treatment-effect claim.':'The completed paired seed analysis below supersedes the preliminary single-seed interpretation.',
         'The first larger matched seed has a small adaptive macro-F1 lead, alongside worse tumor and debris recall.':'The completed three-seed study has a small adaptive average macro-F1 lead, alongside worse mean tumor and debris recall than standard.',
         'Complete the bounded paired replications and summarize means and variability at equal seed counts.':'Three matched replications per arm are complete; assess the reported paired variability before deciding on another preregistered comparison.',
         'The current finding is a small single-seed macro-F1 difference accompanied by class-level tradeoffs.':'The completed finding is a modest three-seed mean macro-F1 difference accompanied by class-level tradeoffs.',
         'GitHub publication of this overnight package remains pending;':'GitHub publication status is recorded in DELIVERY_STATUS.json;'
        }
        for p in d.paragraphs:
            text=p.text
            for before,after in replacements.items():text=text.replace(before,after)
            if p.text.startswith('An exact-file audit found duplicate leakage'):
                text='An exact-file audit found duplicate leakage in the historical image splits. A provisional manifest retains 29,910 unique files; patient and slide independence remain unverified. Across three matched seeds per arm, adaptive augmentation reached mean accuracy 68.41% and macro-F1 0.6836, versus standard 67.70% and 0.6776. Residual-fixed reached 0.6822 macro-F1. Adaptive mean tumor recall was lower than standard. These small differences and three-seed uncertainty do not establish superiority. The earlier negative three-epoch pilot and preprocessing stability findings are reported separately.'
            if text!=p.text:
                if p.runs:p.runs[0].text=text;[setattr(r,'text','') for r in p.runs[1:]]
                else:p.add_run(text)

    def list_entries(anchor,items):
        previous=anchor
        for text,mark in items:
            p=para(text+'\t',11,False,WD_ALIGN_PARAGRAPH.LEFT,after=6)
            p.paragraph_format.tab_stops.add_tab_stop(Inches(6.05),2,1)
            field(p,'PAGEREF '+mark+' \\h','1');font(p,11)
            previous._p.addnext(p._p);previous=p
    list_entries(contents_anchor,markers)
    list_entries(fig_anchor,figures);list_entries(table_anchor,tables)
    update=OxmlElement('w:updateFields');update.set(qn('w:val'),'true');d.settings.element.append(update)
    target=build_dir/'Progress_Report.docx';generated=build_dir/'report_generated.docx';d.save(generated)
    # Retain all source opaque parts byte-for-byte. Only document structure,
    # related added media/header/footer parts and updateFields are editable.
    original={n:source_zip.read(n) for n in source_zip.namelist()}
    candidate=zipfile.ZipFile(generated)
    editable={'word/document.xml','word/_rels/document.xml.rels','[Content_Types].xml','word/settings.xml'}
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for name in candidate.namelist():
            z.writestr(name,original[name] if name in original and name not in editable else candidate.read(name))
        for name,content in original.items():
            if name not in candidate.namelist():z.writestr(name,content)
    final=zipfile.ZipFile(target)
    changed=[n for n in original if n not in editable and original[n]!=final.read(n)]
    if changed:raise RuntimeError('Preserve-only template parts changed: '+str(changed))
    (build_dir/'template_fidelity.json').write_text(json.dumps({'template_sha256':digest,'preserve_only_parts_unchanged':True,
         'preserved_part_count':len(original)-len(editable),'intentional_changes':'See .sync-backups/template_report/artifact.md',
         'as_of':data['as_of'],'identity_source':identity['source']},indent=2))
    return target


def architecture(path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,3.5));ax.set_axis_off()
    boxes=[(.03,.64,'Upload → FastAPI\noriginal predictor'),(.38,.64,'Eight probabilities\nSQLAlchemy → SQLite'),
           (.03,.16,'Train / validation only\nversioned stain cache'),(.38,.16,'Random augmentation\nCNN → BiLSTM → attention'),(.76,.16,'Best / last\ncheckpoints')]
    for x,y,t in boxes:ax.text(x,y,t,ha='left',va='center',fontsize=10,bbox=dict(boxstyle='square,pad=.6',facecolor='white',edgecolor='black'),transform=ax.transAxes)
    for start,end in [((.31,.64),(.37,.64)),((.31,.16),(.37,.16)),((.70,.16),(.75,.16))]:
        ax.annotate('',xy=end,xytext=start,xycoords='axes fraction',arrowprops=dict(arrowstyle='->'))
    ax.text(.03,.93,'Application path',weight='bold',transform=ax.transAxes)
    ax.text(.03,.40,'Isolated research path — no test data or dashboard activation',weight='bold',transform=ax.transAxes,fontsize=10)
    fig.savefig(path,dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig)


def results_plot(rows,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.4,3.3));ys=[r['macro_f1'] or 0 for r in rows]
    bars=ax.bar([LABELS[r['arm']] for r in rows],ys,color=['#bfbfbf','#dddddd','#999999','#555555'],edgecolor='black')
    ax.set_ylim(0,1);ax.set_ylabel('Validation macro F1');ax.spines[['top','right']].set_visible(False)
    for bar,y in zip(bars,ys):ax.text(bar.get_x()+bar.get_width()/2,y+.02,f'{y:.4f}',ha='center',fontsize=10)
    fig.tight_layout();fig.savefig(path,dpi=180,facecolor='white');plt.close(fig)


if __name__=='__main__':
    from refresh_evaluation import snapshot
    evidence=snapshot(ROOT/'stain_research/runs/overnight_20261113')
    print(build(evidence,ROOT/'.sync-backups/template_report/final_build'))
