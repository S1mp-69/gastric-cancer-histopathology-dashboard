"""Evidence-derived IEEE source and editable preparation notes."""
from pathlib import Path
import csv,json,shutil,statistics

def tex(s):
    return str(s).replace('\\',r'\textbackslash{}').replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('−','-').replace('±',r'$\pm$')

def build_text(d,build,package):
    paper=package/'overleaf';paper.mkdir(exist_ok=True)
    (paper/'figures').mkdir(exist_ok=True)
    shutil.copy2(build/'publication_validation.png',paper/'figures/validation.png')
    rows='\n'.join(tex(r['label'])+f" & {r['n']} & "+(f"${r['macro_f1_mean']:.4f}\\pm{r['macro_f1_sd']:.4f}$" if r['n']==3 else 'Pending replication')+r'\\' for r in d['hiesd'])
    hmu_rows='\n'.join(tex(r['label'])+f" & ${r['mean']:.4f}\\pm{r['sd']:.4f}$"+r'\\' for r in d['hmu'])
    codes=[('tumor','Tumor'),('mucosa_inflammation','Muc./infl.'),('intestinal_metaplasia','IM'),('lymphoid','Lymphoid'),('other','Other')]
    class_table=r'\begin{table}[t]\caption{HiESD validation mean per-class recall over three seeds. Support counts distinct validation cases once.}\centering\begin{tabular}{lrrrr}\toprule Class & $n$ & Std. & Shuf. & Adapt.\\\midrule'+'\n'
    class_table+='\n'.join(label+' & '+str(next(int(x['support']) for x in d['class_rows'] if x['class_code']==code))+' & '+' & '.join(f"{statistics.mean(x['recall'] for x in d['class_rows'] if x['class_code']==code and x['arm']==arm):.3f}" for arm in ['standard','shuffled','adaptive'])+r'\\' for code,label in codes)+r'\bottomrule\end{tabular}\end{table}'
    testtable=''
    if d['test_complete']:
        testtable=r'\begin{table}[t]\caption{Frozen reserved-patient test: three-seed mean and sample SD.}\centering\begin{tabular}{lr}\toprule Arm & Macro-F1 mean $\pm$ SD\\\midrule'+'\n'
        testtable+='\n'.join(tex(d['labels'][a])+f" & ${v['macro_f1']:.4f}\\pm{statistics.stdev(r['macro_f1'] for r in d['test']['per_seed'] if r['arm']==a):.4f}$"+r'\\' for a,v in d['test']['arm_means'].items())+r'\bottomrule\end{tabular}\end{table}'
        testtable+=r''' Adaptive's validation lead over within-class shuffled reverses on reserved patients: adaptive-minus-within-class differences are negative in all three test seeds. Global shuffled has the highest validation mean and the lowest test mean among these six arms; within-class shuffled has the highest test point estimate. These are descriptions of the frozen outputs, not a basis for choosing a new model or attributing the reversal to QC. The patient-cluster interval includes zero, so inferiority is not established. Adaptive test mean intestinal-metaplasia recall is 0.833 versus standard 0.768, while tumor recall is 0.622 versus standard 0.670. Other recall remains about 0.200 for adaptive; lymphoid support is only nine patches.'''
    source=r'''\documentclass[conference]{IEEEtran}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,booktabs,graphicx,url}
\title{Evaluating Decomposition-Guided Stain Augmentation in Gastric Histopathology}
\author{\IEEEauthorblockN{Amlan Choudhury}\IEEEauthorblockA{School of Computer Science and Engineering\\KIIT Deemed to be University}}
\begin{document}\maketitle
\begin{abstract}
ABSTRACT
\end{abstract}
\begin{IEEEkeywords}gastric histopathology, stain augmentation, controlled ablation, patient grouping, reproducibility\end{IEEEkeywords}
\section{Introduction}
Color variation in hematoxylin and eosin (H\&E) images motivates stain augmentation, but a numerically stable decomposition need not produce useful training perturbations. A controller based on reconstruction residual and basis conditioning might reduce unstable color changes; alternatively, its apparent effect may come from using weaker perturbations or assigning different strengths to tissue classes. These explanations require explicit controls.

We investigate whether source-specific decomposition diagnostics add classification value when the strength distribution and training conditions are controlled. An initial eight-class HMU experiment is followed by a patient-grouped HiESD experiment with a standard pretrained backbone. The two studies have different labels, preprocessing and architectures and are reported separately. Their score difference is not a cross-dataset treatment effect.

The empirical contribution is an auditable falsification study: exact-duplicate analysis, repeated matched augmentation comparisons, a class-conditional assignment control and transparent reporting of negative findings. We do not claim invention of stain augmentation, residual preservation or adaptive augmentation. A separate saved-prediction reweighting analysis illustrates a known evaluation property and is not proposed as a new metric.

\section{Related Work and Scope}
Lou et al. introduced the HMU gastric tissue patch dataset and evaluated classification methods under their release protocol \cite{lou}. Wang et al. released HiESD with gastric ESD slide annotations, patient metadata and quality masks \cite{wang}. Our selected subset and five-group mapping are not their full benchmark. Published AUC or differently constructed benchmark scores cannot be ranked against our validation macro-F1.

Stain SAN samples stain representations from a training-derived distribution \cite{kim}. Marini et al. investigate data-driven H\&E color augmentation and admissible variation \cite{marini}. Adaptive and quality-aware augmentation are established areas; changing the diagnostic formula alone does not demonstrate methodological novelty. The implemented extraction follows attributed Stain SAN-derived code. Our hypothesis concerns whether the source-to-strength association adds value beyond a matched strength bank.
Hong's 2026 CASA preprint calibrates adversarial stain budgets using multi-center statistics \cite{hong}. Its Camelyon17-WILDS evaluation addresses a different task and protocol. Our source-diagnostic rule provides no analogous coverage guarantee; its validation scores are not directly comparable with that study.

F1 depends on more than class recall. Brabec et al. show that prevalence changes can alter classifier order \cite{brabec}; Opitz examines evaluation properties and common practice \cite{opitz}. Thus a reversal under hypothetical weighting is expected to be possible. Its presence here motivates fuller class reporting, not a claim of a previously unknown phenomenon.

\section{Data, Provenance and Leakage Controls}
\subsection{HMU image-level study}
The local release contains 31,096 PNG images of size $224\times224$, with 3,887 per class: ADI, DEB, LYM, MUC, MUS, NOR, STR and TUM. An exact-file audit identified 932 duplicate groups containing 2,118 files. Of these, 334 groups crossed historical splits and 165 linked training and test. No cross-class exact-duplicate conflicts were found. Retaining original assignments while excluding redundant copies gives 23,761 training, 3,059 validation and 3,090 test images. There is no verified image-to-patient mapping; this is an exact-duplicate-controlled image split, not patient-independent validation. Near duplicates are not excluded by this audit.

\subsection{Patient-grouped HiESD subset}
HiESD contains 104 slides from 44 patients \cite{wang}. Patient allocation was frozen before acquisition: 30 training, seven validation and seven reserved-test patients, seed 20261116. Source files use recorded checksums; annotation and quality resources are pinned to release revision \texttt{89878d1}. The selected development subset has 2,543 training patches from 16 patients and 1,251 validation patches from seven patients. The class groups are tumor, mucosa/inflammation, intestinal metaplasia, lymphoid and other. The latter is heterogeneous; coarse annotation boundaries are not a segmentation ground truth of pixel-level precision.

Patches require at least 90\% annotation-group purity, at most 5\% artifact pixels, at most 5\% unknown-quality pixels and at least 80\% quality-mask tissue. A level-zero $896\times896$ region is resized to $224\times224$. Sampling uses a deterministic SHA order capped at 128 patches per patient and class. Nominal downsample-64 annotation dimensions must agree within two mask pixels; quality aspect ratio must agree within 0.5\%. One irregular annotation pyramid is excluded instead of stretched. These automated alignment and quality checks are not expert pathology review. Validation class supports are 99, 823, 166, 7 and 156 in the above order.

\section{Methods}
\subsection{Residual-preserving transformation}
Let an optical-density image be $X=BH+R$, with source basis $B$, concentrations $H$ and residual $R$. For a sampled target basis $B'$ and intensity factor $s$, the implementation uses
\begin{equation}
 X'=X+\alpha(sB'-B)H,
\end{equation}
on the tissue mask, followed by nonnegative optical-density clipping, RGB conversion and integer rounding. Untouched background pixels retain their source RGB values. At $\alpha=0$ the implementation returns the original image exactly. Residual preservation does not guarantee preserved diagnostic morphology or valid clinical staining.

The two diagnostics are $\log\kappa(B)$ and relative reconstruction residual norm. Training-set 10th and 90th percentiles $l_j,h_j$ scale them:
\begin{equation}
q=\frac12\sum_{j=1}^{2}\mathrm{clip}\left(\frac{d_j-l_j}{\max(h_j-l_j,10^{-8})},0,1\right),
\quad \alpha=1-0.9q.
\end{equation}
These are numerical proxies, not annotated pathology quality. Invalid decompositions use unchanged-image fallback. All 3,794 HiESD development decompositions passed extraction. Cached bases, concentrations and diagnostics remain attached to their source image. Target stain samples, intensity perturbations and augmentation geometry remain stochastic; the cache does not store final random augmented images.

\subsection{Matched arms and budgets}
Standard augmentation has no added decomposition-based transform. Residual-fixed uses $\alpha=1$; uniform-weaker uses the training mean adaptive strength (approximately 0.6135 in HiESD). Adaptive uses each image's diagnostic strength. Global shuffled permutes the adaptive strengths across valid training images every epoch. The within-class shuffle permutes within each training tissue class, preserving each class's exact strength multiset while breaking source-specific assignment. It is a training-only falsification control, not a deployed label-dependent predictor. The stronger control was added after earlier validation inspection and is explicitly post-hoc.

Seeds 42, 43 and 44 share seed-specific initial weights, sample order and augmentation streams across arms. In HiESD, each stain arm adds its transform before the same horizontal/vertical flips, right-angle rotation and RGB color jitter as standard. Shared streams isolate assignment differences without freezing randomness across epochs. All arms use torchvision ImageNet-1K V1 pretrained ResNet18, replacing its final layer with five outputs. The earlier HMU model instead uses a CNN over sixteen spatial subpatches, a two-layer bidirectional LSTM and additive attention; this spatial sequence is not a patient time series.

HiESD fine-tuning uses AdamW, learning rate $10^{-4}$, weight decay $10^{-4}$, label smoothing 0.05, gradient clipping at five, batch 16 and ImageNet normalization. FP16 gradient scaling is used. Validation negative log-likelihood drives ReduceLROnPlateau (factor 0.5, patience two). Best validation macro-F1 uses minimum gain $10^{-4}$, with early-stopping patience seven and a 40-epoch ceiling. Equal budgets mean identical stopping rules and maximum epochs, not forcing runs to identical observed stopping epochs. Best and last checkpoints include resumable optimizer, scheduler, scaler and random state.

\subsection{Verification and evaluation}
The audit checks initial-state hashes, train-only diagnostic fitting, source/cache alignment, finite model tensors, stopping and checkpoint selection, and saved-prediction metric reproduction. Forty cached/uncached stain-output checks were pixel-identical; shared geometry/RGB checks and all-epoch strength-multiset checks passed. The independent ranking audit reproduced 360 weighted confusion matrices and scores within $2.8\times10^{-15}$. These are integrity checks, not evidence of clinical validity.

Macro-F1 is the arithmetic mean of class F1 scores with zero division defined as zero. We also retain accuracy, per-class precision/recall, balanced accuracy, calibration and probability files. Means and sample standard deviations across three seeds describe training variability on the same cases; seeds are not new patient cohorts. Exploratory intervals with three seeds cannot support a strong population claim.

\section{Results}
\subsection{Completed validation comparisons}
\begin{table}[t]\caption{HMU exact-duplicate-controlled validation. Three seeds per arm; no verified patient mapping.}\centering\begin{tabular}{lr}\toprule Arm & Macro-F1 mean $\pm$ SD\\\midrule
HMU_ROWS
\bottomrule\end{tabular}\end{table}
In HMU, adaptive minus global shuffled mean macro-F1 is -0.0054; an exploratory three-seed $t$ interval includes zero. Adaptive mean tumor recall is 75.56\% versus standard 79.97\%. Thus the small adaptive gain over standard is not evidence for diagnostic-specific assignment.

\begin{table}[t]\caption{HiESD selected patient-grouped validation. Incomplete added controls remain explicitly partial.}\centering\begin{tabular}{lrr}\toprule Arm & Seeds & Macro-F1 mean $\pm$ SD\\\midrule
VALIDATION_ROWS
\bottomrule\end{tabular}\end{table}
The original HiESD five-arm experiment completed all 15 runs. Adaptive-minus-global-shuffled seed differences are -0.0273, +0.0008 and -0.0391; the mean is -0.0219. Standard reaches 0.7270 versus adaptive 0.7131. These data do not establish diagnostic-guided accuracy benefit. Added within-class-control status: WITHIN_STATUS.
WITHIN_TEXT

\begin{figure}[t]\centering\includegraphics[width=\linewidth]{figures/validation.png}\caption{HiESD validation macro-F1. Error bars are sample SD across three seeds on the same validation cases; only fully replicated arms are shown.}\end{figure}
The validation lymphoid support is only seven patches. The broad other class has weak recall. These failure modes limit claims based on a single aggregate score. Earlier v2/v3 experiments changed selected patients, quality filtering, normalization and architecture, so their score increase is not attributed to one intervention. A three-epoch pilot is reported as historical exploratory evidence, not convergence.
There is a class tradeoff: adaptive mean tumor recall is 0.906 versus 0.859 for standard and 0.828 for global shuffled, while other recall is 0.058 versus standard 0.092. Higher recall in one group does not establish overall superiority or clinical benefit.
CLASS_TABLE

\subsection{Secondary construction and weighting analyses}
A training-mask audit reconstructed the reference sampling exactly. Relaxing purity or removing patient-class caps changed membership substantially, but those variants were not retrained. A separate analysis reweighted the same 1,251 saved validation cases using 24 predefined, correlated recipes. Relative to natural weighting, 22 of 23 alternative recipes changed the winning arm and 21 changed the adaptive-minus-shuffled contrast sign. Lymphoid contributed about 69.4\% of the natural-to-equal-class contrast shift. Equal-class weighting increases its target mass from about 0.56\% to 20\%; this hypothetical target is not clinical prevalence. No claim is made that an actual QC intervention caused a ranking reversal. Class-only reweighting leaves recall and balanced accuracy unchanged.

An exploratory train-only rank-assignment feasibility study showed local loss-response continuity, including at initialization. Its subsequent exact finite-grid comparison did not distinguish ranked from random assignments: same-direction mean loss difference was about $+0.00030$ with an interval including zero, and the independent-direction difference was slightly negative. This proposal was not promoted to a new training campaign.

\subsection{Frozen reserved-patient test}
TEST_TEXT
TEST_TABLE
The protocol fixes all six arms, three seeds, validation-selected best checkpoints and the primary adaptive-minus-within-class-shuffled contrast before reserved-pixel acquisition. Exact development/test patch collisions or missing test classes stop confirmation rather than trigger cohort revision. Patient-cluster bootstrap uses 1,000 resamples and fixed five-class, zero-division-zero scoring, reporting draws missing classes. With only seven patient clusters, the interval is descriptive and potentially unstable. No tuning is permitted after test access.

\section{Implementation and Reproducibility}
The preserved demonstration application uses FastAPI, HTML/CSS/JavaScript, PyTorch inference, SQLAlchemy and SQLite prediction history. The original CNN/LSTM dashboard and HiESD ResNet18 experiment are separate implementations; research checkpoints are never activated in the dashboard. The isolated demo uses a separate database and training-only prototype references. An audit identified 41 historical prototype-reference paths overlapping the HMU provisional test split; this demo contamination is disclosed and does not involve the isolated research trainer. Historical notebook metrics are not current research test results.

Experiments use a fresh Python 3.12 environment with PyTorch 2.7/CUDA 12.8 on a 4-GB RTX 3050 Laptop GPU. HiESD uses two CPU threads and zero loader workers to avoid oversubscription. Versioned float64 caches preserve stochastic target augmentation. The earlier HMU warm-cache pipeline measured 238.7 training images per second; it used a different model/batch/workload and is not a ResNet18 throughput claim. Logs and an independent local supervisor retain restartable state without repeated ChatGPT requests. Manifests, source hashes, configuration, saved probabilities, QC receipts and audit scripts accompany the portable source; raw slides, private databases, environments and large model artifacts are excluded from Git.

\section{Limitations and Conclusion}
Repeated reuse of development validation, post-hoc control addition and exploratory secondary analyses limit confirmatory interpretation. The selected HiESD subset is small, imbalanced and annotation-driven; mask boundaries are coarse and no local expert adjudication has occurred. Patient grouping controls known IDs within the public release, but held-out public-release patients do not establish independent cross-hospital or clinical generalization. The HMU study has no verified patient mapping, and exact-file checks do not resolve near duplicates.
The shuffled controls preserve strength distributions, not realized pixel-change energy: the latter also depends on the receiving image and sampled stain transformation. Patient-cluster intervals condition on the three trained seeds and are not a precise estimate of generalization across institutions or training seeds.

The original validation comparisons and frozen patient test do not establish a classification benefit for the tested diagnostic-assignment rule. The results emphasize conditional permutation controls, provenance audits and separate validation/test reporting when evaluating heuristic stain policies. The selected cohort does not establish algorithmic priority or clinical usefulness.

\section*{Data, Code and Assistance Statement}
Public data must be obtained from the original HMU and HiESD releases under their terms; source records preserve their identifiers and checksums. Research source and portable evidence are maintained in the existing project research branch. No new human recruitment or clinical deployment is reported; institutional review requirements for public de-identified data remain for the authors to confirm. Author, funding, competing-interest and ethics declarations require final human confirmation. AI assistance was used for coding, analysis and drafting; the named author must verify all claims, citations and responsibility before submission. This draft has not received a Turnitin assessment or peer review. Snapshot: SNAPSHOT.
Repository and numerical evidence: \url{https://github.com/S1mp-69/gastric-cancer-histopathology-dashboard/tree/research/local-backup-2026-11-13}. Raw images and large checkpoints are excluded from the portable package.
\begin{thebibliography}{9}
\bibitem{lou} S. Lou et al., ``A large histological images dataset of gastric cancer with tumour microenvironment annotation for AI,'' \emph{Scientific Data}, vol.12, art.138, 2025, doi:10.1038/s41597-025-04489-9.
\bibitem{wang} C. Wang et al., ``A fully annotated pathology slide dataset for early gastric cancer and precancerous lesions,'' \emph{Scientific Data}, 2025, doi:10.1038/s41597-025-05679-1.
\bibitem{kim} T. Kim et al., ``Stain SAN: simultaneous augmentation and normalization for histopathology images,'' \emph{Journal of Medical Imaging}, vol.11, no.4, art.044006, 2024, doi:10.1117/1.JMI.11.4.044006.
\bibitem{marini} N. Marini et al., ``Data-driven color augmentation for H\&E stained images in computational pathology,'' \emph{Journal of Pathology Informatics}, vol.14, art.100183, 2023, doi:10.1016/j.jpi.2022.100183.
\bibitem{hong} M. Hong, ``Physics-grounded adversarial stain augmentation with calibrated coverage guarantees,'' arXiv:2605.13889, 2026, preprint.
\bibitem{brabec} J. Brabec, T. Komarek, V. Franc, and L. Machlica, ``On model evaluation under non-constant class imbalance,'' \emph{ICCS}, 2020, arXiv:2001.05571.
\bibitem{opitz} J. Opitz, ``A closer look at classification evaluation metrics and a critical reflection of common evaluation practice,'' \emph{Transactions of the ACL}, vol.12, pp.820--836, 2024, doi:10.1162/tacl\_a\_00675.
\end{thebibliography}\end{document}
'''
    for key,val in [('ABSTRACT',tex(d['abstract'])),('HMU_ROWS',hmu_rows),('VALIDATION_ROWS',rows),('CLASS_TABLE',class_table),('WITHIN_STATUS',tex(d['within_status'])),('WITHIN_TEXT',tex(d['within_text'])),('TEST_TEXT',tex(d['test_text'])),('TEST_TABLE',testtable),('SNAPSHOT',tex(d['as_of']))]:source=source.replace(key,val)
    if d['completed_hiesd_runs']==18:source=source.replace('Incomplete added controls remain explicitly partial.','Three matched seeds per arm.')
    (paper/'main.tex').write_text(source,encoding='utf-8')
    (paper/'README.md').write_text('Compile main.tex with IEEEtran.cls in this folder. This is a guide-review draft, not an accepted paper.\n\nOverleaf: New Project → Upload Project → select Overleaf_Paper_Source.zip → keep private → set main.tex as main document → compile → record the actual project URL. No authenticated project link has been created by this workflow.\n',encoding='utf-8')
    note=f'''# Publication package — {d['as_of']}

Use Progress_Report.docx/PDF (prescribed teacher structure), Progress_Presentation.pptx/PDF (editable), IEEE_Paper_Draft.pdf, and Overleaf_Paper_Source.zip.

## Verified finding
Adaptive validation macro-F1 is below global shuffled in both original three-seed studies: HMU 0.6836 vs 0.6890; HiESD 0.7131 vs 0.7350. This supports a controlled empirical evaluation, not an accuracy-superiority or guaranteed-novelty claim.

HiESD completed runs: {d['completed_hiesd_runs']}/18. Within-class control: {d['within_status']}.

{d['test_text']}

## Evidence
publication_results/PUBLICATION_SNAPSHOT.json; publication_readiness/publication_audit_20261117/AUDIT.json; publication_readiness/ranking_audit_20261117; repository stain_research/results. The frozen final-test protocol lives in stain_research/runs/final_test_hiesd_v1 and a portable copy accompanies this package. No post-test tuning.

## Remaining human requirements
Read and verify the manuscript; obtain guide critique and approval; confirm ethics/funding/conflict declarations, author email and appropriate authorship; attach genuine institutional Turnitin reports; create a private authenticated Overleaf project and record its real link. Teacher thresholds are similarity below 10% and AI score below 20%; neither is guaranteed. No journal submission, payment or external message has been sent.

## Local continuation without ChatGPT polling
The bounded within-class queue continues the three seed controls, then the frozen final-test pipeline and local artifact refresh. Supervisor: .sync-backups/hiesd_progress/run_within_publication.ps1. If interrupted, rerun that same supervisor; it resumes saved state and skips completed runs. Do not launch a second trainer or search new hyperparameters after reserved-test access.
'''
    (package/'START_HERE.md').write_text(note,encoding='utf-8')
    (package/'PUBLICATION_READINESS.md').write_text(note,encoding='utf-8')
    script='''# Solo presentation script

Opening: I tested whether numerical stain-decomposition diagnostics help decide augmentation strength for gastric tissue classification. My central result is a controlled negative finding; I am not claiming a new clinically useful classifier.

Data: The first dataset has eight tissue classes. Exact duplicates crossed its historical splits, so I created a provisional exact-file-controlled manifest. Without image-to-patient IDs I cannot claim patient independence. The second dataset, HiESD, supplies patient IDs; I froze patient partitions before acquisition and screened a selected annotated subset using released quality masks.

Method: A stain model separates basis, concentration and reconstruction residual. The controller weakens perturbations when numerical diagnostics look worse. I compare standard augmentation, fixed full strength, one uniform mean strength, a global shuffle, adaptive source assignment and a within-class shuffle. The last control preserves each tissue class strength bank but removes which image gets which strength. It is an evaluation control, not a deployment feature.

Fairness: The same seed shares initial weights, split, geometry, batch order, optimizer and stopping rules. Runs have a 40-epoch ceiling, seven-epoch early stopping and full resumable checkpoints. The original HMU architecture is CNN/BiLSTM/attention; HiESD uses pretrained ResNet18, so I do not attribute cross-study score differences to one change.

Results: HMU adaptive mean validation macro-F1 is 0.6836 against shuffled 0.6890. HiESD adaptive is 0.7131 against shuffled 0.7350 and standard 0.7270. The controller does not win on average. Lower numerical noise amplification is not proof of better classification. I retain these negative results.

Secondary analysis: Rankings change under hypothetical class and patient weighting. This is a known property of F1, independently checked on saved predictions. It is not a measured effect of changing QC and not an assumed clinical prevalence. Seven validation lymphoid patches make that class particularly fragile.

Final evaluation: Read the frozen-test status on the final-results slide. Explain the prespecified adaptive-minus-within-class contrast, all-arm reporting, small patient-cluster uncertainty and prohibition of tuning after test access. Do not quote pending values as completed results.

Demonstration: The application accepts a patch, displays eight probabilities and stores a prediction row in isolated SQLite. It uses the original configured predictor, not the research ResNet18. Prototype fallback is explicitly identified and training-only reference images are used by the demo wrapper. The output is not a patient diagnosis.

Closing: My contribution is a reproducible empirical study with a stronger assignment control and transparent limitations. Algorithmic novelty, clinical value and publication acceptance remain unconfirmed.
'''
    (package/'SOLO_PRESENTATION_SCRIPT.md').write_text(script+'\nValidation class tradeoff: HiESD adaptive tumor recall is 0.906 versus standard 0.859; other recall is 0.058 versus standard 0.092. This is not overall or clinical superiority.\n\n'+d['within_text']+'\n\nCurrent snapshot: '+d['test_text']+'\n'+('On the frozen test, adaptive intestinal-metaplasia recall is 0.833 versus standard 0.768, but tumor recall is 0.622 versus standard 0.670. The validation benefit against within-class shuffle reverses. No parameters or checkpoints are changed after this result.\n' if d['test_complete'] else ''),encoding='utf-8')
    viva={
     'What is the novelty?':'No algorithmic priority is established. The proposed empirical contribution is a matched evaluation of diagnostic-to-strength assignment, with a class-conditional permutation falsification control and data-integrity evidence. Venue suitability requires guide review.',
     'Why can a negative result be research?':'A properly controlled experiment can reject a plausible hypothesis. It must answer a useful question and meet scientific and venue standards; a failed method alone does not guarantee publication.',
     'Why within-class shuffled?':'A global shuffle changes class-strength association. Permuting within tissue class preserves that association while breaking source-specific diagnostic assignment. It uses training labels only.',
     'Does the adaptive controller improve accuracy?':'The completed original replicated validation studies do not show benefit over shuffled strength assignment. Use the frozen-test snapshot for the final contrast, without tuning to its result.',
     'Are patients independent?':'HiESD partitions use supplied patient IDs with disjoint groups. The selected public-release test is not independent cross-hospital validation. HMU has no verified image-to-patient mapping.',
     'Why not compare published accuracies?':'Different cohorts, labels, preprocessing, splits, models and metrics prevent a fair direct rank. Compare methods qualitatively and report our matched same-split experiment numerically.',
     'What does macro-F1 mean?':'Compute F1 from precision and recall separately for each class, then take the arithmetic mean. It can change under prevalence shifts because precision changes.',
     'Does reweighting prove your QC changes model rankings?':'No. It reweights fixed saved predictions under hypothetical target mixtures. Actual QC variants would require separate controlled data construction and retraining.',
     'What are the main failures?':'Other has poor recall; only seven validation lymphoid patches make rare-class estimates unstable. Coarse masks and selected patient coverage further limit inference.',
     'What is stored in SQL?':'Prediction ID, image filename, predicted tissue class, confidence, probability payload and timestamp. It is application history, not a verified patient registry.',
     'What remains for submission?':'Guide review, verified declarations and references, final authentic Overleaf URL, genuine Turnitin outputs and venue-specific formatting/requirements. No guaranteed acceptance or AI-detection score.',
    }
    (package/'VIVA_ANSWERS.md').write_text('# Viva answers\n\n'+'\n\n'.join('## '+q+'\n\n'+a for q,a in viva.items()),encoding='utf-8')
    (package/'DEMO_CHECKLIST.md').write_text('''# Demonstration checklist

1. Keep the research supervisor running. Do not start another trainer or activate a research checkpoint.
2. Start the preserved app through `& .venv-research/Scripts/python.exe stain_research/demo_serve.py` (CPU default, isolated DB/uploads and training-only prototype references). Open http://127.0.0.1:8000.
3. Upload an HMU training example; identify the returned model_source. A prototype fallback must be called prototype-baseline, not a validated research model.
4. Explain eight tissue probabilities and prediction history; do not describe the heuristic weighted score as a clinical diagnosis, disease prevalence or tumor area.
5. Show saved scientific comparisons separately. HiESD ResNet18 is a different isolated research model with five groups.
6. Explain the frozen final-test protocol and actual current status. Do not tune after test access.
7. Stop only the demo server with Ctrl+C when finished; leave the research supervisor/checkpoints intact.

The original demo historically initialized prototypes from all_image, including 41 provisional HMU test paths. The wrapper fixes its reference source without editing the original application. This historical exposure is disclosed; it does not turn HMU into patient-independent data.
''',encoding='utf-8')
    (package/'OVERLEAF_IMPORT_INSTRUCTIONS.md').write_text((paper/'README.md').read_text(encoding='utf-8'),encoding='utf-8')
    (package/'MENTOR_REVIEW.md').write_text('''# Guide review required before submission

Proposed paper: Evaluating Decomposition-Guided Stain Augmentation in Gastric Histopathology.

- Is this controlled negative study a useful empirical contribution for the selected venue? Do not frame it as proven algorithmic invention.
- Review patient allocation, selected-subset QC, coarse-mask limitations, rare-class support and HMU historical prototype exposure.
- Review all-seed results and frozen-test contrast, including uncertainty and per-class failures. Reweighting is secondary and uses known metric behavior.
- Confirm the student can explain the code, method and figures and that authorship, institutional affiliation, funding, conflicts, ethics/data-use declarations and AI assistance are correct.
- Confirm final-evaluation report format, genuine Turnitin outputs and authentic Overleaf project URL.

No mentor message has been sent by this workflow. Share the package yourself after checking it.
''',encoding='utf-8')
    failures='# Per-class results\n\nValidation uses the same 1,251 cases across seeds; support counts are not multiplied by the number of models. Means below describe training-seed variability, not new patient cohorts. Partial arms are labeled by their completed seed count.\n\n| Arm | Class | Seeds | Support | Precision | Recall | F1 |\n|---|---|---:|---:|---:|---:|---:|\n'
    for arm,label in d['labels'].items():
        for code,_ in codes:
            items=[r for r in d['class_rows'] if r['arm']==arm and r['class_code']==code]
            if items:failures+=f"| {label} | {code} | {len(items)} | {int(items[0]['support'])} | "+' | '.join(f"{statistics.mean(r[k] for r in items):.4f}" for k in ['precision','recall','f1-score'])+' |\n'
    if d['test_complete']:
        failures+='\nFrozen reserved-test results (all three seeds; no tuning):\n\n| Arm | Class | Support | Mean precision | Mean recall | Mean F1 |\n|---|---|---:|---:|---:|---:|\n'
        for arm,label in d['labels'].items():
            for code,_ in codes:
                items=[r['per_class'][code] for r in d['test']['per_seed'] if r['arm']==arm]
                failures+=f"| {label} | {code} | {int(items[0]['support'])} | "+' | '.join(f"{statistics.mean(r[k] for r in items):.4f}" for k in ['precision','recall','f1-score'])+' |\n'
    (package/'PER_CLASS_RESULTS.md').write_text(failures,encoding='utf-8')
    ledger='# Claim–evidence ledger\n\n| Claim | Evidence / permitted wording |\n|---|---|\n| Controller implemented | Code, cache-equivalence and saved-run audit receipts support implementation. |\n| Numerical benefit | Initial positive noise result; matched follow-up intervals include zero. Not classifier accuracy. |\n| Adaptive superiority | Original three-seed validation means are below global shuffled on both studies. No superiority claim. |\n| Within-class contrast | '+d['within_text']+' |\n| Patient independence | HiESD known IDs are disjoint within the public release; HMU mapping absent. No cross-hospital claim. |\n| Metric novelty | Reweighting illustrates established F1 prevalence sensitivity, not a new discovery. |\n| Algorithmic novelty | Unconfirmed. Residual preservation and adaptive augmentation have prior art. |\n| Clinical benefit | Unconfirmed; no clinical deployment or expert outcome study. |\n| Overleaf | Uploadable ZIP compiled locally; authenticated project/link pending. |\n| Turnitin | Genuine institutional reports pending; no score guarantee. |\n\n'+d['test_text']+'\n'
    (package/'CLAIM_EVIDENCE_LEDGER.md').write_text(ledger,encoding='utf-8')
    if d['test_complete']:
        result='# Frozen reserved-patient test\n\n'+d['test_text']+'\n\n| Arm | Mean accuracy | Mean macro-F1 | Mean balanced accuracy | ECE (15 bins) | Brier |\n|---|---:|---:|---:|---:|---:|\n'
        for arm,row in d['test']['arm_means'].items():result+='| '+d['labels'][arm]+' | '+' | '.join(f'{row[k]:.4f}' for k in ['accuracy','macro_f1','balanced_accuracy','ece_15','brier'])+' |\n'
        result+='\nAll 18 seed results:\n\n| Seed | Arm | Accuracy | Macro-F1 | Balanced accuracy |\n|---|---|---:|---:|---:|\n'
        for row in d['test']['per_seed']:result+=f"| {row['seed']} | {d['labels'][row['arm']]} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['balanced_accuracy']:.4f} |\n"
        result+='\nNo model selection or tuning follows these results. The same patients/cases recur across seeds; do not multiply the patient count by 18. See PER_CLASS_RESULTS.md for all class failures and independent verification receipts under publication_results.\n'
        (package/'FINAL_TEST_RESULTS.md').write_text(result,encoding='utf-8')
        morning=f'''# Morning status — 17 November 2026

Prepared {d['as_of']} in Asia/Calcutta, before the 09:00 reporting milestone. This is a status snapshot, not a training stop instruction.

## Completed and active work

- HMU: 15 matched runs complete, five arms × seeds 42/43/44. Exact-file control only; no verified patient mapping.
- HiESD: 18 matched runs complete, six arms × seeds 42/43/44. Added within-class runs stopped at 22, 22 and 27 epochs; selected best epochs 15, 15 and 20. No three-epoch convergence claim.
- Frozen reserved-patient test: all 18 models evaluated on 1,613 patches from seven patients. Independent metrics and patient-bootstrap verification passed. Raw data, databases and checkpoints remain local; research checkpoints are not activated in the dashboard.
- Active training runs: none. Current training epoch: not applicable. Remaining model-training/test time: zero. No additional training is launched after reserved-test access.
- Local artifact generation and export status: PUBLICATION_BUILD.json and its snapshot/hash receipt. Original model artifacts remain in stain_research/runs; source and portable report synchronization is recorded separately in CLOUD_SYNC_STATUS.json.

## Validation and final results

HiESD validation mean macro-F1: standard 0.7270, residual fixed 0.7260, uniform weaker 0.7220, global shuffled 0.7350, adaptive 0.7131, within-class shuffled 0.7063. Adaptive-minus-within-class validation mean is +0.0068 with exploratory n=3 interval [-0.0077,+0.0214].

{d['test_text']}

Final-test mean macro-F1: standard 0.6600, residual fixed 0.6696, uniform weaker 0.6495, global shuffled 0.6350, adaptive 0.6602, within-class shuffled 0.6749. All individual seeds and secondary metrics are in FINAL_TEST_RESULTS.md and publication_results/final_test_RESULTS.json. The same cases recur across models and seeds.

## Class failures and limits

Adaptive test recall: tumor 62.24%, mucosa/inflammation 87.15%, intestinal metaplasia 83.25%, lymphoid 100% on only nine patches, other 19.96%. Standard tumor recall is 66.98% and intestinal-metaplasia recall 76.85%. These class tradeoffs and sparse lymphoid support prevent clinical claims. PER_CLASS_RESULTS.md includes every arm. The expert-review CSV is a queue, not completed pathology adjudication or corrected labels.

## Deliverables in this same evaluation_package

Progress_Report.docx / Progress_Report.pdf (prescribed teacher structure); Progress_Presentation.pptx / Progress_Presentation.pdf; IEEE_Paper_Draft.pdf; Overleaf_Paper_Source.zip; Research_Evaluation_Package.zip; SOLO_PRESENTATION_SCRIPT.md; VIVA_ANSWERS.md; DEMO_CHECKLIST.md; CLAIM_EVIDENCE_LEDGER.md; MENTOR_REVIEW.md; VENUE_AND_SUBMISSION_PLAN.md.

Remaining human requirements: guide review/signatures, verified declarations and references, a real private Overleaf project after login, and genuine institutional Turnitin reports. No novelty, acceptance, clinical benefit or detection-score guarantee is made. No journal submission, payment or external message has been sent.
'''
        (package/'MORNING_STATUS.md').write_text(morning,encoding='utf-8')
        (package/'MORNING_STATUS_20261117_0900.md').write_text(morning,encoding='utf-8')
    (package/'VENUE_AND_SUBMISSION_PLAN.md').write_text('''# Venue and submission plan — 17 November 2026

Do not submit a method-superiority paper on the current evidence. First complete the frozen control/test package and obtain guide assessment of a rigorous empirical/negative-results framing. No publication is guaranteed.

PLOS ONE is a possible broad-scope assessment candidate because its official criteria explicitly consider negative/null results and require rigorous controls, adequate samples, appropriate conclusions and reproducibility: https://journals.plos.org/plosone/s/criteria-for-publication . This is a fit lead, not an endorsement, acceptance prediction or fee quote. The selected subset and few patient clusters may remain insufficient. Read current scope, fees/waiver eligibility and guide advice before choosing.

AI assistance must be disclosed as required by the institution and chosen journal; PLOS has an explicit policy: https://journals.plos.org/plosone/s/ethical-publishing-practice . Do not add an AI system as an author or promise Turnitin outcomes.

For an IEEE conference submission, verify the particular conference scope, real publisher/indexing record, current call/deadline, peer review, page limit, fees and prior negative/empirical papers. The supplied IEEE source meets the teacher formatting requirement; it does not identify a suitable conference by itself. No invented deadline or guaranteed indexing is offered. Budget is unconfirmed; no payment or submission is authorized by this file.

Submission gate: final test completed or blocker disclosed; no unsupported novelty/clinical claim; complete uncertainty/per-class reporting; actual literature checked; code/data-access and license statements; human-verified declarations and authorship; authentic Overleaf URL; genuine institution-issued Turnitin reports; guide approval; chosen venue formatting checked. Do not submit simultaneously to multiple venues.
''',encoding='utf-8')
