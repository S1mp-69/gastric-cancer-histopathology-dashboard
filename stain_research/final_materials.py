"""Finish portable evaluation text using the audited completed campaign."""
import json,shutil,zipfile,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def finish_package(data,package):
 a=data['final_analysis'];identity=json.loads((ROOT/'stain_research/report_identity.json').read_text())
 source_dir=ROOT/'stain_research/results/final_campaign_20261113'
 dest=package/'final_results';dest.mkdir(exist_ok=True)
 for source in source_dir.iterdir():
  if source.is_file():shutil.copy2(source,dest/source.name)
 paper=package/'overleaf/main.tex';text=paper.read_text(encoding='utf-8')
 text=text.replace(r'\author{\IEEEauthorblockN{Amlan}}',r'\author{\IEEEauthorblockN{Amlan Choudhury}\IEEEauthorblockA{School of Computer Science and Engineering\\KIIT Deemed to be University}}')
 text=text.replace('active progress snapshot','completed exploratory campaign')
 section=r'''
\section{Completed Three-Seed Comparison}
All 12 bounded runs completed, using seeds 42, 43 and 44 for each arm.
Results below are validation mean and sample standard deviation across seeds;
the same 3,059 validation images recur, so these are not independent cohorts.
\begin{table}[t]\caption{Completed campaign validation results, three seeds per arm.}\centering
\begin{tabular}{lrr}\toprule
Arm & Accuracy (\%) & Macro-F1\\\midrule
Standard & $67.70\pm1.15$ & $0.6776\pm0.0106$\\
Residual fixed & $68.30\pm1.38$ & $0.6822\pm0.0141$\\
Uniform weaker & $68.22\pm0.63$ & $0.6804\pm0.0061$\\
Adaptive & $68.41\pm0.59$ & $0.6836\pm0.0047$\\
\bottomrule\end{tabular}\end{table}
Adaptive minus standard macro-F1 differences were 0.0021, 0.0023 and 0.0137;
adaptive minus fixed differences were 0.0227, -0.0101 and -0.0084. The mean
adaptive advantage over fixed was only 0.0014. Exploratory paired 95\% Student
$t$ intervals with two degrees of freedom were [-0.0105,0.0225] versus standard,
[-0.0445,0.0473] versus fixed and [-0.0223,0.0286] versus weaker. These wide
intervals assume approximately normal independent seed differences, use only
three seeds, are not multiplicity adjusted and do not quantify patient uncertainty.
No superiority claim follows from them.

Mean TUM recall was 79.97\% for standard and 75.56\% for adaptive; mean DEB
recall was 65.72\% and 62.89\%, respectively. Adaptive improved mean ADI, MUS,
NOR and STR recall, but had higher mean ten-bin calibration error (0.0213 versus
0.0157). The numerical tradeoffs preclude a uniform or clinical-benefit claim.
Checkpoint hashes, per-seed initialization hashes, common configurations,
ordered validation identities, finite best-model tensors, saved prediction
metrics and resumable-state keys passed audit. This does not independently
repeat all inference or prove bit-identical interrupted optimization. AMP and
Windows file-lock recoveries remain documented. The final test set is untouched.
Shuffled diagnostic assignments were not tested at this full budget. Patient
mapping, near-duplicate control and independent external validation are unresolved.

'''
 text=text.replace(r'\section{Discussion and Limitations}',section+r'\section{Discussion and Limitations}')
 text=text.replace('Next steps are a converged baseline, verified grouping where available, multiple seeds and fair controlled comparisons.','Three matched seeds per arm are now complete. Next steps are verified grouping, a fuller related-work review and a locked independent evaluation design.')
 text=text.replace('One seed, three epochs, a small validation sample and prior exploratory validation use restrict inference.','The historical pilot used one seed and three epochs. The completed campaign uses three seeds, but shares validation images and validation-based checkpoint selection; prior exploratory validation use also restricts inference.')
 text=text.replace("The evidence supports a progress-stage study and bounded follow-up, rather than a claim of improved gastric cancer diagnosis.","The completed campaign has a modest adaptive mean macro-F1 lead with worse tumor recall than standard and sign changes against fixed control. Superiority, novelty and improved diagnosis remain unconfirmed.")
 abstract=r'''We investigate whether stain-decomposition diagnostics can guide residual-preserving augmentation for eight-class gastric tissue classification. A CNN--BiLSTM--attention classifier is evaluated using standard, residual-fixed, uniformly weaker and adaptive augmentation. An exact-file-controlled provisional split retains 23,761 training and 3,059 validation images; patient independence remains unverified. All 12 bounded runs completed across three matched seeds per arm. Adaptive mean validation macro-F1 was $0.6836\pm0.0047$, versus $0.6776\pm0.0106$ for standard and $0.6822\pm0.0141$ for fixed augmentation. Mean tumor recall fell from 79.97\% for standard to 75.56\% for adaptive. Paired seed differences against fixed changed sign, and exploratory uncertainty was wide. A historical three-epoch pilot did not establish benefit beyond weaker controls. Preprocessing numerical stability, classifier validation and legacy notebook metrics are kept distinct. The results support a reproducible exploratory method and identify data-provenance limitations, but do not establish superiority, novelty or clinical benefit. Final test inference remains reserved.'''
 text=re.sub(r'(?s)(\\begin\{abstract\}).*?(\\end\{abstract\})',lambda m:m.group(1)+'\n'+abstract+'\n'+m.group(2),text)
 text=text.replace('Seed 42 values\nare best-so-far validation selections. Active runs are incomplete.', 'Seed 42 illustration; all four runs are complete and use their selected best checkpoints.')
 paper.write_text(text,encoding='utf-8')
 (package/'overleaf/README.md').write_text('Private Overleaf import pending. Upload the ZIP, choose main.tex, use pdfLaTeX, and compile. No project link exists. Local Tectonic build is verified separately. Author details supplied by the student; guide is not listed as a coauthor without authorization.\n')
 with zipfile.ZipFile(package/'Overleaf_Paper_Source.zip','w',zipfile.ZIP_DEFLATED) as z:
  for f in (package/'overleaf').iterdir():
   if f.is_file():z.write(f,f.name)
 (package/'Presentation_Script_and_Viva.md').write_text('''# Solo presentation script and viva

Use the 15-slide deck. Aim for 8–10 minutes; the brief demo may add 2 minutes.
Author: Amlan Choudhury, 23051241. Guide: Dr. Asif Uddin Khan. Year supplied: 2027.

1. Title (20 sec): I study gastric tissue classification and adaptive stain augmentation. The application works; the research outcome is modest and mixed.
2. Problem (30 sec): H&E appearance varies. I ask whether decomposition reliability can guide augmentation strength rather than changing every image equally.
3. Dataset (35 sec): 31,096 patches, eight classes. Clinical tables are separate and cannot link patches to patients in this release.
4. Application (30 sec): FastAPI, HTML/CSS/JavaScript and SQLite prediction history. This is a prototype, not clinical diagnosis.
5. Architecture (40 sec): Sixteen spatial patches, CNN features, two-layer bidirectional LSTM, additive attention and eight logits. Spatial sequence does not represent time.
6. Prior work (35 sec): Stain SAN supplies sampling/decomposition foundations; admissibility-based color augmentation is related work. The proposed diagnostic signal is not automatically novel.
7. Controller (50 sec): Measure decomposition error and conditioning, normalize with training-fitted ranges, reduce stain perturbation for risky images, preserve residual information and retain extraction failures unchanged.
8. Preprocessing study (35 sec): Mean synthetic-noise gain favored adaptive, but descriptive intervals included no improvement. Numerical stability is not classifier validation.
9. Historical pilot (25 sec): The six-arm 512/128-image, three-epoch pilot was short, not convergence.
10. Negative pilot results (30 sec): Adaptive F1 0.2651 did not beat weaker 0.2802 or shuffled 0.2768. I retain that negative result.
11. Duplicate audit (40 sec): 932 duplicate groups, 334 crossing historical splits, 165 spanning train/test. The retained manifest has 23,761 train, 3,059 validation and 3,090 test. Exact-file control does not prove patient independence.
12. Completed research (40 sec): Four arms times three seeds are complete. Adaptive mean tumor recall 75.56% was worse than standard 79.97%. Patient mapping and a locked independent evaluation remain needed.
13. Final table (60 sec): Standard mean F1 0.6776, fixed 0.6822, weaker 0.6804, adaptive 0.6836. SDs describe seeds, not patients. Adaptive beats fixed in one of three seeds. Mean fixed difference is only 0.0014 and exploratory paired intervals include zero. I do not claim superiority.
14. Compute (30 sec): CUDA training, batch64/four workers, mixed precision, deterministic caching and random augmentation. Last/best checkpoints are separate. All twelve completed outputs were audited; recoveries remain in logs.
15. Demo and conclusion (45 sec): Show a TRAIN patch, model source, eight probabilities and isolated SQL history. The demo uses prototype-baseline, not research weights. Conclude with the small average gain, tumor-recall regression and remaining validation work.

## Viva answers

**What is the research contribution?** A candidate decomposition-aware strength controller plus a matched, auditable evaluation. It is not a proven novel superior algorithm.

**Why four arms?** Standard tests baseline benefit, residual-fixed tests value beyond residual preservation, weaker tests whether simply reducing strength is sufficient, and adaptive tests the proposed assignment.

**What would show the diagnostics matter?** A full-budget shuffled-assignment control, under a fixed protocol. It was only tested in the small pilot; no new campaign has been launched.

**Why not claim improvement?** Three seeds give a modest average difference, sign reversals versus fixed, wide exploratory paired intervals and worse mean tumor recall than standard.

**What does macro-F1 mean?** Compute F1 for each tissue class and average equally, so frequent/easier classes do not dominate the summary.

**What is the main data limitation?** Patient/slide mapping is unavailable. Exact duplicates are controlled, but patient independence and complete near-duplicate freedom are not demonstrated.

**Did you tune on test data?** No test inference was run in this campaign. Validation selected checkpoints and informed exploratory development; final confirmation requires a locked design.

**Why preserve the residual?** Stain decomposition does not reconstruct every image component. Retaining the reconstruction residual avoids deliberately discarding that information; its benefit still needs comparison.

**Does caching freeze randomness?** No. Source decompositions/diagnostics are cached; target stains, intensity and geometry remain random by image/epoch.

**How reliable are interrupted runs?** Full state is resumable and best-checkpoint reload checks passed. Recovery logs are retained. These checks do not prove bit-identical uninterrupted optimization.

**Is the demo the research model?** No. It identifies its prototype-baseline source; research weights were not activated.

**Is the weighted dashboard percentage a clinical risk?** No. It is an uncalibrated heuristic, not diagnosis or tumor area.

**Is the report final and signed?** It follows the supplied KIIT template. Guide certification signatures and an authentic Turnitin attachment remain pending. No similarity score is promised.

**Is the IEEE paper published or on Overleaf?** It is a locally compiled draft with an uploadable source ZIP. No authenticated private Overleaf import or real link has been completed.

**What next?** Review class failures, obtain grouped mapping/external data, fix the evaluation protocol and related-work position before further experiments.
''',encoding='utf-8')
 (package/'DEMO_CHECKLIST.md').write_text('''# Demo checklist

1. Keep completed research artifacts intact. Do not start another campaign or activate research weights.
2. Run `.venv-research/Scripts/python.exe stain_research/demo_serve.py` from the project root; open http://127.0.0.1:8000.
3. The unchanged application uses isolated demo SQLite/uploads and CPU. Original database and GPU research checkpoints stay protected.
4. Identify the actual model source. Rehearsal used prototype-baseline; do not call its predictions research results.
5. Upload a TRAIN example, never a reserved test image. Show all eight probabilities and the added prediction-history row.
6. Explain FastAPI, the CNN/BiLSTM/attention architecture, the controller and SQL prediction_records storage.
7. Show the three-seed summary, tumor-recall regression, negative pilot and duplicate audit. Separate preprocessing, classifier and notebook metrics.
8. Use the offline slide/report PDFs if the demo fails. Do not fabricate a live result.
9. Stop the demo with Ctrl+C. No additional training is authorized by this checklist.
''')
 (package/'START_HERE.md').write_text('''# Completed research evaluation package

All 12 bounded runs completed on 13 November 2026. No new training is scheduled.
The teacher-format report, 15-slide deck, script, viva, demo checklist and IEEE
draft use the audited final three-seed comparison. Read final_results/FINAL_RESULTS.md.
Accuracy and F1 are validation metrics, not independent final test performance.

Student: Amlan Choudhury, roll23051241, B.Tech CSE. Guide: Dr.Asif Uddin Khan.
Year supplied:2027; no academic-year range was inferred.
The report follows templates/Project_Report_Format.docx. Certification signatures
and a genuine institutional Turnitin report remain pending. No score is promised.

- Progress_Report.docx and .pdf: editable teacher-format report and print copy.
- Progress_Presentation.pptx and .pdf: editable slides and offline backup.
- IEEE_Paper_Draft.pdf and Overleaf_Paper_Source.zip: compiled draft and source.
- Presentation_Script_and_Viva.md and DEMO_CHECKLIST.md: solo preparation.
- final_results/: means, SDs, paired differences, learning curves and confusion matrices.
- DELIVERY_STATUS.json: actual cloud publication links and remaining requirements.

Adaptive mean F1 is0.6836 versus standard0.6776 and fixed0.6822. The gain is modest;
adaptive has lower mean tumor recall than standard. Superiority, novelty, patient
independence and clinical benefit remain unconfirmed. Final test images stay reserved.

The preserved 08:59 snapshot and OVERNIGHT_REVIEW_20261113.md are historical records,
not the final campaign summary. The progress evaluation notice listed13November11AM
B-305; final evaluation listed19November11AM, Asia/Calcutta, final venue unverified.
Overleaf requires authenticated import; upload the ZIP privately and compile main.tex.
Do not invent a project URL. Source/class license notices are preserved.
''')
 (package/'OVERLEAF_STATUS.md').write_text('No authenticated Overleaf session was available at the recorded inspection. No private project or real link is claimed. The updated IEEE ZIP is uploadable and local compilation is checked in IEEE_LOCAL_BUILD.json. Sign in, upload the ZIP as a private project, select main.tex and pdfLaTeX, compile, then record the real project URL. Verified author: Amlan Choudhury; no unverified email or guide coauthorship is inserted.\n')
