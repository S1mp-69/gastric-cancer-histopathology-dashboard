# Solo presentation script and viva

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
