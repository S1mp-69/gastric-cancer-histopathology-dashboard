# Publication package — 2026-11-17T06:17:12+05:30

Use Progress_Report.docx/PDF (prescribed teacher structure), Progress_Presentation.pptx/PDF (editable), IEEE_Paper_Draft.pdf, and Overleaf_Paper_Source.zip.

## Verified finding
Adaptive validation macro-F1 is below global shuffled in both original three-seed studies: HMU 0.6836 vs 0.6890; HiESD 0.7131 vs 0.7350. This supports a controlled empirical evaluation, not an accuracy-superiority or guaranteed-novelty claim.

HiESD completed runs: 16/18. Within-class control: 1 of three seeds complete; seed 43 active.

The final-test protocol was frozen before reserved-pixel acquisition. It evaluates all six arms and seeds 42, 43 and 44 using validation-selected best checkpoints, with adaptive-minus-within-class-shuffled patch macro-F1 as the primary contrast. The seven reserved patients cover 13 source slides (about 7.53 GiB). Model hashes are frozen before acquisition. No tuning follows test access. Current stage: waiting for within-class controls. No final-test scores are available in this snapshot.

## Evidence
publication_results/PUBLICATION_SNAPSHOT.json; publication_readiness/publication_audit_20261117/AUDIT.json; publication_readiness/ranking_audit_20261117; repository stain_research/results. The frozen final-test protocol lives in stain_research/runs/final_test_hiesd_v1 and a portable copy accompanies this package. No post-test tuning.

## Remaining human requirements
Read and verify the manuscript; obtain guide critique and approval; confirm ethics/funding/conflict declarations, author email and appropriate authorship; attach genuine institutional Turnitin reports; create a private authenticated Overleaf project and record its real link. Teacher thresholds are similarity below 10% and AI score below 20%; neither is guaranteed. No journal submission, payment or external message has been sent.

## Local continuation without ChatGPT polling
The bounded within-class queue continues the three seed controls, then the frozen final-test pipeline and local artifact refresh. Supervisor: .sync-backups/hiesd_progress/run_within_publication.ps1. If interrupted, rerun that same supervisor; it resumes saved state and skips completed runs. Do not launch a second trainer or search new hyperparameters after reserved-test access.
