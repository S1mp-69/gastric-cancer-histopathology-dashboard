# Publication package — 2026-11-17T07:15:02+05:30

Use Progress_Report.docx/PDF (prescribed teacher structure), Progress_Presentation.pptx/PDF (editable), IEEE_Paper_Draft.pdf, and Overleaf_Paper_Source.zip.

## Verified finding
Adaptive validation macro-F1 is below global shuffled in both original three-seed studies: HMU 0.6836 vs 0.6890; HiESD 0.7131 vs 0.7350. This supports a controlled empirical evaluation, not an accuracy-superiority or guaranteed-novelty claim.

HiESD completed runs: 18/18. Within-class control: 3 of three seeds complete.

The frozen final test completed on 1613 patches from 7 reserved patients. The prespecified adaptive-minus-within-class-shuffled mean macro-F1 difference was -0.0147, with descriptive patient-cluster interval [-0.0376, +0.0069]. Of 1,000 cluster draws, 31 lacked at least one class; fixed five-class zero-division-zero scoring was retained. Only 7 usable patient clusters limit precision. All models were frozen before acquisition; no tuning follows test access. This is held-out public-release testing, not independent clinical external validation.

## Evidence
publication_results/PUBLICATION_SNAPSHOT.json; publication_readiness/publication_audit_20261117/AUDIT.json; publication_readiness/ranking_audit_20261117; repository stain_research/results. The frozen final-test protocol lives in stain_research/runs/final_test_hiesd_v1 and a portable copy accompanies this package. No post-test tuning.

## Remaining human requirements
Read and verify the manuscript; obtain guide critique and approval; confirm ethics/funding/conflict declarations, author email and appropriate authorship; attach genuine institutional Turnitin reports; create a private authenticated Overleaf project and record its real link. Teacher thresholds are similarity below 10% and AI score below 20%; neither is guaranteed. No journal submission, payment or external message has been sent.

## Local continuation without ChatGPT polling
The bounded within-class queue continues the three seed controls, then the frozen final-test pipeline and local artifact refresh. Supervisor: .sync-backups/hiesd_progress/run_within_publication.ps1. If interrupted, rerun that same supervisor; it resumes saved state and skips completed runs. Do not launch a second trainer or search new hyperparameters after reserved-test access.
