# Morning status — 17 November 2026

Prepared 2026-11-17T07:15:02+05:30 in Asia/Calcutta, before the 09:00 reporting milestone. This is a status snapshot, not a training stop instruction.

## Completed and active work

- HMU: 15 matched runs complete, five arms × seeds 42/43/44. Exact-file control only; no verified patient mapping.
- HiESD: 18 matched runs complete, six arms × seeds 42/43/44. Added within-class runs stopped at 22, 22 and 27 epochs; selected best epochs 15, 15 and 20. No three-epoch convergence claim.
- Frozen reserved-patient test: all 18 models evaluated on 1,613 patches from seven patients. Independent metrics and patient-bootstrap verification passed. Raw data, databases and checkpoints remain local; research checkpoints are not activated in the dashboard.
- Active training runs: none. Current training epoch: not applicable. Remaining model-training/test time: zero. No additional training is launched after reserved-test access.
- Local artifact generation and export status: PUBLICATION_BUILD.json and its snapshot/hash receipt. Original model artifacts remain in stain_research/runs; source and portable report synchronization is recorded separately in CLOUD_SYNC_STATUS.json.

## Validation and final results

HiESD validation mean macro-F1: standard 0.7270, residual fixed 0.7260, uniform weaker 0.7220, global shuffled 0.7350, adaptive 0.7131, within-class shuffled 0.7063. Adaptive-minus-within-class validation mean is +0.0068 with exploratory n=3 interval [-0.0077,+0.0214].

The frozen final test completed on 1613 patches from 7 reserved patients. The prespecified adaptive-minus-within-class-shuffled mean macro-F1 difference was -0.0147, with descriptive patient-cluster interval [-0.0376, +0.0069]. Of 1,000 cluster draws, 31 lacked at least one class; fixed five-class zero-division-zero scoring was retained. Only 7 usable patient clusters limit precision. All models were frozen before acquisition; no tuning follows test access. This is held-out public-release testing, not independent clinical external validation.

Final-test mean macro-F1: standard 0.6600, residual fixed 0.6696, uniform weaker 0.6495, global shuffled 0.6350, adaptive 0.6602, within-class shuffled 0.6749. All individual seeds and secondary metrics are in FINAL_TEST_RESULTS.md and publication_results/final_test_RESULTS.json. The same cases recur across models and seeds.

## Class failures and limits

Adaptive test recall: tumor 62.24%, mucosa/inflammation 87.15%, intestinal metaplasia 83.25%, lymphoid 100% on only nine patches, other 19.96%. Standard tumor recall is 66.98% and intestinal-metaplasia recall 76.85%. These class tradeoffs and sparse lymphoid support prevent clinical claims. PER_CLASS_RESULTS.md includes every arm. The expert-review CSV is a queue, not completed pathology adjudication or corrected labels.

## Deliverables in this same evaluation_package

Progress_Report.docx / Progress_Report.pdf (prescribed teacher structure); Progress_Presentation.pptx / Progress_Presentation.pdf; IEEE_Paper_Draft.pdf; Overleaf_Paper_Source.zip; Research_Evaluation_Package.zip; SOLO_PRESENTATION_SCRIPT.md; VIVA_ANSWERS.md; DEMO_CHECKLIST.md; CLAIM_EVIDENCE_LEDGER.md; MENTOR_REVIEW.md; VENUE_AND_SUBMISSION_PLAN.md.

Remaining human requirements: guide review/signatures, verified declarations and references, a real private Overleaf project after login, and genuine institutional Turnitin reports. No novelty, acceptance, clinical benefit or detection-score guarantee is made. No journal submission, payment or external message has been sent.
