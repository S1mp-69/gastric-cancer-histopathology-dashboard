# Verified evaluation-weight sensitivity package

Independent verification passed for 15 runs x 24 scenarios = 360 weighted confusion matrices. The largest numerical difference was 2.78e-15. All scheme rankings agree with the original analysis.

Main finding: lymphoid precision contributes 69.4% of the observed-to-equal-class adaptive-minus-shuffled contrast shift. Both methods correctly classify the same seven lymphoid cases in every seed; reweighting does not improve their predictions. Macro-F1 is prevalence-sensitive, which is established prior work, so this finding cannot be presented as a newly discovered metric property.

Files:
- AUDIT_REPORT.md: verification and class decomposition.
- all_24_weightings.csv: every specified scenario, macro-F1 and balanced accuracy.
- verified_seed_metrics.csv and verified_class_metrics.csv: detailed reproducible numbers.
- class_reversal_contributions.csv: exact additive explanation of the contrast shift.
- all_weightings.png and class_drivers.png: visually inspected scientific figures.
- LITERATURE_AND_CLAIMS.md: closest papers, supported/unsupported claims and remaining work.
- manuscript_section.tex, all_weightings_table.tex, references.bib: editable IEEE-compatible insert fragments; not a complete compiled manuscript or Overleaf upload.
- VERIFICATION.json: implementation, source hashes and pass evidence.

No training, model inference, GPU checks or new image access occurred. Source predictions are unchanged. The prescribed report, PPT, main IEEE manuscript and cloud copies have NOT been refreshed by this analysis package; their separate update is still outstanding. The sensitivity section should accompany the original negative/mixed study, not replace the original endpoint with a more favorable weighting.
