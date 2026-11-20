# Research audit 16 November 2026

Verdict: the saved15-run experiment is internally consistent on the checks below.
The current submission documents are stale and the study does not establish
diagnostic-assignment superiority, publication novelty or clinical benefit.
No training or test-pixel decoding was performed by this audit.

## Verified experiment outputs

- All15 metrics files were recomputed from saved validation probabilities: accuracy,
  macro-F1, class precision/recall/F1/support, confusion matrices, ECE and Brier score.
- Same ordered3059 validation image identities across every run; same23761 training
  and3059 validation records in the cache and frozen split manifest.
- No exact SHA256 overlap between retained train and validation records.
- Initial-weight hashes agree across all five arms within each seed; shared numeric
  hyperparameters and validation policy agree. Best checkpoint hashes agree.
- Best epoch selection and stopping reasons reproduce the recorded rules.
- All best-model tensors were explicitly checked for finite values through the
  model_state_dict. The earlier final_analysis.py implementation skipped this nested
  dictionary; it was corrected. The new independent check passed every model.
- All last checkpoints contain model, optimizer, scheduler, scaler, RNG and position
  keys; all completed runs recorded successful best-model probability reload checks.
  This audit did not independently rerun full inference or prove identical interrupted
  and uninterrupted optimization.
- Training-only diagnostic quantile bounds recomputed exactly from cached features.
- Shuffled strengths preserve the exact valid-image strength multiset for120 tested
  seed/epoch combinations (3seeds x40epochs). Invalid-image fallback remains separate.
- Original preservation audit passed:69 protected artifacts and31096 dataset files,
  with no byte changes. No classifier training/supervisor was active at inspection.

## Actual five arm results

| Arm | Validation accuracy mean | Macro F1 mean ± SD | TUM recall mean |
|---|---:|---:|---:|
| Standard |67.70%|0.6776 ±0.0106|79.97%|
| Residual fixed |68.30%|0.6822 ±0.0141|74.18%|
| Uniform weaker |68.22%|0.6804 ±0.0061|75.56%|
| Adaptive |68.41%|0.6836 ±0.0047|75.56%|
| Shuffled |69.10%|0.6890 ±0.0029|74.53%|

Shuffled exceeds adaptive in all three seeds. Adaptive-minus-shuffled mean F1 is
-0.00544; exploratory95%t interval[-0.01496,+0.00407]. This does not establish
superiority for either method. Three seed replicates on identical validation cases
are not independent patient samples. Shuffled was a post-hoc follow-up; do not
describe all15 runs as a jointly preregistered confirmatory five-arm study.

## Important findings before submission

1. The24-page report,15-slide presentation and4-page IEEE draft were last generated
   on13November and omit the completed16November shuffled control. Statements that
   adaptive has the highest mean performance must be revised. The published Drive
   package was uploaded before this follow-up; its live contents were not recertified
   during this local audit. Update and verify the cloud copies after revising them.
2. Legacy dashboard prototype selection reads the first60 files per class from the
   all-image directory. Filename-only comparison found41 retained reserved-test
   paths among these480 potential exemplars. The scientific classifier training
   workflow does not import/use these prototypes and passed its train/validation
   checks. However, do not claim test images were never accessed anywhere in the
   project. Restrict future demo prototypes to training-only exemplars, or disable
   this fallback, before further demonstrations. No original application was changed.
3. The approved dashboard pointer refers to an existing August checkpoint, not the
   November research models. Current runtime loading was not tested. The last saved
   rehearsal identified prototype-baseline. Do not advertise research inference in
   the dashboard without a separately authorized/verified activation.
4. Patient/slide mapping remains unavailable for the HMU cohort. Exact-file control
   does not remove near duplicates or establish patient independence. HiESD metadata
   now provide a separate grouped cohort, but no HiESD images or results were acquired;
   that work cannot be included as completed external validation.
5. Source diagnostic features are numerical proxies, not pathology quality labels.
   Preserving the strength distribution does not guarantee matched pixel/OD change.
   Original SAN appears in the short pilot only, not the full-budget classifier study.
   Do not imply a full-budget head-to-head superiority result against published SAN.
6. Source records retain restarts and AMP backoffs. Report them; model reload and
   hash checks do not remove this limitation. Current original-run trainer bytes are
   not a guarantee that every historical interrupted attempt used identical code.
7. Overleaf private import/link, guide certification signatures and genuine Turnitin
   reports remain outstanding in the last verified delivery records. The prescribed
   teacher template and verified author details are present. No plagiarism/AI score
   is certified and no publication acceptance or novelty is established.

## Immediate recommendation

Finalize an exploratory methods/negative-result paper asking whether numerical
decomposition diagnostics add classification value beyond matched-strength random
assignment. Include all15 runs, class tradeoffs, the post-hoc design, data limitations
and exact source versions. Revise the IEEE paper/report/PPT together. Create the
real private Overleaf project and obtain the actual institutional similarity reports.
Do not launch additional experiments to seek a favorable ranking under this deadline.

Machine-readable details: audit.json, five_arm_summary.csv,
shuffled_assignment_audit.json. Original run outputs/checkpoints remain unchanged.
