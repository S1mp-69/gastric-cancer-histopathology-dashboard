# Morning research status

Updated 2026-11-13T08:59:00+05:30 (Asia/Calcutta).

Progress evaluation: 13 November 2026, 11:00 AM, B-305.
Final evaluation: 19 November 2026, 11:00 AM. Final venue has not been independently verified.

Campaign phase: running. Active run: seed43_standard. Supervisor PID: 5752.
The 09:00 snapshot is a reporting milestone, not a training stop time.

## Actual validation progress

| Arm | Seed | State | Current/completed epoch | Best epoch | Best validation accuracy | Best macro-F1 |
|---|---:|---|---:|---:|---:|---:|
| standard | 42 | complete | 40 | 39 | 0.6865 | 0.6867 |
| residual_fixed | 42 | complete | 34 | 27 | 0.6672 | 0.6661 |
| uniform_weaker | 42 | complete | 40 | 38 | 0.6780 | 0.6760 |
| adaptive | 42 | complete | 40 | 36 | 0.6901 | 0.6888 |
| standard | 43 | training | 31 | 29 | 0.6803 | 0.6800 |
| residual_fixed | 43 | queued | 0 | - | - | - |
| uniform_weaker | 43 | queued | 0 | - | - | - |
| adaptive | 43 | queued | 0 | - | - | - |
| standard | 44 | queued | 0 | - | - | - |
| residual_fixed | 44 | queued | 0 | - | - | - |
| uniform_weaker | 44 | queued | 0 | - | - | - |
| adaptive | 44 | queued | 0 | - | - | - |

These are best-so-far validation selections for active runs. Completion and convergence are separate from epoch count. No final test inference has run.

## Class-specific validation errors

standard seed 42 validation false-negative counts by true class: {"ADI": 98, "DEB": 95, "LYM": 67, "MUC": 129, "MUS": 110, "NOR": 200, "STR": 180, "TUM": 80}
residual_fixed seed 42 validation false-negative counts by true class: {"ADI": 91, "DEB": 127, "LYM": 60, "MUC": 131, "MUS": 123, "NOR": 202, "STR": 185, "TUM": 99}
uniform_weaker seed 42 validation false-negative counts by true class: {"ADI": 88, "DEB": 130, "LYM": 54, "MUC": 133, "MUS": 94, "NOR": 200, "STR": 190, "TUM": 96}
adaptive seed 42 validation false-negative counts by true class: {"ADI": 76, "DEB": 150, "LYM": 71, "MUC": 119, "MUS": 83, "NOR": 195, "STR": 157, "TUM": 97}
standard seed 43 validation false-negative counts by true class: {"ADI": 93, "DEB": 144, "LYM": 51, "MUC": 131, "MUS": 117, "NOR": 186, "STR": 189, "TUM": 67}

## Active run and remaining time

Latest worker progress: {"status": "training", "epoch": 31, "items_in_epoch": 64, "total_train_images": 23761, "updated_at": "2026-11-13T08:58:51+05:30"}
Observed/benchmark-based epoch estimate: 0.95 minutes. Remaining to the 40-epoch ceiling: 9.5 minutes; early stopping can finish earlier.
Current per-class extraction fallbacks: {"ADI": 0, "DEB": 0, "LYM": 0, "MUC": 0, "MUS": 0, "NOR": 0, "STR": 0, "TUM": 0}
Completed runs: 4/12. Full remaining-queue upper-budget estimate: 10.18 hours, refined as epochs complete. Replications start only where the 24-hour campaign budget permits.

## Compute and limits

Selected batch size 64, loader workers 4, AMP=True. Main CPU threads 2, each worker 1.
GPU sample (utilization %, used MiB, temperature C, power W): 88, 2191, 60.44, 49.50
Versioned float64 decompositions and diagnostics are cached. Random targets, intensity factors and geometry remain stochastic by epoch and image.
Candidate split: 23,761 training / 3,059 validation / 3,090 held-out test files. Exact-file duplicates are controlled. Patient/slide mapping and complete near-duplicate freedom remain unverified.
Historical three-epoch pilot: adaptive macro-F1 0.2651, uniform weaker 0.2802, shuffled 0.2768. Those values are separate from this campaign.
Preprocessing stability results measure synthetic-noise response, not classification accuracy. Historical notebook metrics are a separate source.

## Evaluation deliverables

Package ready marker: True.
- evaluation_package/Progress_Presentation.pptx
- evaluation_package/Progress_Presentation.pdf
- evaluation_package/Progress_Report.docx
- evaluation_package/Progress_Report.pdf
- evaluation_package/Overleaf_Paper_Source.zip
- evaluation_package/Presentation_Script_and_Viva.md
- evaluation_package/DEMO_CHECKLIST.md
- evaluation_package/OVERLEAF_STATUS.md
The report format is provisional because no prescribed institutional template was provided. Only the supplied short author name Amlan is verified.

## Resuming after interruption

Supervisor logs, campaign_state.json, each run/history.json, last.pt and best.pt remain on disk. The last checkpoint includes optimizer, scheduler, scaler, RNG states and mid-epoch position.
Restart command: `.venv-research/Scripts/python.exe -u stain_research/overnight_supervisor.py --campaign stain_research/runs/overnight_20261113`.
A live-supervisor lock prevents launching a second supervisor. Original application, database, data and model activation are not written by this campaign.
