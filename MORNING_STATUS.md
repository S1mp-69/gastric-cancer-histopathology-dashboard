# Morning research status

Updated 2026-11-13T17:42:29+05:30 (Asia/Calcutta).

Progress evaluation: 13 November 2026, 11:00 AM, B-305.
Final evaluation: 19 November 2026, 11:00 AM. Final venue has not been independently verified.

Campaign phase: bounded_campaign_complete. Active run: none. Supervisor PID: 5752.
The 09:00 snapshot is a reporting milestone, not a training stop time.

## Actual validation progress

| Arm | Seed | State | Current/completed epoch | Best epoch | Best validation accuracy | Best macro-F1 |
|---|---:|---|---:|---:|---:|---:|
| standard | 42 | complete | 40 | 39 | 0.6865 | 0.6867 |
| residual_fixed | 42 | complete | 34 | 27 | 0.6672 | 0.6661 |
| uniform_weaker | 42 | complete | 40 | 38 | 0.6780 | 0.6760 |
| adaptive | 42 | complete | 40 | 36 | 0.6901 | 0.6888 |
| standard | 43 | complete | 36 | 29 | 0.6803 | 0.6800 |
| residual_fixed | 43 | complete | 40 | 40 | 0.6930 | 0.6924 |
| uniform_weaker | 43 | complete | 40 | 38 | 0.6793 | 0.6779 |
| adaptive | 43 | complete | 38 | 31 | 0.6839 | 0.6823 |
| standard | 44 | complete | 39 | 32 | 0.6643 | 0.6660 |
| residual_fixed | 44 | complete | 40 | 36 | 0.6888 | 0.6881 |
| uniform_weaker | 44 | complete | 40 | 40 | 0.6894 | 0.6873 |
| adaptive | 44 | complete | 35 | 28 | 0.6783 | 0.6797 |

These are best-so-far validation selections for active runs. Completion and convergence are separate from epoch count. No final test inference has run.

## Class-specific validation errors

standard seed 42 validation false-negative counts by true class: {"ADI": 98, "DEB": 95, "LYM": 67, "MUC": 129, "MUS": 110, "NOR": 200, "STR": 180, "TUM": 80}
residual_fixed seed 42 validation false-negative counts by true class: {"ADI": 91, "DEB": 127, "LYM": 60, "MUC": 131, "MUS": 123, "NOR": 202, "STR": 185, "TUM": 99}
uniform_weaker seed 42 validation false-negative counts by true class: {"ADI": 88, "DEB": 130, "LYM": 54, "MUC": 133, "MUS": 94, "NOR": 200, "STR": 190, "TUM": 96}
adaptive seed 42 validation false-negative counts by true class: {"ADI": 76, "DEB": 150, "LYM": 71, "MUC": 119, "MUS": 83, "NOR": 195, "STR": 157, "TUM": 97}
standard seed 43 validation false-negative counts by true class: {"ADI": 93, "DEB": 144, "LYM": 51, "MUC": 131, "MUS": 117, "NOR": 186, "STR": 189, "TUM": 67}
residual_fixed seed 43 validation false-negative counts by true class: {"ADI": 73, "DEB": 136, "LYM": 54, "MUC": 135, "MUS": 97, "NOR": 164, "STR": 184, "TUM": 96}
uniform_weaker seed 43 validation false-negative counts by true class: {"ADI": 90, "DEB": 110, "LYM": 58, "MUC": 116, "MUS": 111, "NOR": 195, "STR": 208, "TUM": 93}
adaptive seed 43 validation false-negative counts by true class: {"ADI": 85, "DEB": 139, "LYM": 48, "MUC": 135, "MUS": 98, "NOR": 177, "STR": 186, "TUM": 99}
standard seed 44 validation false-negative counts by true class: {"ADI": 125, "DEB": 160, "LYM": 53, "MUC": 139, "MUS": 115, "NOR": 168, "STR": 182, "TUM": 85}
residual_fixed seed 44 validation false-negative counts by true class: {"ADI": 85, "DEB": 127, "LYM": 55, "MUC": 97, "MUS": 112, "NOR": 201, "STR": 171, "TUM": 104}
uniform_weaker seed 44 validation false-negative counts by true class: {"ADI": 92, "DEB": 98, "LYM": 52, "MUC": 113, "MUS": 105, "NOR": 205, "STR": 191, "TUM": 94}
adaptive seed 44 validation false-negative counts by true class: {"ADI": 93, "DEB": 143, "LYM": 66, "MUC": 135, "MUS": 106, "NOR": 167, "STR": 187, "TUM": 87}

## Active run and remaining time

Completed runs: 12/12. Full remaining-queue upper-budget estimate: 0.00 hours, refined as epochs complete. Replications start only where the 24-hour campaign budget permits.

## Compute and limits

Selected batch size 64, loader workers 4, AMP=True. Main CPU threads 2, each worker 1.
GPU sample (utilization %, used MiB, temperature C, power W): 0, 0, 39.53, 10.27
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
