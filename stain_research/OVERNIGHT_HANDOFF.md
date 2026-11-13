# Overnight campaign and low AI usage

The user authorized the provisional image-level campaign on 13 November 2026.
Use the existing `runs/overnight_20261113` workflow rather than starting duplicates.
Four arms use seeds 42, 43 and 44 where feasible, at most 40 epochs per run with
patience-seven early stopping. No final test inference or dashboard activation.

## Monitoring without ChatGPT calls

`overnight_supervisor.py` is an ordinary local Python process. It monitors the
worker, writes logs/status, restarts from checkpoints, and refreshes the evaluation
files without calling ChatGPT or the OpenAI API. Do not keep an AI agent polling
routine progress. The user explicitly wants to conserve the usage allowance.
Read a compact status only when asked, or when an actual failure needs repair.

Local commands that do not invoke an AI agent:

```powershell
nvidia-smi -l 2
Get-Content .\MORNING_STATUS.md
Get-Content .\stain_research\runs\overnight_20261113\seed42_standard\worker.log -Tail 8 -Wait
```

The GPU performs model forward/backward and optimizer work. CPU loader workers
perform image decoding, geometry and cached stain reconstruction. Intermittent
low GPU utilization during loading or restart is not evidence of CPU model
training. The benchmark selected batch 64 and four workers, with approximately
1.2 GiB CUDA free. Batch 96 was rejected for insufficient headroom.

## Resume and recovery

```powershell
.\.venv-research\Scripts\python.exe -u stain_research/overnight_supervisor.py --campaign stain_research/runs/overnight_20261113
```

A live supervisor lock prevents duplicate launches. Saved `last.pt` files include
optimizer, scheduler, scaler, random states and mid-epoch position. Best checkpoints
are separate. Checkpoint writes use atomic replacement.

At epoch 26 of seed-42 standard training, the original fatal gradient check
interrupted normal GradScaler overflow handling. The corrected code lets AMP skip
the unsafe update and lower its scale, following PyTorch's documented procedure.
Nonfinite loss or an unhandled nonfinite gradient still fails. The resumed worker
completed epoch 26 with one logged skipped AMP step. Failed-attempt logs are kept.

`MORNING_STATUS.md` updates locally. A separate local status thread prepares the
09:00 milestone snapshot within the preceding minute and records the actual write
time. Training continues afterward. The campaign starts no further runs after
its 24-hour scheduling budget, and has at most 12 runs. Final test data stays out
of tuning. Patient independence, clinical benefit and novelty remain unconfirmed.

## Evaluation package

Update the one existing `evaluation_package`, not conflicting copies. Its original
transfer is retained under `.sync-backups`. `refresh_evaluation.py` and
`build_evaluation_deck.mjs` update actual results and use native Office PDF export.
Only the provided short author name Amlan is verified. The report is provisional
because the institutional template was not supplied. Overleaf required sign-in;
no private project or real project link exists. Keep the uploadable IEEE ZIP.

The supervisor holds extra arms until `evaluation_package/PACKAGE_READY.json`
exists. That marker is written only after the initial package QA is complete.
