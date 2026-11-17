# Other-PC handoff — 17 November 2026

Use GitHub repository `S1mp-69/gastric-cancer-histopathology-dashboard`, branch
`research/local-backup-2026-11-13`. This handoff takes precedence over older
in-progress queue descriptions: **all 33 matched training runs and the frozen
18-model final-test evaluation are complete. No new training is required.**

## Safely get the current source

In your existing checkout, inspect `git status` first. Preserve any local work
in a commit or stash before switching or pulling; do not overwrite it.

```powershell
git fetch origin
git switch research/local-backup-2026-11-13
git pull --ff-only origin research/local-backup-2026-11-13
git rev-parse HEAD
```

If a fast-forward fails, retain both versions and review their differences.
Do not use a hard reset. Matching folder names or Codex accounts do not sync files.
Read `evaluation_package/START_HERE.md`, `evaluation_package/FINAL_TEST_RESULTS.md`,
and `stain_research/REPRODUCE_PUBLICATION.md` next. The new transfer receipt under
`evaluation_package/OTHER_PC_TRANSFER.json` records the actual Drive files.

## What is complete and what the result means

- HMU: 15 matched runs; image-level exact-duplicate controls, no patient mapping.
- HiESD: 18 matched runs, including three seeds of within-class shuffled control.
- Frozen final test: 1,613 patches, seven reserved patients, all 18 models.
- Adaptive minus within-class shuffled test macro-F1: -0.01466, descriptive
  patient-cluster interval [-0.03757, +0.00692]. Adaptive superiority and clinical
  benefit are unconfirmed. Small patient counts and rare-class support limit inference.
- No post-test tuning, no new claim of novelty, and no research checkpoint activation
  in the original dashboard. Saved prototype demo results are not ResNet validation.

## Transfer files and environment

Git contains source, portable numerical evidence, editable report/PPT and IEEE
source. Drive contains the evaluation ZIP and a dated transfer extras archive
with the prescribed teacher DOCX template, template-format notes, and local
experiment metadata. Separate checkpoint ZIP parts contain model files omitted
from Git. Consult the transfer receipt for which parts actually uploaded; do not
infer that a local archive is on Drive without its successful upload receipt.

**Transfer limitation:** Drive returned `storageQuotaExceeded` after 35 of 64
checkpoint pieces, comprising complete archives 01–07 of 13 (3,096,066,871 bytes).
Archives 08–13 remain local; 29 pieces have not uploaded. Every checkpoint and
all 13 complete ZIP archives are preserved on this PC. Free Drive space or use
a direct external-drive/network transfer later to finish this optional model
backup. The paper, reports and saved numerical verification do not need weights.

- [Transfer extras: teacher template, metadata and evaluation ZIP](https://drive.google.com/file/d/124p_vWLVJ1dTjKUPnHHEXGy7WBB2Y0hY/view)
- [Checkpoint backup folder: partial, archives 01–07 complete](https://drive.google.com/drive/folders/1CE8IpFcy4JxIRdwT4EQAhLG2QF03B0H8)
- [Standalone complete evaluation ZIP](https://drive.google.com/file/d/1LVJ_DmkzrxDd2ku35YurnpTQF-iG2fL8/view)

Download checkpoint pieces together with the restore instructions/manifest ZIP
from that folder. Extract the instructions beside the pieces, then use:

```powershell
py -3.12 restore_checkpoint_transfer.py DOWNLOAD_FOLDER --available-only
```

This reconstructs only complete archives into a new `reassembled` subfolder,
verifying both chunk and ZIP SHA-256 hashes. It never overwrites an existing ZIP.
The transfer receipt lists the missing pieces; do not treat this as a full backup.

Extract extras into a staging folder first. Compare existing files before copying
selected files into the checkout. Never overwrite a destination database/model
pointer or activate research weights as part of transfer. Each archive contains a
SHA-256 file manifest. Checkpoint paths are relative to the project root; their
original configs may contain this PC's absolute paths, which must be adapted
locally. Keep frozen protocol and checkpoint hashes intact.

Do not copy `.venv*`. The other laptop has 16 GB RAM and Intel Iris integrated
graphics, so use it for writing, source inspection and CPU numerical verification.
Create a fresh Python 3.12 environment if needed. The portable verifier requires
NumPy, no CUDA, images, checkpoints or training:

```powershell
py -3.12 -m venv .venv-review
.venv-review\Scripts\python.exe -m pip install numpy
.venv-review\Scripts\python.exe evaluation_package/publication_results/verify_portable_test.py
```

Dataset pixels, rebuildable caches, copied environments and local databases are
not included in this portable transfer. They are preserved on the NVIDIA PC.
Dataset provenance and acquisition references are in `REPRODUCE_PUBLICATION.md`.
Do not redownload/retrain merely to edit the paper or verify saved predictions.

## Next publication work

1. Review the current paper/report with Dr. Asif Uddin Khan; verify authorship,
   contact details, ethics/funding/conflict declarations and certificate signatures.
2. Import `evaluation_package/Overleaf_Paper_Source.zip` into a private Overleaf
   project and record its real URL. You reported an authenticated Codex in-app
   browser session; this chat had no attached browser controls, so import and the
   real project link remain pending. Do not invent a link or claim this is done.
3. Obtain genuine institutional Turnitin reports. Required thresholds were below
   10% similarity and below 20% AI score; no score or acceptance is guaranteed.
4. Present both positive and negative results and the limits of seven patient
   clusters. Keep final-test results out of subsequent model selection.

Use one active source editor at a time, or separate branches for simultaneous
work. Commit and push portable changes before switching PCs, then safely pull.
Ignored data/checkpoints only synchronize when separately transferred and verified.
