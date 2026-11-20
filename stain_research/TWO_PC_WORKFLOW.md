# Two computer workflow

Current handoff: read `../OTHER_PC_HANDOFF.md` first. As of 17 November 2026,
all 33 matched training runs and 18 frozen final-test evaluations are complete.
The historical acquisition/12-run status below is superseded by that handoff.
Drive checkpoint backup is partial because storage quota was exceeded; see
`evaluation_package/OTHER_PC_TRANSFER.json` for the exact uploaded/missing pieces.

Use GitHub branch `research/local-backup-2026-11-13` for source and portable reports.
Run `python stain_research/pc_sync_check.py` on each computer. A true
`source_synced` means the branch matches the live remote and there are no visible
local changes. It does not verify ignored artifacts. The script makes no changes.

Before switching computers, commit and push reviewed source changes. On the
destination, inspect local work and use a fast-forward pull only when safe.
If both machines have edits, preserve both on separate branches and review the
merge. Do not use hard reset or a sync tool to overwrite a working directory.
Check both the branch and commit; a shared Codex account or matching folder name
does not prove that project files match. Use one active source editor at a time
or separate branches for simultaneous work.

Large datasets, SQLite databases and checkpoints are private local artifacts.
Use the separately prepared `PRIVATE_ARTIFACT_MANIFEST.json` and run:

```powershell
python stain_research/pc_sync_check.py --verify-manifest PATH_TO_PRIVATE_ARTIFACT_MANIFEST.json
```

Missing or different files must be reviewed. Preserve a destination backup before
copying selected artifacts. This verification never overwrites files. Existing
dataset counts alone are not a byte comparison. Do not publish the private
manifest, model archive or database through GitHub.

Create a fresh environment on each PC; never copy `.venv*`. Absolute model and
dataset paths must be checked locally. A pointer JSON does not prove runtime model
loading; do not activate research weights during transfer. The integrated-Iris
laptop can store/prepare data; use the NVIDIA PC for CUDA classifier training.

The completed HMU study has 12 runs; novelty and superiority remain unconfirmed.
The new HiESD patient-grouped manifest is local under `data/external/hiesd_metadata`.
It records 44 patients,104 slides and308 strips; 30/7/7 patients are allocated to
train/validation/reserved-test. Only metadata have been acquired; different labels
require a separate protocol. Do not run the old eight-class training script on it.
No new training campaign is started by these instructions.
