# Reproducing the publication study

Use the existing research branch. Original application/data/checkpoints are preserved; the demo and research output locations are isolated. Do not copy a virtual environment between machines. Source and portable results synchronize through Git; data, environments, caches and large checkpoints remain local. Record `git rev-parse HEAD` on both machines and inspect dirty files before concluding they are synchronized.

## Environment and data

The working local `.venv-research` was created on this PC with Python 3.12.14. PyTorch 2.7.0+cu128 / torchvision 0.22.0+cu128 run on the existing RTX 3050 Laptop GPU; the known workload uses batch 16, loader workers 0, two CPU threads and FP16. Reuse that verified environment on this PC. On a new machine, create a fresh environment, install the pinned base requirements and the matching official PyTorch build, plus requests, openslide-python and openslide-bin. Consult `protocols/publication_20261117/ENVIRONMENT_FREEZE.txt` for the observed installed packages, not a guarantee every bundled authoring package is needed for training. Do not expect CUDA on the Iris-only laptop.

HMU source: doi:10.1038/s41597-025-04489-9. HiESD source: doi:10.1038/s41597-025-05679-1 and Figshare doi:10.6084/m9.figshare.28919840. Obtain images from the publishers' releases under their terms. Dataset pixels are excluded from this repository. Patient allocation, exact development patch coordinates/QC, final-test protocol and checksums are exported under `protocols/publication_20261117`. Existing acquisition scripts retain official raw checksums and pinned mask revision. Do not redownload a different revision silently or change sampling to obtain preferable results.

## Completed experiments and added control

`hiesd_ablation_train.py` and `hiesd_ablation_data.py` are frozen original experiment sources. `hiesd_within_class_train.py` wraps those sources to permute training strengths within class; it leaves source decompositions, targets, geometry, initialization, optimizer and stopping unchanged. `within_class_publication_queue.py` checks the 120 epoch/seed multisets and runs seeds 42/43/44 sequentially, maximum 40 epochs, patience seven. Existing RESULT files are skipped, last checkpoints resume.

To resume on this PC, run `powershell -NoProfile -ExecutionPolicy Bypass -File stain_research/run_publication.ps1`. Its named mutex rejects a duplicate supervisor. Do not bypass that guard by launching individual trainers while the queue is active. CPU-only `publication_campaign_audit.py` verifies the original comparisons and `publication_control_audit.py` verifies the added ones. No repeated GPU check is needed.

## Frozen final test

The final protocol is byte-hashed before reserved pixels. `publication_finalize_pipeline.py` requires all three controls complete, freezes every best-checkpoint SHA, acquires only the 13 preselected reserved slides, applies unchanged QC/sampling, rejects exact development/test collisions, and evaluates all six arms × three seeds. Missing classes or integrity failures stop the evaluation; do not fix them by changing the cohort/model after seeing test outcomes. Saved predictions support restart without tuning. The primary contrast and cluster-bootstrap conventions are in the frozen protocol. This public-release patient holdout is not independent clinical external validation.

## Results and materials

`results` contains portable numeric summaries, histories and saved validation predictions. `export_publication_evidence.py` exports metadata and observed timing without pixels/weights. `refresh_publication_package.py` rebuilds the ONE `evaluation_package` from those saved outputs. DOCX uses the retained private teacher template; PDFs use native Microsoft Word/PowerPoint, and the IEEE source is compiled with the verified local Tectonic executable. Editable PowerPoint uses the Codex bundled artifact runtime. These authoring dependencies are machine-specific and documented; source/results can be inspected without them. Every changed export needs final visual review.

`PUBLICATION_READINESS.md`, `MENTOR_REVIEW.md` and the solo script record remaining human requirements. An uploadable Overleaf ZIP is not evidence of an authenticated project link. Genuine Turnitin reports, guide signatures and publication acceptance cannot be generated or promised by these scripts.
