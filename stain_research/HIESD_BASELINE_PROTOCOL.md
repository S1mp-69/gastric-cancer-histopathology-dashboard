# HiESD single-seed baseline pilot

Run directory: `stain_research/runs/hiesd_baseline_seed42_20261116`.

This isolated run uses seed 42, a fresh five-output CNN-BiLSTM-attention model,
769 training patches and 1042 validation patches. Patient assignments follow the
frozen prospective split; there are five selected train patients and five selected
validation patients, with no overlap. Reserved-test data is not used.

The five broad groups are verified against Figure 1 and Table 2 of the HiESD paper
(DOI 10.1038/s41597-025-05679-1). The paper was retrieved through Europe PMC's
fullTextXML endpoint and retained locally. The released ten-color annotation
mapping is consistent with these groups. The original upstream preprocessing code
was inspected at commit f511e3bf4c47eb7fcdd1042b3d0ffde1137e05b2.
This does not verify individual patch labels or make our different extraction,
model and selected subset directly comparable to the published benchmark.

Training uses RGB channel mean/std fitted only on training patches, random flips,
right-angle rotations, and modest color jitter. Random augmentation is applied
per image and epoch, not cached. RGB patches are preloaded in RAM. Batch size is
32, loader workers zero, main CPU threads two; GPU computation uses FP16 AMP.
No CPU fallback is permitted. All architectural and training choices are saved
in config.json before fitting. initial.pt is saved for later matched-arm runs.

Budget: at most 40 epochs; stop after seven epochs without a validation macro-F1
improvement greater than 0.0001. AdamW lr=.001, weight decay=.0001, label
smoothing=.05, gradient clip=5. ReduceLROnPlateau monitors validation NLL,
patience two, factor .5. Best checkpoint is selected by validation macro-F1.
The latest full training state and best state are saved after every epoch.

A disposable benchmark measured about 122 training images/s with actual standard
augmentation and 1.27 seconds for full validation. The initial estimate is about
7.56 seconds/epoch, or 5.04 minutes for forty epochs, excluding checkpoint,
reporting and other overhead. About 2326 MiB VRAM remained free in the benchmark.
Its model/optimizer state is discarded; the actual run reloads initial.pt.

Files include config.json, benchmark.json, initial.pt, last.pt, best.pt,
history.json, progress.json, validation_predictions.csv, RESULTS.json and logs.
Completed portable outputs and figures are written under
`stain_research/results/hiesd_baseline_seed42_20261116`.
Training stops after this one run; no extra augmentation arms start automatically.

The independent local supervisor prevents duplicate supervisor instances and
keeps the system awake during this bounded run without changing clock rates or
thermal protections. The local progress window does not call ChatGPT. Closing
that window leaves training running. Read logs and FAILURE.json for real failures.
To resume after investigating an interruption:

```powershell
.venv-research/Scripts/python.exe -u stain_research/hiesd_baseline.py --run stain_research/runs/hiesd_baseline_seed42_20261116 --resume
```

Preserve existing checkpoints and logs before retrying. Do not run two trainers
against the same directory. No dashboard models are activated.

Interpretation limits: annotation-guided biased subset; very few patients;
class imbalance, including only twelve validation lymphoid patches; coarse-mask
boundary uncertainty; no expert pathology re-review or GrandQC-mask filtering.
Single-seed validation results do not establish convergence, clinical validity,
controller benefit, novelty or publication readiness. No patient-level confidence
interval is claimed. Inspect learning curves and per-class failures before any
subsequent experiments.
