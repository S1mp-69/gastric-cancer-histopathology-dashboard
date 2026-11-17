# Fixed HiESD expansion and pretrained baseline (v3)

The user authorized expansion, consistent quality screening, and one pretrained
ResNet-18 baseline. Original applications, databases, raw data, earlier manifests,
reports and checkpoints remain in place. No reserved-test slide or mask is opened.

Frozen input split SHA256:
243cc41a2952930adfdaf178334d7213a858eee7839cae3004caef5a94e45b3c.
Release revision: 89878d1841ef7ca3830ba23f7844bac4cc0cdb7a.

The immutable acquisition plan is data/external/hiesd_expanded_v3/PLAN.json and a
portable copy is under stain_research/results/hiesd_expanded_v3. It selects 24
additional slides totaling 10.79 GiB, below the 12 GiB bound. Selection uses released
train/validation annotation counts and patient coverage, never classifier outputs.
This remains an annotation-guided selected cohort, not unbiased external validation.
Coverage targets are aspirations, not guaranteed achieved counts. The frozen
validation partition has lymphoid annotations in only two patients; no patient
assignments or test membership are changed to address that limitation.

All included old and new slides receive the same rules: nominal downsample64
annotation dimensions must match the level-zero extent within two mask pixels;
released quality mask aspect ratio must agree with the full-slide aspect ratio
within 0.5%. Coordinates in the released quality mask use normalized full-slide
extent, not guessed offsets. This is a consistency check, not landmark registration.
The quality release description defines label 1 tissue, 2-6 artifacts, 7 background.
Unrecognized values (including zero) count as unknown. Patches pass only with <=5%
artifact pixels, <=5% unknown, >=80% quality-mask tissue, and >=90% grouped
annotation purity. Fractions use native quality-mask pixels, not resized colors.
Verified masks are loaded one at a time with a one-billion-pixel bound.
Misaligned/unsupported slides are excluded and recorded; no replacement downloads
are added beyond the frozen plan. A low-quality whole slide can reduce coverage.
Automated masks are proxies, not expert assessment or proof of diagnostic quality.

Patch geometry remains 896 level-zero pixels resized to 224. After QC, retain at
most 128 patches per patient/class in deterministic SHA256 coordinate order. This
cap prevents a single patient's many slides from dominating the sampled class.
Old patches remain preserved in old directories; new QC-screened patches and
manifest go in hiesd_expanded_v3. Before training, assert no patient or exact-patch
SHA256 overlap across partitions, and require every class to have >=16 train and
>=4 validation patches from at least two patients per partition. If this minimum
engineering gate fails, stop and write FAILURE.json; do not loosen QC or reshuffle.

One standard pretrained baseline starts only after those checks. ResNet-18 uses
torchvision ImageNet1K_V1 weights downloaded from the official PyTorch URL and
verified by torchvision's weight-hash check. Full cached-file SHA256:
f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec.
All layers are fine-tuned; five-output head starts fresh, seed42. Fixed ImageNet
RGB normalization; same specified random flips/right-angle rotations and modest
color jitter, sampled per image/epoch. No image stochastic augmentation is cached.
Batch16, two CPU threads, zero loader workers, FP16 AMP, AdamW lr1e-4/wd1e-4,
label smoothing .05, clip5; validation-NLL plateau scheduler factor.5/patience2.
Up to forty epochs with macro-F1 early stopping patience7/min-delta.0001. Latest
and best full checkpoints save every epoch. The disposable benchmark is discarded
before the run reloads initial.pt. Preflight must leave >=768 MiB free VRAM.

A three-update engineering smoke passed on the RTX3050 with 2838 MiB free VRAM.
This measures execution only. Actual expanded-dataset epoch/40-epoch estimates
are saved in the run benchmark.json before training; they cannot be known from
current small-patch smoke alone.

The cohort, backbone, initialization, normalization and QC differ from the old
pilot. Its score cannot isolate a backbone/QC benefit relative to those old runs.
Treat this as a new reference baseline. No statistical/clinical/novelty claims,
method selection on reserved test data, or automatic augmentation comparisons.

The independent pipeline runs under a named-mutex supervisor, keeps the machine
awake without altering clock rates/thermal protection, and saves logs under
.sync-backups/hiesd_progress. The local progress window calls no ChatGPT APIs.
Partial downloads resume, existing file integrity is verified, and run checkpoints
retain optimizer/scheduler/scaler/RNG state. An interruption can be investigated
and resumed using hiesd_expand_v3.py; a completed pipeline exits without rerunning.
A single training epoch is the maximum lost training work on interruption.

Completion outputs:
- data/external/hiesd_expanded_v3/QC receipts, manifest, status/failure records.
- stain_research/results/hiesd_expanded_v3/QC_REPORT.json and PATCH_QC.jpg.
- stain_research/runs/hiesd_resnet18_seed42_v3 latest/best state and predictions.
- stain_research/results/hiesd_resnet18_seed42_v3 metrics, curves and confusion matrix.

Training ends after this one baseline. User can request a review when the local
window says complete or needs attention; no continuous ChatGPT monitoring occurs.
