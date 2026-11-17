# HiESD acquisition and engineering pilot

The prospective patient split in `data/external/hiesd_metadata/prospective_grouped_manifest.csv`
is frozen (SHA256 `243cc41a2952930adfdaf178334d7213a858eee7839cae3004caef5a94e45b3c`).
No patient assignments are changed and reserved-test slides or annotation masks
are not acquired or inspected by these scripts.

## Version 1

`hiesd_pilot.py` acquired six slides chosen by file size and distinct patient
identity: four train and two validation. Five passed the nominal mask alignment
check and yielded 308 patches. The third slide was excluded because its irregular
native pyramid does not establish a reliable nominal downsample64 mapping to
level-zero coordinates. Sources and failure history are preserved.

This subset had no tumor or lymphoid patches. No classifier was trained.

## Version 2: annotation-guided coverage amendment

`hiesd_expand_pilot.py` examines published annotation masks only within the frozen
train/validation partitions, pinned to Hugging Face revision
`89878d1841ef7ca3830ba23f7844bac4cc0cdb7a`. Verified mask LFS SHA256 checksums and
published raw-slide MD5/byte sizes are used. Original v1 data and manifests remain
unchanged; a new combined patch manifest is written under
`data/external/hiesd_pilot_v2/`.

Selection seeks missing tissue groups and distinct patients, preferring lower
download cost. Targets are 32 train and 16 validation patches per class, with
two patients per class in each partition. An absolute bound of 8 GiB and twelve
additional slides applies. The frozen first v2 plan selects seven slides totaling
about 4.12 GiB. It predicts all five groups but does **not** meet the validation
lymphoid target: only twelve eligible patches from two patients are available in
the examined validation annotations. Do not silently loosen purity, change
patient assignments, or describe targets as met.

This deliberately selected subset is biased and cannot stand in for an unbiased
held-out cohort. Validation annotations informed coverage selection; selection is
not based on model performance. It is an engineering feasibility study, not a
confirmatory validation protocol or external validation result.

Patches use non-overlapping 896-by-896 level-zero regions resized to 224-by-224,
nominally 10x for a 40x source. A 14-by-14 downsample64 annotation window must have
at least 90% pixels in one grouped class. At most 128 patches per slide per class
are retained. These coarse boundaries and nominal coordinate checks are not expert
verification of tissue labels. Misaligned slides are explicitly excluded without
stretching or inferred cropping. Additional slides are not substituted automatically
after acquisition failures or alignment exclusions.

The current provisional group mapping is tumor (tub1/tub2/pap),
mucosa/inflammation (normal gland/chronic gastritis/atrophic gastritis), intestinal
metaplasia (complete/incomplete), lymphoid, and the released "Others" category.
The meaning of "Others" and equivalence to the published five-class task need
source verification before scientific comparisons. This is distinct from HMU's
eight-class task.

## Local execution and review gate

Use this PC's `.venv-research`, with `openslide-python==1.4.7`,
`openslide-bin==4.0.1.2`, and `requests==2.34.2` installed in addition to existing
NumPy/Pillow. Network retries and partial-file resume are bounded; source files
are integrity checked before extraction. Keep at least 15 GiB disk headroom beyond
planned slide bytes. No dashboard, database or existing checkpoints are changed.

The independent Python process writes `STATUS.json`, `receipts.json`, logs,
`patch_manifest.csv`, and `PATCH_QC.jpg`. Its local Windows progress display reads
these files without calling ChatGPT. Closing that display does not stop downloads.

No automatic classifier training is included. Review actual class/patient counts,
alignment exclusions, mask semantics and images after download, then benchmark a
short GPU baseline before any expanded comparison. A short smoke run would verify
execution only and must not be described as convergence or a novelty result.
