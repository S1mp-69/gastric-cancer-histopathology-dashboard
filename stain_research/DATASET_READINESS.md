# Dataset readiness — 13 November 2026

The newer local source and portable pilot evidence are backed up on the GitHub
branch `research/local-backup-2026-11-13`. The dashboard source, database, original
checkpoints, raw images and existing experiment outputs remain intact.

## What the provenance audit established

The [Figshare release](https://api.figshare.com/v2/articles/25954813) lists two
files: HMU-GC-HE-30K.zip and HMU-GC-Clinical.csv. The local ZIP has exactly the
published MD5 (`0d55aa70ac70cb237402bc035bbd2355`) and byte size (3,251,341,572),
so this is the original published archive, rather than an incomplete extraction.
Its directory contains 31,096 PNG files and no metadata tables or whole-slide
files. Header inspection found no embedded metadata in any of the PNGs. No test
pixels were decoded and no test inference was performed.

The local clinical CSV has 300 distinct Patient values, but no image, slide or
patch linkage column. Class-number filenames such as ADI_1.png do not identify
patients or slides. Their numeric suffixes must not be used as group labels.

The [dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11754904/) describes
patches derived from 300 whole-slide images and links the authors' code. Its
Data Records section discusses slide and patch data, but the downloadable
archive inspected here contains patches only. This discrepancy needs clarification
from the authors. It does not supply a verified mapping for the present files.

The [authors' repository](https://github.com/TomHardy1997/GCTC) has a dataloader
that expects path/class tables and references a splits/independent_test_set.csv.
The inspected public repository file listing does not contain that split table
or a patch-to-slide/patient mapping. No mapping was found in these checked sources;
this is not a claim that none can exist elsewhere or be obtained from the authors.

The paper's supplementary DOCX could not be inspected in this session: the PMC
download returned an HTML page, and the publisher's alternate host could not be
resolved. This access limit is recorded in the local supplementary_access.json;
the supplement remains an unchecked resource, not evidence that no mapping exists.

## Provisional duplicate-controlled candidate

`prepare_audited_split.py` verified the original image byte hashes and saved a
separate manifest under `runs/split_audit_20261113/`. It retained one representative
of each exact-file hash group and excluded 1,186 redundant file copies, leaving
29,910 unique-file images:

| Split | Retained | Excluded duplicate copies |
|---|---:|---:|
| Training | 23,761 | 1,115 |
| Validation | 3,059 | 51 |
| Test | 3,090 | 20 |

Within each duplicate group, an original test representative takes priority,
then validation, then training. Selection within a split is lexicographic. No
retained image changed its historical split assignment; no original test image
entered development. Conflicting-label duplicate groups would be excluded
entirely; none were present. Assertions verified zero remaining exact-file-hash
duplication, complete retained/excluded accounting, and unchanged assignments.

This is an **image-level candidate**, not a verified patient-independent split.
Near-duplicates and different PNG encodings of the same test pixels have not
been comprehensively screened. Exact-file deduplication alone does not establish
absence of leakage or an independent final confirmation sample. The old manifests
and test partition have been preserved as historical evidence.

## What is needed before convergence training

Obtain a verified table linking each class/filename to an anonymous patient ID
and slide ID, preferably including original patch coordinates. The local
`patient_slide_mapping_to_complete.csv` contains the expected image identities
with blank group fields. No patient group identities have been invented.

Once the mapping is available, validate complete coverage, duplicate consistency,
patient/slide hierarchy and provenance, then build a new frozen grouped evaluation
design. The earlier image-level exploration must be disclosed; a newly grouped
split of already explored data is not automatically an independent replication.
Use an external, unexposed patient cohort for stronger final confirmation when
available.

An isolated baseline protocol is saved as `baseline_plan.json`: existing
CNN-BiLSTM-attention, standard augmentation, seed 42, batch 16, AdamW, up to
40 epochs with early stopping, training-only statistics and deterministic
validation. **It has not been launched.** The current pilot runner retains its
five-epoch cap. After establishing a useful baseline, compare standard, uniform
weaker and adaptive augmentation with equal initialization and budgets.

The three-epoch pilot remains inconclusive: adaptive macro-F1 was 0.2651 versus
0.2802 for the weaker control and 0.2768 for shuffled strengths. No accuracy,
clinical benefit or novelty claim is justified. Do not expand automatically.

## Reproduce the audits

Choose fresh output directories; existing completed audit reports refuse overwrite.

```powershell
$python = '.\.venv-research\Scripts\python.exe'
& $python stain_research/prepare_audited_split.py --audit stain_research/runs/classifier_local_20261113 --output stain_research/runs/split_audit_NEW
```

`audit_dataset_provenance.py` expects the saved public Figshare API response in
its output directory. The original response and local evidence are saved in
`runs/provenance_audit_20261113/`. Public aggregate results are under `results/`;
full working manifests, patient-linkage templates and local data stay on this PC.

An unsent request for the missing mapping is in `PATIENT_MAPPING_REQUEST.md`.
