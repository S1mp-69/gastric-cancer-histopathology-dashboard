# stain-san

Repo for implementing the methodology described in Stain SAN: Simultaneous Augmentation and Normalization.

```
pip install .
```

## Instructions to run the code

### 1. Clone repo

cbcs_joint/Paths.py has instructions for setting up the data directory.

### 2. Install code

Download the github repository using
```
git clone https://github.com/taebinkim7/stain-san.git
```

Using python >= 3.7, install the package stain-san by running
```
pip install -e .
```

### 3. Stain color matrix extraction

Extract stain color matrices and store them in a desired directory by running
```
python scripts/extract_stain.py \
-i <path for the image directory> \
-o <path for the directory where stain color matrices will be stored> \
--extractor-type svd \
--n-jobs 40
```

### 4. Apply Stain SAN

Apply Stain SAN using the stored stain color matrices by running
```
python scripts/san_images.py \
--train-input-dir <path for train input directory> \
--train-output-dir <path for train output directory> \
--test-input-dir <path for test input directory> \
--test-output-dir <path for test output directory> \
--extractor-type svd \
--n-jobs 50
```
