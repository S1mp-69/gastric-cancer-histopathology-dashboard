# Gastric Cancer Histopathology Analysis Dashboard

This project is a gastric cancer analysis website built around the GCHTID histopathology dataset, the notebook benchmark results, the clinical CSV, and the biomarker spreadsheet.

This folder is the GitHub-safe version of the project. It keeps the code, support files, trained checkpoint, and benchmark artifacts, but leaves out the large raw image dataset, local virtual environment, uploads, and SQLite database file.

## What it includes

- A gastric tissue classifier based on a lightweight CNN encoder plus BiLSTM and attention
- A live upload workflow that predicts tissue class and estimates cancer percentage
- SQLite-backed storage for prediction history
- A FastAPI dashboard for:
  - live inference
  - dataset distribution
  - notebook benchmark metrics
  - confusion matrix display
  - clinical stage analysis
  - biomarker target exploration
- A bundled trained checkpoint so the copied repo can still run predictions without the raw dataset

## Project structure

- `app/`: FastAPI app, analytics layer, SQL models, inference logic, dashboard UI
- `training/`: manifest generation and training scripts
- `data/raw/`: empty in this GitHub-safe copy; add the extracted histopathology images here if you want full dataset-backed training or the fallback similarity mode
- `data/support/`: clinical CSV, biomarker XLSX, notebook, and benchmark JSON
- `data/uploads/`: uploaded inference samples
- `models/checkpoints/`: trained model checkpoints
- `model_train/logs/`: training summaries
- `model_train/runs/`: generated manifests and experiment artifacts

## Real dataset used

- Histopathology image dataset: `31,096` images across `8` classes
- Classes: `ADI`, `DEB`, `LYM`, `MUC`, `MUS`, `NOR`, `STR`, `TUM`
- Clinical file: `HMU-GC-Clinical.csv`
- Biomarker file: `Gastric.xlsx`
- Benchmark notebook: `gastric-cancer-histopathology-tissue.ipynb`

## Setup

Use Python `3.11` if possible because the PyTorch dependency is most reliable there.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If `py -3.11` is not available, install Python 3.11 first.

## Run the dashboard

```powershell
uvicorn app.main:app --reload
```

Then open:

`http://127.0.0.1:8000`

In this GitHub-safe copy, the included checkpoint is enabled by default so the upload demo can still work even before the raw dataset is downloaded.

## Train the model checkpoint

Generate a manifest:

```powershell
python -m training.prepare_manifest
```

Run training:

```powershell
python -m training.train --epochs 5 --batch-size 32
```

The trained checkpoint is saved to:

`models/checkpoints/gastric_rnn_classifier.pt`

## Dataset download for full training and fallback mode

The raw image dataset is not bundled in this copy because it is too large for a normal GitHub repository.

Place the extracted dataset at:

`data/raw/HMU-GC-HE-30K/all_image`

Useful source links:

- Kaggle dataset page: `https://www.kaggle.com/datasets/orvile/gastric-cancer-histopathology-tissue-image-dataset`
- Figshare archive: `https://figshare.com/articles/dataset/Gastric_Cancer_Histopathology_Tissue_Image_Dataset_GCHTID_/25954813`

Once the dataset is present, the training scripts and the fallback similarity mode can use the full image collection.

## Suggested presentation flow

1. Open the dashboard and explain the dataset size and tissue classes.
2. Show the notebook benchmark confusion matrix and accuracy section.
3. Explain the CNN + BiLSTM + attention architecture.
4. Upload a tissue patch and show the predicted tissue class plus estimated cancer percentage.
5. Show the clinical stage analysis and biomarker table as supporting modules.
6. Show the SQLite prediction history section to demonstrate storage and traceability.

## Notes

- This project intentionally avoids ResNet50.
- The "cancer percentage" is derived from tissue-class probabilities for reporting inside the dashboard.
- The benchmark numbers displayed in the dashboard come from the provided notebook outputs and confusion matrix.
- The GitHub-safe copy excludes the raw image dataset, `.venv`, `app.db`, and uploaded sample files.
- For real medical deployment, the model must be validated further with clinical supervision.
