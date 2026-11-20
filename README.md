# Gastric Cancer Histopathology Analysis Dashboard

This project is a gastric cancer analysis website built around the GCHTID histopathology dataset, the notebook benchmark results, the clinical CSV, and the biomarker spreadsheet.

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
- A fallback similarity model so the website still works even before a full checkpoint is trained

## Project structure

- `app/`: FastAPI app, analytics layer, SQL models, inference logic, dashboard UI
- `training/`: manifest generation and training scripts
- `data/raw/HMU-GC-HE-30K/all_image/`: extracted histopathology images
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

Use Python `3.11` if possible (Python `3.12` is also supported by the pinned PyTorch build). The requirements file is configured for the official CUDA 12.8 wheels.

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

## Train the model checkpoint

Generate a new local-path manifest without overwriting prior manifests:

```powershell
python -m training.prepare_manifest --output model_train/runs/dataset_manifest_local.csv
```

Run training:

```powershell
python -m training.train --epochs 40 --batch-size 32
```

Each training run is saved in its own timestamped folder under:

`model_train/runs/`

The original checkpoint at `models/checkpoints/gastric_rnn_classifier.pt` is retained as the baseline. A new checkpoint is activated for the dashboard only when its validation accuracy exceeds that baseline; otherwise the fallback similarity model remains available.

## Fallback inference mode

If you do not train the checkpoint, the app still works using a fallback similarity model built from the real dataset. This keeps the upload workflow functional while the trained model is being prepared.

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
- For real medical deployment, the model must be validated further with clinical supervision.
