from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Gastric Cancer Histopathology Analysis Dashboard"
    database_url: str = f"sqlite:///{BASE_DIR / 'app.db'}"
    upload_dir: Path = BASE_DIR / "data" / "uploads"
    raw_data_dir: Path = BASE_DIR / "data" / "raw"
    histology_dataset_dir: Path = BASE_DIR / "data" / "raw" / "HMU-GC-HE-30K" / "all_image"
    support_dir: Path = BASE_DIR / "data" / "support"
    dataset_snapshot_json: Path = BASE_DIR / "data" / "support" / "dataset_snapshot.json"
    clinical_csv: Path = BASE_DIR / "data" / "support" / "HMU-GC-Clinical.csv"
    biomarker_xlsx: Path = BASE_DIR / "data" / "support" / "Gastric.xlsx"
    notebook_path: Path = BASE_DIR / "data" / "support" / "gastric-cancer-histopathology-tissue.ipynb"
    notebook_metrics_json: Path = BASE_DIR / "data" / "support" / "notebook_metrics.json"
    confusion_matrix_image: Path = BASE_DIR / "app" / "static" / "assets" / "confusion_matrix.png"
    model_checkpoint: Path = BASE_DIR / "models" / "checkpoints" / "gastric_rnn_classifier.pt"
    patch_size: int = 56
    image_size: int = 224
    sequence_length: int = 16
    prototype_samples_per_class: int = 60
    minimum_checkpoint_accuracy: float = 0.5

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
