import json
from functools import lru_cache

import pandas as pd

from app.core.config import settings
from app.ml.constants import CLASS_COLORS, CLASS_LABELS, TISSUE_CLASSES
from app.ml.inference import inference_service


def _as_chart_items(mapping: dict[str, int | float], use_labels: bool = False) -> list[dict]:
    items = []
    for key, value in mapping.items():
        items.append(
            {
                "key": key,
                "label": CLASS_LABELS.get(key, key) if use_labels else key,
                "value": value,
                "color": CLASS_COLORS.get(key, "#6b7280"),
            }
        )
    return items


@lru_cache(maxsize=1)
def dataset_summary() -> dict:
    counts = {}
    total_images = 0
    for class_name in TISSUE_CLASSES:
        class_dir = settings.histology_dataset_dir / class_name
        count = len(list(class_dir.glob("*.png"))) if class_dir.exists() else 0
        counts[class_name] = count
        total_images += count

    return {
        "dataset_path": str(settings.histology_dataset_dir),
        "total_images": total_images,
        "image_size": settings.image_size,
        "sequence_length": settings.sequence_length,
        "patch_size": settings.patch_size,
        "class_distribution": _as_chart_items(counts, use_labels=True),
    }


@lru_cache(maxsize=1)
def clinical_summary() -> dict:
    df = pd.read_csv(settings.clinical_csv)
    her2 = pd.to_numeric(
        df["HER-2(Negative=0，1=1+，2=2+，3=3+)"],
        errors="coerce",
    )
    histology_counts = df["Histological Type"].fillna("Unknown").value_counts().head(6).to_dict()

    return {
        "total_patients": int(len(df)),
        "mean_age": round(float(pd.to_numeric(df["Age"], errors="coerce").mean()), 1),
        "sex_distribution": [{"label": key, "value": int(value)} for key, value in df["Sex"].value_counts().to_dict().items()],
        "stage_distribution": [
            {"label": key, "value": int(value)}
            for key, value in df["Comprehensive Staging"].astype(str).value_counts().to_dict().items()
        ],
        "top_histologies": [{"label": key, "value": int(value)} for key, value in histology_counts.items()],
        "her2_distribution": [
            {"label": str(int(key)) if pd.notna(key) else "Unknown", "value": int(value)}
            for key, value in her2.value_counts(dropna=True).to_dict().items()
        ],
    }


@lru_cache(maxsize=1)
def biomarker_summary() -> dict:
    df = pd.read_excel(settings.biomarker_xlsx)
    grouped = (
        df.groupby("target_symbol", as_index=False)[["predicted.sum", "all.sum"]]
        .max()
        .sort_values(by=["predicted.sum", "all.sum"], ascending=False)
        .head(10)
    )
    return {
        "sheet_name": "Table s2",
        "rows": int(len(df)),
        "top_targets": [
            {
                "target_symbol": row["target_symbol"],
                "predicted_sum": int(row["predicted.sum"]),
                "all_sum": int(row["all.sum"]),
            }
            for _, row in grouped.iterrows()
        ],
    }


@lru_cache(maxsize=1)
def benchmark_summary() -> dict:
    payload = json.loads(settings.notebook_metrics_json.read_text(encoding="utf-8"))
    payload["confusion_matrix_image"] = "/static/assets/confusion_matrix.png"
    payload["source_notebook"] = str(settings.notebook_path)
    return payload


def demo_summary() -> dict:
    return {
        "model_status": inference_service.status(),
        "dataset": dataset_summary(),
        "clinical": clinical_summary(),
        "biomarkers": biomarker_summary(),
        "benchmark": benchmark_summary(),
    }
