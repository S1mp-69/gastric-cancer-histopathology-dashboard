import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from app.core.config import settings
from app.ml.constants import TISSUE_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a train/val/test manifest for the gastric histopathology dataset.")
    parser.add_argument("--output", type=Path, default=Path("model_train/runs/dataset_manifest.csv"))
    parser.add_argument("--limit-per-class", type=int, default=0)
    return parser.parse_args()


def build_records(limit_per_class: int) -> pd.DataFrame:
    records: list[dict] = []
    for class_name in TISSUE_CLASSES:
        class_dir = settings.histology_dataset_dir / class_name
        image_paths = sorted(class_dir.glob("*.png"))
        if limit_per_class > 0:
            image_paths = image_paths[:limit_per_class]
        for image_path in image_paths:
            records.append({"image_path": str(image_path.resolve()), "class_code": class_name})
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    manifest = build_records(args.limit_per_class)
    if manifest.empty:
        raise ValueError(f"No images found under {settings.histology_dataset_dir}")

    train_df, temp_df = train_test_split(
        manifest,
        test_size=0.2,
        random_state=42,
        stratify=manifest["class_code"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["class_code"],
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    output = pd.concat([train_df, val_df, test_df], ignore_index=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Saved manifest to {args.output}")
    print(output["split"].value_counts().to_string())
    print(output.groupby("class_code").size().sort_index().to_string())


if __name__ == "__main__":
    main()
