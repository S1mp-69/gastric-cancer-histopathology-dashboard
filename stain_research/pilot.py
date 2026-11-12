"""Fit on training images only; produce labeled augmentation previews and QC."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance

from transforms import extract, fit, sample, transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-class", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output folder")
    if args.train_per_class < 2:
        parser.error("Need at least two training images per class")
    manifest = pd.read_csv(args.project / "model_train/runs/dataset_manifest.csv")
    classes = sorted(manifest.class_code.unique())
    records, bases = [], []
    start = time.perf_counter()
    for code in classes:
        rows = manifest[(manifest.split == "train") & (manifest.class_code == code)]
        rows = rows.sample(n=args.train_per_class, random_state=42)
        for row in rows.itertuples(index=False):
            with Image.open(row.image_path) as source:
                image = np.array(source.convert("RGB"))
            try:
                bases.append(extract(image))
                records.append({"path": row.image_path, "class": code, "status": "ok", "split": "train"})
            except ValueError as error:
                records.append({"path": row.image_path, "class": code, "status": str(error), "split": "train"})
        print(f"Fitted source extraction: {code}", flush=True)
    parameters = fit(bases)
    args.output.mkdir(parents=True)
    serialized = {key: value.tolist() if isinstance(value, np.ndarray) else value
                  for key, value in parameters.items()}
    (args.output / "training_parameters.json").write_text(json.dumps(serialized, indent=2))
    pd.DataFrame(records).to_csv(args.output / "fit_manifest.csv", index=False)
    columns = ["original", "color_jitter", "spherical", "elliptical", "uniform"]
    sheet = Image.new("RGB", (5 * 224, len(classes) * 248 + 30), "white")
    draw = ImageDraw.Draw(sheet)
    for column, name in enumerate(columns):
        draw.text((column * 224 + 8, 8), name, fill="black")
    qc = []
    for index, code in enumerate(classes):
        rows = manifest[(manifest.split == "val") & (manifest.class_code == code)]
        row = rows.sample(n=1, random_state=42).iloc[0]
        with Image.open(row.image_path) as source:
            image = np.array(source.convert("RGB").resize((224, 224)))
        previews = {"original": image}
        previews["color_jitter"] = np.array(ImageEnhance.Color(
            ImageEnhance.Contrast(Image.fromarray(image)).enhance(1.1)).enhance(0.8))
        for mode in columns[2:]:
            # Same random intensity draw is not required for a visual preview.
            transformed, failed = transform(image, parameters, mode, np.random.default_rng(42 + index))
            repeat, _ = transform(image, parameters, mode, np.random.default_rng(42 + index))
            assert np.array_equal(transformed, repeat)
            assert transformed.shape == image.shape and transformed.dtype == np.uint8
            previews[mode] = transformed
            background = np.sum((-np.log(np.clip(image.astype(float), 1e-4, 255) / 255)) ** 2, axis=2) <= 0.09
            qc.append({"class": code, "mode": mode, "source": row.image_path,
                       "extraction_fallback": failed,
                       "mean_absolute_pixel_change": float(np.abs(transformed.astype(float) - image).mean()),
                       "background_max_change": int(np.abs(transformed.astype(int) - image)[background].max())
                       if background.any() else None})
        for column, name in enumerate(columns):
            sheet.paste(Image.fromarray(previews[name]), (column * 224, 30 + index * 248))
            draw.text((column * 224 + 6, 30 + index * 248 + 225), code, fill="black")
    sheet.save(args.output / "augmentation_preview.png")
    pd.DataFrame(qc).to_csv(args.output / "preview_qc.csv", index=False)
    sampling_qc = {}
    for mode in columns[2:]:
        rng = np.random.default_rng(123)
        draws = np.stack([sample(parameters, mode, rng).reshape(-1) for _ in range(10000)])
        sampling_qc[mode] = {"post_clipping_covariance_trace": float(np.trace(np.cov(draws, rowvar=False))),
                             "zero_component_fraction": float((draws == 0).mean())}
    summary = {"status": "preprocessing validation only; no classification improvement evaluated",
               "training_attempted": len(records), "training_success": len(bases),
               "training_failed": len(records) - len(bases), "preview_images": len(classes),
               "target_pre_clipping_variance_trace": float(parameters["variance"] * 6),
               "sampling_qc": sampling_qc, "wall_seconds": time.perf_counter() - start,
               "seed": 42, "normalization_at_inference": "deterministic training-median matrix for all SAN variants",
               "note": "Validation images are augmented only to illustrate transforms, not scored or used to fit parameters."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
