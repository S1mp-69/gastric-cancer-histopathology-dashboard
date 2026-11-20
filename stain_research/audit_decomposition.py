"""Exploratory training-sample audit of stain decomposition stability."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from transforms import extract, rgb2od


def main():
    root = Path(__file__).parent
    output = root / "runs/decomposition_audit"
    output.mkdir(exist_ok=False)
    manifest = pd.read_csv(root / "runs/preprocessing_v1/fit_manifest.csv")
    records = []
    for row in manifest.to_dict("records"):
        with Image.open(row["path"]) as source:
            image = np.array(source.convert("RGB"))
        basis = extract(image)
        od = rgb2od(image).reshape(-1, 3)
        mask = (od ** 2).sum(axis=1) > 0.09
        concentrations = np.linalg.lstsq(basis, od.T, rcond=None)[0]
        residual = np.linalg.norm((od.T - basis @ concentrations)[:, mask])
        residual /= max(np.linalg.norm(od[mask]), 1e-12)
        records.append({"path": row["path"], "class_code": row["class"],
                        "condition_number": float(np.linalg.cond(basis)),
                        "relative_residual": float(residual),
                        "negative_concentration_fraction": float((concentrations[:, mask] < 0).mean())})
    frame = pd.DataFrame(records)
    frame.to_csv(output / "measurements.csv", index=False)
    columns = ["condition_number", "relative_residual", "negative_concentration_fraction"]
    summary = {"images": len(frame), "split": "train", "status": "descriptive; no training or efficacy claim",
               "quantiles": frame[columns].quantile([0, 0.5, 0.9, 1]).to_dict(),
               "per_class_medians": frame.groupby("class_code")[columns].median().to_dict()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
