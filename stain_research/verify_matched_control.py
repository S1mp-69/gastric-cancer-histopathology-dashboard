"""Directly verify the interpolated primary uniform-control comparison."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from controller_pilot import load
from strength_controls import paired_interval, render, state
from transforms import sample


def main():
    root = Path(__file__).parent
    run = root / "runs/strength_controls_v1"
    first = root / "runs/controller_v1"
    summary = json.loads((run / "summary.json").read_text())
    stain = json.loads((first / "stain_parameters.json").read_text())
    rows = pd.read_csv(run / "new_validation_sample.csv")
    curves = np.load(run / "curves.npz")
    results = {}
    for s, sigma in enumerate(curves["noise_sigmas"]):
        alpha = summary["results"][str(sigma)]["magnitude_matched_uniform_strength"]
        exact, changes = [], []
        for i, row in rows.iterrows():
            image = load(row.image_path)
            prepared = state(image)
            gains, rms = [], []
            for repeat in range(summary["repeats_per_sigma"]):
                seed = 40000 + i * summary["repeats_per_sigma"] + repeat
                rng = np.random.default_rng(seed)
                target = sample(stain, "spherical", rng)
                scale = 1 + rng.uniform(-0.2, 0.2)
                noisy = np.clip(np.rint(image.astype(float) + np.random.default_rng(seed + 100000)
                                       .normal(0, sigma, image.shape)), 0, 255).astype(np.uint8)
                clean = render(prepared, target, scale, alpha).astype(float)
                perturbed = render(state(noisy), target, scale, alpha).astype(float)
                gains.append(np.sqrt(np.mean((clean - perturbed) ** 2)) /
                             np.sqrt(np.mean((image.astype(float) - noisy) ** 2)))
                rms.append(np.sqrt(np.mean((clean - image) ** 2)))
            exact.append(np.mean(gains))
            changes.append(np.mean(rms))
        exact = np.array(exact)
        difference = curves["adaptive_gain"][s] - exact
        results[str(sigma)] = {
            "direct_uniform_gain": float(exact.mean()),
            "interpolated_uniform_gain": summary["results"][str(sigma)]["magnitude_matched_uniform_gain"],
            "direct_uniform_change_rms": float(np.mean(changes)),
            "adaptive_change_rms": float(curves["adaptive_rms"][s].mean()),
            "adaptive_minus_direct_uniform": float(difference.mean()),
            "descriptive_stratified_image_bootstrap_95_interval": paired_interval(
                difference, rows.class_code.to_numpy(), np.random.default_rng(42))}
    (run / "direct_control_check.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
