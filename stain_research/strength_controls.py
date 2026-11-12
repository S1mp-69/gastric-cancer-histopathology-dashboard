"""Follow-up preprocessing controls using cached decompositions and strength curves."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from controller import apply, diagnostics
from controller_pilot import load
from transforms import rgb2od, sample


def state(image):
    values, basis, concentrations, tissue = diagnostics(image)
    return image, values, basis, concentrations, tissue, rgb2od(image).reshape(-1, 3)


def render(prepared, target, scale, alpha):
    image, _, basis, concentrations, tissue, od = prepared
    if alpha == 0:
        return image.copy()
    updated = od.copy()
    updated[tissue] += alpha * ((target * scale - basis) @ concentrations).T[tissue]
    output = np.rint(np.clip(255 * np.exp(-np.maximum(updated, 0)), 0, 255)).reshape(image.shape).astype(np.uint8)
    output.reshape(-1, 3)[~tissue] = image.reshape(-1, 3)[~tissue]
    return output


def component_strength(values, parameters, component=None):
    scaled = np.clip((values - np.array(parameters["low"])) /
                     np.maximum(np.array(parameters["high"]) - parameters["low"], 1e-8), 0, 1)
    risk = scaled.mean() if component is None else scaled[component]
    return float(1 - 0.9 * risk)


def interpolate(curves, grid, alphas):
    return np.array([np.interp(alpha, grid, curve) for alpha, curve in zip(alphas, curves)])


def matched_scale(curves, grid, base, target):
    # Offline validation-only magnitude matching; no labels or classifier outputs.
    low, high = 0.0, 10.0
    for _ in range(35):
        scale = (low + high) / 2
        actual = interpolate(curves, grid, np.clip(base * scale, 0, 1)).mean()
        if actual < target:
            low = scale
        else:
            high = scale
    return np.clip(base * ((low + high) / 2), 0, 1)


def paired_interval(difference, codes, rng):
    # Resample images within tissue classes; patient grouping remains unavailable.
    groups = [np.flatnonzero(codes == code) for code in np.unique(codes)]
    samples = []
    for _ in range(2000):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        samples.append(difference[indices].mean())
    return np.quantile(samples, [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() or args.per_class < 2 or args.repeats < 2:
        parser.error("Use a fresh output directory and at least 2 images/class and repeats")
    root = Path(__file__).parent
    original = root / "runs/controller_v1"
    first = json.loads((original / "summary.json").read_text())
    parameters = first["controller_parameters"]
    stain = json.loads((original / "stain_parameters.json").read_text())
    manifest = pd.read_csv(args.project / "model_train/runs/dataset_manifest.csv")
    old = pd.read_csv(original / "validation_sample.csv")
    available = manifest[(manifest.split == "val") & ~manifest.image_path.isin(old.image_path)]
    classes = sorted(available.class_code.unique())
    rows = pd.concat([available[available.class_code == code].sample(n=args.per_class, random_state=91)
                      for code in classes]).reset_index(drop=True)
    assert set(rows.image_path).isdisjoint(old.image_path)
    grid = np.linspace(0, 1, 21)
    sigmas = [0.5, 2.0]
    modes = ["adaptive", "condition_only", "residual_only"]
    counts = (len(sigmas), len(rows))
    gains = np.zeros((*counts, len(grid)))
    changes = np.zeros_like(gains)
    adaptive_gains = {mode: np.zeros(counts) for mode in modes}
    adaptive_changes = {mode: np.zeros(counts) for mode in modes}
    features, strengths = [], []
    for i, row in rows.iterrows():
        image = load(row.image_path)
        prepared = state(image)
        features.append(prepared[1])
        strengths.append(component_strength(prepared[1], parameters))
        for s, sigma in enumerate(sigmas):
            for repeat in range(args.repeats):
                seed = 40000 + i * args.repeats + repeat
                rng = np.random.default_rng(seed)
                target = sample(stain, "spherical", rng)
                scale = 1 + rng.uniform(-0.2, 0.2)
                noise = np.random.default_rng(seed + 100000).normal(0, sigma, image.shape)
                noisy = np.clip(np.rint(image.astype(float) + noise), 0, 255).astype(np.uint8)
                disturbed = state(noisy)
                input_rms = np.sqrt(np.mean((image.astype(float) - noisy) ** 2))
                for j, alpha in enumerate(grid):
                    clean = render(prepared, target, scale, alpha).astype(float)
                    changed = render(disturbed, target, scale, alpha).astype(float)
                    gains[s, i, j] += np.sqrt(np.mean((clean - changed) ** 2)) / input_rms / args.repeats
                    changes[s, i, j] += np.sqrt(np.mean((clean - image) ** 2)) / args.repeats
                for mode, component in zip(modes, [None, 0, 1]):
                    alpha = component_strength(prepared[1], parameters, component)
                    noisy_alpha = component_strength(disturbed[1], parameters, component)
                    clean = render(prepared, target, scale, alpha).astype(float)
                    changed = render(disturbed, target, scale, noisy_alpha).astype(float)
                    adaptive_gains[mode][s, i] += np.sqrt(np.mean((clean - changed) ** 2)) / input_rms / args.repeats
                    adaptive_changes[mode][s, i] += np.sqrt(np.mean((clean - image) ** 2)) / args.repeats
                if i == s == repeat == 0:
                    direct, _ = apply(image, stain, parameters, np.random.default_rng(seed))
                    assert np.array_equal(direct, render(prepared, target, scale, strengths[-1]))
        if (i + 1) % 10 == 0:
            print(f"Follow-up measured {i + 1}/{len(rows)}", flush=True)
    features = np.stack(features)
    strengths = np.array(strengths)
    codes = rows.class_code.to_numpy()
    rng = np.random.default_rng(42)
    results = {}
    for s, sigma in enumerate(sigmas):
        target_change = float(adaptive_changes["adaptive"][s].mean())
        matched = matched_scale(changes[s], grid, np.ones(len(rows)), target_change)
        matched_gain = interpolate(gains[s], grid, matched)
        shuffled_means, shuffled_change = [], []
        for _ in range(500):
            assignment = rng.permutation(strengths)
            assignment = matched_scale(changes[s], grid, assignment, target_change)
            shuffled_means.append(float(interpolate(gains[s], grid, assignment).mean()))
            shuffled_change.append(float(interpolate(changes[s], grid, assignment).mean()))
        comparisons = {}
        for mode in modes:
            mode_target = float(adaptive_changes[mode][s].mean())
            mode_match = matched_scale(changes[s], grid, np.ones(len(rows)), mode_target)
            mode_baseline = interpolate(gains[s], grid, mode_match)
            difference = adaptive_gains[mode][s] - mode_baseline
            comparisons[mode] = {
                "mean_gain": float(adaptive_gains[mode][s].mean()),
                "mean_augmentation_rms": mode_target,
                "magnitude_matched_uniform_gain": float(mode_baseline.mean()),
                "difference_vs_matched_uniform": float(difference.mean()),
                "descriptive_stratified_image_bootstrap_95_interval": paired_interval(difference, codes, rng)}
        results[str(sigma)] = {
            "fixed_gain": float(gains[s, :, -1].mean()),
            "condition_vs_fixed_gain_spearman": float(spearmanr(features[:, 0], gains[s, :, -1]).statistic),
            "residual_vs_fixed_gain_spearman": float(spearmanr(features[:, 1], gains[s, :, -1]).statistic),
            "magnitude_matched_uniform_strength": float(matched[0]),
            "magnitude_matched_uniform_gain": float(matched_gain.mean()),
            "matched_uniform_augmentation_rms": float(interpolate(changes[s], grid, matched).mean()),
            "shuffled_assignments": 500,
            "matched_shuffled_mean_gain": float(np.mean(shuffled_means)),
            "matched_shuffled_gain_5_95_percentiles": np.quantile(shuffled_means, [.05, .95]).tolist(),
            "fraction_shuffles_at_least_as_stable_as_adaptive": float(
                (np.asarray(shuffled_means) <= adaptive_gains["adaptive"][s].mean()).mean()),
            "maximum_shuffled_rms_matching_error": float(np.max(np.abs(np.array(shuffled_change) - target_change))),
            "components": comparisons}
    args.output.mkdir(parents=True)
    rows.to_csv(args.output / "new_validation_sample.csv", index=False)
    np.savez_compressed(args.output / "curves.npz", grid=grid, noise_sigmas=sigmas,
                        fixed_strength_gains=gains, fixed_strength_rms=changes,
                        diagnostics=features, adaptive_strength=strengths,
                        **{f"{mode}_gain": adaptive_gains[mode] for mode in modes},
                        **{f"{mode}_rms": adaptive_changes[mode] for mode in modes})
    summary = {"status": "exploratory preprocessing control study, no classifier evidence",
               "images": len(rows), "excluded_previous_validation_images": len(old),
               "repeats_per_sigma": args.repeats, "results": results,
               "limitations": ["Offline magnitude matching uses validation image changes, not a deployable policy.",
                               "Fixed/shuffled scores interpolate 21-point strength curves; adaptive scores are direct.",
                               "Matching controls mean per-image RGB RMS, not every image or OD/pixel-change distribution.",
                               "Image bootstrap ignores unknown slide/patient clustering and controller fit uncertainty.",
                               "Synthetic noise only; no clinical, classification or uniqueness claim."]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
