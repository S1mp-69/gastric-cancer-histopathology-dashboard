"""Independent validation-sample preprocessing test, with no classifier training."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.stats import spearmanr

from controller import apply, diagnostics, fit_controller, quality, strength
from transforms import extract, fit


def load(path):
    with Image.open(path) as source:
        return np.array(source.convert("RGB").resize((224, 224)))


def correlation(x, y):
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=6)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output folder")
    if args.per_class < 2 or args.repeats < 2:
        parser.error("At least two images per class and repeats required")
    manifest = pd.read_csv(args.project / "model_train/runs/dataset_manifest.csv")
    classes = sorted(manifest.class_code.unique())
    train = pd.concat([manifest[(manifest.split == "train") & (manifest.class_code == c)]
                           .sample(n=args.per_class, random_state=42) for c in classes])
    validation = pd.concat([manifest[(manifest.split == "val") & (manifest.class_code == c)]
                                .sample(n=args.per_class, random_state=17) for c in classes])
    assert set(train.image_path).isdisjoint(validation.image_path)
    train_images = [load(p) for p in train.image_path]
    parameters = fit_controller(train_images)
    bases = []
    for image in train_images:
        try:
            bases.append(extract(image))
        except ValueError:
            pass
    stain = fit(bases)
    # Uniform control uses the mean strength on TRAINING, never validation.
    training_alpha = []
    for image in train_images:
        try:
            training_alpha.append(strength(diagnostics(image)[0], parameters))
        except ValueError:
            training_alpha.append(0.0)
    uniform = float(np.mean(training_alpha))
    images = [load(p) for p in validation.image_path]
    alphas, scores = [], []
    for image in images:
        try:
            values = diagnostics(image)[0]
            scores.append(quality(values, parameters))
            alphas.append(strength(values, parameters))
        except ValueError:
            scores.append(float("nan"))
            alphas.append(0.0)
    shuffled = np.random.default_rng(43).permutation(alphas)
    modes = ["fixed", "uniform_weaker", "shuffled_strength", "adaptive"]
    records, repeat_records, previews = [], [], {}
    for i, (image, row) in enumerate(zip(images, validation.itertuples(index=False))):
        fixed = {"fixed": 1.0, "uniform_weaker": uniform,
                 "shuffled_strength": float(shuffled[i]), "adaptive": None}
        values = {mode: [] for mode in modes}
        changes = {mode: [] for mode in modes}
        failures = {mode: 0 for mode in modes}
        for repeat in range(args.repeats):
            seed = 10000 + i * args.repeats + repeat
            noise = np.random.default_rng(seed).normal(0, 1, image.shape)
            noisy = np.clip(np.rint(image.astype(float) + noise), 0, 255).astype(np.uint8)
            input_rms = float(np.sqrt(np.mean((noisy.astype(float) - image) ** 2)))
            for mode in modes:
                clean, info = apply(image, stain, parameters, np.random.default_rng(seed), fixed[mode])
                perturbed, noisy_info = apply(noisy, stain, parameters, np.random.default_rng(seed), fixed[mode])
                gain = float(np.sqrt(np.mean((perturbed.astype(float) - clean) ** 2)) / max(input_rms, 1e-8))
                delta = float(np.sqrt(np.mean((clean.astype(float) - image) ** 2)))
                values[mode].append(gain)
                changes[mode].append(delta)
                failures[mode] += int(info["fallback"] or noisy_info["fallback"])
                repeat_records.append({"path": row.image_path, "class_code": row.class_code,
                                       "repeat": repeat, "mode": mode, "gain": gain,
                                       "augmentation_rms": delta, "fallback": info["fallback"] or noisy_info["fallback"]})
                if row.class_code not in previews and repeat == 0:
                    previews.setdefault("pending", {})[mode] = clean
        if row.class_code not in previews:
            previews[row.class_code] = {"original": image, **previews.pop("pending")}
        record = {"path": row.image_path, "class_code": row.class_code,
                  "quality_risk": scores[i], "adaptive_strength": alphas[i]}
        for mode in modes:
            record[f"{mode}_gain"] = float(np.mean(values[mode]))
            record[f"{mode}_augmentation_rms"] = float(np.mean(changes[mode]))
            record[f"{mode}_fallback_repeats"] = failures[mode]
        records.append(record)
        if (i + 1) % 20 == 0:
            print(f"Measured {i + 1}/{len(images)} validation images", flush=True)
    frame = pd.DataFrame(records)
    valid = frame[np.isfinite(frame.quality_risk)]
    association = correlation(valid.quality_risk, valid.fixed_gain)
    # Stratify permutation by class to check whether tissue identity explains association.
    rng = np.random.default_rng(77)
    permutations = []
    for _ in range(1000):
        shuffled_risk = valid.quality_risk.to_numpy().copy()
        for code in classes:
            positions = np.flatnonzero(valid.class_code.to_numpy() == code)
            shuffled_risk[positions] = rng.permutation(shuffled_risk[positions])
        permutations.append(correlation(shuffled_risk, valid.fixed_gain))
    p_value = float((1 + np.sum(np.asarray(permutations) >= association)) / 1001) if association is not None else None
    summary = {"status": "exploratory numerical stability study; no classification or clinical benefit established",
               "training_images": len(train), "validation_images": len(frame),
               "controller_parameters": parameters, "training_mean_strength": uniform,
               "validation_mean_strength": float(np.mean(alphas)),
               "noise": "independent RGB Gaussian sigma=1/255, rounded/clipped to uint8",
               "repeats_per_image": args.repeats,
               "spearman_quality_vs_fixed_gain": association,
               "within_class_permutation_one_sided_p": p_value,
               "mode_results": {}, "per_class": {},
               "limitations": ["All diagnostics are proxies, not expert image-quality labels.",
                               "Noise is synthetic; no evidence of real scanner robustness.",
                               "Residual preservation is used in ALL four arms to isolate controller effect.",
                               "Shuffled-strength arm matches the validation strength multiset as an offline control, not a deployable policy.",
                               "Pilot uses one training fit and one validation sample; patient identities unverified.",
                               "No classification training, accuracy comparison, or confirmed novelty."]}
    for mode in modes:
        difference = (frame[f"{mode}_gain"] - frame.uniform_weaker_gain).to_numpy()
        boot = []
        for _ in range(2000):
            boot.append(float(rng.choice(difference, size=len(difference), replace=True).mean()))
        summary["mode_results"][mode] = {
            "mean_gain": float(frame[f"{mode}_gain"].mean()),
            "median_gain": float(frame[f"{mode}_gain"].median()),
            "mean_augmentation_rms": float(frame[f"{mode}_augmentation_rms"].mean()),
            "mean_gain_difference_vs_uniform": float(difference.mean()),
            "descriptive_image_bootstrap_95_interval": np.quantile(boot, [.025, .975]).tolist(),
            "fallback_repeats": int(frame[f"{mode}_fallback_repeats"].sum())}
    for code in classes:
        subset = frame[frame.class_code == code]
        summary["per_class"][code] = {"mean_strength": float(subset.adaptive_strength.mean()),
            "quality_vs_fixed_gain_spearman": correlation(subset.quality_risk, subset.fixed_gain)}
    args.output.mkdir(parents=True)
    frame.to_csv(args.output / "image_metrics.csv", index=False)
    pd.DataFrame(repeat_records).to_csv(args.output / "repeat_metrics.csv", index=False)
    train.to_csv(args.output / "training_sample.csv", index=False)
    validation.to_csv(args.output / "validation_sample.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    serialized = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in stain.items()}
    (args.output / "stain_parameters.json").write_text(json.dumps(serialized, indent=2))
    columns = ["original", *modes]
    sheet = Image.new("RGB", (224 * len(columns), 248 * len(classes) + 30), "white")
    draw = ImageDraw.Draw(sheet)
    for c, name in enumerate(columns):
        draw.text((c * 224 + 5, 8), name, fill="black")
    for r, code in enumerate(classes):
        for c, name in enumerate(columns):
            sheet.paste(Image.fromarray(previews[code][name]), (c * 224, r * 248 + 30))
            draw.text((c * 224 + 5, r * 248 + 255), code, fill="black")
    sheet.save(args.output / "controller_preview.png")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
