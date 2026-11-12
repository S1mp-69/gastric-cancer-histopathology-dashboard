"""Reproducible GPU-capable training for the project's CNN + BiLSTM + attention classifier."""

import argparse
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from app.core.config import settings
from app.ml.constants import TISSUE_CLASSES
from app.ml.model import GastricCancerRNN
from training.dataset import HistopathologySequenceDataset
from training.prepare_manifest import build_records

matplotlib.use("Agg")
from matplotlib import pyplot as plt


class CudaOutOfMemory(RuntimeError):
    """Signals a whole-run retry with a smaller batch size."""


def log_event(run_dir: Path, message: str) -> None:
    print(message)
    with (run_dir / "training.log").open("a", encoding="utf-8") as log_file:
        log_file.write(f"{message}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the gastric histopathology CNN-BiLSTM-attention classifier.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7, help="Early-stopping patience in epochs.")
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--limit-per-class", type=int, default=0)
    parser.add_argument("--manifest", type=Path, default=None, help="Optional existing, local-path manifest.")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalization-samples", type=int, default=512)
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA automatic mixed precision.")
    parser.add_argument("--no-activate", action="store_true", help="Do not update the dashboard active-model pointer.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def make_split_manifest(limit_per_class: int) -> pd.DataFrame:
    manifest = build_records(limit_per_class)
    if manifest.empty:
        raise ValueError(f"No images found under {settings.histology_dataset_dir}")
    missing = sorted(set(TISSUE_CLASSES) - set(manifest["class_code"]))
    if missing:
        raise ValueError(f"Dataset is missing expected class directories/images: {', '.join(missing)}")
    train_df, temp_df = train_test_split(
        manifest, test_size=0.2, random_state=42, stratify=manifest["class_code"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["class_code"]
    )
    return pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    )


def estimate_normalization(manifest: pd.DataFrame, sample_count: int) -> dict[str, list[float]]:
    """Estimate RGB moments on deterministic train samples; validation/test never influence them."""
    paths = manifest.loc[manifest["split"] == "train", "image_path"].sort_values().head(sample_count)
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    pixels = 0
    for value in tqdm(paths, desc="Estimating normalization", leave=False):
        with Image.open(value) as image:
            rgb = image.convert("RGB").resize((settings.image_size, settings.image_size))
            array = np.asarray(rgb, dtype=np.float32) / 255.0
        channel_sum += array.sum(axis=(0, 1))
        channel_sq_sum += np.square(array).sum(axis=(0, 1))
        pixels += array.shape[0] * array.shape[1]
    mean = channel_sum / pixels
    std = np.sqrt(np.maximum(channel_sq_sum / pixels - np.square(mean), 1e-8))
    return {"mean": mean.round(8).tolist(), "std": std.round(8).tolist(), "samples": int(len(paths))}


def build_loaders(
    manifest: pd.DataFrame, batch_size: int, normalization: dict[str, list[float]], num_workers: int
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180, interpolation=InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02),
        ]
    )
    train_ds = HistopathologySequenceDataset(
        manifest[manifest["split"] == "train"], image_transform=train_transform, normalization=normalization
    )
    val_ds = HistopathologySequenceDataset(manifest[manifest["split"] == "val"], normalization=normalization)
    test_ds = HistopathologySequenceDataset(manifest[manifest["split"] == "test"], normalization=normalization)
    common = {
        "batch_size": batch_size,
        "num_workers": max(num_workers, 0),
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        common["persistent_workers"] = True
    return (
        DataLoader(train_ds, shuffle=True, **common),
        DataLoader(val_ds, shuffle=False, **common),
        DataLoader(test_ds, shuffle=False, **common),
    )


def _is_cuda_oom(exc: RuntimeError) -> bool:
    return torch.cuda.is_available() and "out of memory" in str(exc).lower()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_items = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits, _ = model(inputs)
            total_loss += criterion(logits, targets).item() * inputs.size(0)
            total_items += int(inputs.size(0))
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    report = classification_report(
        y_true, y_pred, labels=list(range(len(TISSUE_CLASSES))), target_names=TISSUE_CLASSES, output_dict=True, zero_division=0
    )
    per_class = [
        {
            "class_code": class_name,
            "precision": float(report[class_name]["precision"]),
            "recall": float(report[class_name]["recall"]),
            "f1_score": float(report[class_name]["f1-score"]),
            "support": int(report[class_name]["support"]),
        }
        for class_name in TISSUE_CLASSES
    ]
    return {
        "loss": total_loss / max(total_items, 1),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(len(TISSUE_CLASSES))), average="macro", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(TISSUE_CLASSES)))).tolist(),
    }


def save_confusion_matrix(matrix: list[list[int]], destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=range(len(TISSUE_CLASSES)), yticks=range(len(TISSUE_CLASSES)), xticklabels=TISSUE_CLASSES, yticklabels=TISSUE_CLASSES)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Test confusion matrix — CNN + BiLSTM + attention")
    for row, values in enumerate(matrix):
        for column, value in enumerate(values):
            axis.text(column, row, str(value), ha="center", va="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def save_training_curves(history: list[dict], destination: Path) -> None:
    epochs = [item["epoch"] for item in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [item["train"]["loss"] for item in history], label="train")
    axes[0].plot(epochs, [item["val"]["loss"] for item in history], label="validation")
    axes[0].set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(epochs, [item["train"]["accuracy"] for item in history], label="train accuracy")
    axes[1].plot(epochs, [item["val"]["accuracy"] for item in history], label="validation accuracy")
    axes[1].plot(epochs, [item["val"]["macro_f1"] for item in history], label="validation macro F1")
    axes[1].set(title="Validation performance", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def checkpoint_summary(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "found": False}
    state = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "path": str(path),
        "found": True,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "val_accuracy": float(state.get("val_accuracy", 0.0)),
        "class_names": state.get("class_names", []),
    }


def train_once(
    args: argparse.Namespace,
    run_dir: Path,
    manifest: pd.DataFrame,
    normalization: dict[str, list[float]],
    device: torch.device,
    batch_size: int,
) -> tuple[dict, dict, Path]:
    train_loader, val_loader, test_loader = build_loaders(manifest, batch_size, normalization, args.num_workers)
    model = GastricCancerRNN(num_classes=len(TISSUE_CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.scheduler_patience
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = checkpoint_dir / "best_checkpoint.pt"
    last_checkpoint = checkpoint_dir / "last_checkpoint.pt"
    history: list[dict] = []
    best_score = (-1.0, -1.0)
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        correct_total = 0
        item_total = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            try:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    logits, _ = model(inputs)
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            except RuntimeError as exc:
                if _is_cuda_oom(exc):
                    torch.cuda.empty_cache()
                    raise CudaOutOfMemory(str(exc)) from exc
                raise
            loss_total += loss.item() * inputs.size(0)
            correct_total += int((logits.argmax(dim=1) == targets).sum().item())
            item_total += int(inputs.size(0))
        train_metrics = {"loss": loss_total / item_total, "accuracy": correct_total / item_total}
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["loss"])
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics, "learning_rate": learning_rate})
        score = (val_metrics["macro_f1"], val_metrics["accuracy"])
        state = {
            "model_state_dict": model.state_dict(),
            "class_names": TISSUE_CLASSES,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch": epoch,
            "normalization": normalization,
            "settings": {"image_size": settings.image_size, "patch_size": settings.patch_size, "sequence_length": settings.sequence_length},
            "architecture": "custom CNN encoder + BiLSTM + additive attention",
        }
        torch.save(state, last_checkpoint)
        if score > best_score:
            best_score = score
            epochs_without_improvement = 0
            torch.save(state, best_checkpoint)
        else:
            epochs_without_improvement += 1
        log_event(
            run_dir,
            f"Epoch {epoch}: train_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} val_loss={val_metrics['loss']:.4f} lr={learning_rate:.2e}"
        )
        if epochs_without_improvement >= args.patience:
            log_event(run_dir, f"Early stopping after {epoch} epochs (patience={args.patience}).")
            break

    best_state = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model_state_dict"])
    best_val = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    return {"history": history, "best_validation": best_val, "test": test_metrics}, best_state, best_checkpoint


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Install the CUDA-enabled PyTorch wheel and verify torch.cuda.is_available() before training."
        )
    set_seed(args.seed)
    run_name = args.run_name or f"cnn_bilstm_attention_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path("model_train/runs") / run_name
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    log_event(run_dir, f"Run started on {torch.cuda.get_device_name(0)} with requested batch size {args.batch_size}.")
    manifest = pd.read_csv(args.manifest) if args.manifest else make_split_manifest(args.limit_per_class)
    if not manifest["image_path"].map(lambda value: Path(value).exists()).all():
        raise ValueError("Manifest contains missing image paths. Regenerate it with training.prepare_manifest.")
    manifest_path = run_dir / "dataset_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    normalization = estimate_normalization(manifest, args.normalization_samples)
    baseline = checkpoint_summary(settings.model_checkpoint)
    device = torch.device("cuda")
    initial_batch_size = args.batch_size
    while True:
        try:
            results, best_state, best_checkpoint = train_once(
                args, run_dir, manifest, normalization, device, initial_batch_size
            )
            break
        except CudaOutOfMemory:
            if initial_batch_size <= 1:
                raise RuntimeError("CUDA out of memory even at batch size 1.")
            initial_batch_size = max(1, initial_batch_size // 2)
            log_event(run_dir, f"CUDA out of memory; retrying the run with batch size {initial_batch_size}.")
            torch.cuda.empty_cache()

    best_val_accuracy = float(results["best_validation"]["accuracy"])
    baseline_accuracy = float(baseline.get("val_accuracy", 0.0))
    is_better = best_val_accuracy > baseline_accuracy
    activation = {"updated": False, "reason": "new checkpoint did not exceed baseline validation accuracy"}
    if is_better and not args.no_activate:
        active_payload = {
            "checkpoint_path": str(best_checkpoint.resolve()),
            "approved": True,
            "baseline_val_accuracy": baseline_accuracy,
            "new_val_accuracy": best_val_accuracy,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        settings.active_model_config.parent.mkdir(parents=True, exist_ok=True)
        settings.active_model_config.write_text(json.dumps(active_payload, indent=2), encoding="utf-8")
        activation = {"updated": True, "path": str(settings.active_model_config)}
    summary = {
        "run_name": run_name,
        "device": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "batch_size": initial_batch_size,
        "amp_enabled": not args.no_amp,
        "manifest": str(manifest_path),
        "normalization": normalization,
        "baseline": baseline,
        "best_checkpoint": str(best_checkpoint.resolve()),
        "best_validation": results["best_validation"],
        "test": results["test"],
        "better_than_baseline": is_better,
        "dashboard_activation": activation,
        "medical_note": "Research-use model evaluation only; these metrics do not establish clinical diagnostic performance.",
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(results["history"], indent=2), encoding="utf-8")
    pd.DataFrame(results["test"]["per_class"]).to_csv(run_dir / "test_per_class_metrics.csv", index=False)
    pd.DataFrame(results["test"]["confusion_matrix"], index=TISSUE_CLASSES, columns=TISSUE_CLASSES).to_csv(
        run_dir / "test_confusion_matrix.csv"
    )
    save_confusion_matrix(results["test"]["confusion_matrix"], run_dir / "test_confusion_matrix.png")
    save_training_curves(results["history"], run_dir / "training_curves.png")
    (run_dir / "training_summary.txt").write_text(
        "\n".join(
            [
                "Custom CNN + BiLSTM + additive attention training run",
                f"Best validation accuracy: {best_val_accuracy:.4%}",
                f"Best validation macro F1: {results['best_validation']['macro_f1']:.4%}",
                f"Test accuracy: {results['test']['accuracy']:.4%}",
                f"Test macro F1: {results['test']['macro_f1']:.4%}",
                f"Better than preserved baseline ({baseline_accuracy:.4%}): {is_better}",
                "Research use only; not a clinical diagnostic tool.",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
