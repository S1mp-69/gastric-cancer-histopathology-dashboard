import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.core.config import settings
from app.ml.constants import TISSUE_CLASSES
from app.ml.model import GastricCancerRNN
from training.dataset import HistopathologySequenceDataset
from training.prepare_manifest import build_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the gastric histopathology CNN-BiLSTM classifier.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--manifest", type=Path, default=Path("model_train/runs/dataset_manifest.csv"))
    parser.add_argument("--limit-per-class", type=int, default=0)
    return parser.parse_args()


def ensure_manifest(manifest_path: Path, limit_per_class: int) -> pd.DataFrame:
    if manifest_path.exists():
        return pd.read_csv(manifest_path)

    manifest = build_records(limit_per_class)
    if manifest.empty:
        raise ValueError(f"No images found under {settings.histology_dataset_dir}")

    from sklearn.model_selection import train_test_split

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
    output = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(manifest_path, index=False)
    return output


def build_loaders(manifest: pd.DataFrame, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = HistopathologySequenceDataset(manifest[manifest["split"] == "train"])
    val_ds = HistopathologySequenceDataset(manifest[manifest["split"] == "val"])
    test_ds = HistopathologySequenceDataset(manifest[manifest["split"] == "test"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def evaluate(model: GastricCancerRNN, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits, _ = model(inputs)
            loss = criterion(logits, targets)
            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * inputs.size(0)
            total_correct += int((predictions == targets).sum().item())
            total_items += int(inputs.size(0))
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    return {
        "loss": total_loss / max(total_items, 1),
        "accuracy": total_correct / max(total_items, 1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(TISSUE_CLASSES)))).tolist(),
    }


def main() -> None:
    args = parse_args()
    manifest = ensure_manifest(args.manifest, args.limit_per_class)
    train_loader, val_loader, test_loader = build_loaders(manifest, args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GastricCancerRNN(num_classes=len(TISSUE_CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_val_accuracy = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_items = 0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            predictions = logits.argmax(dim=1)
            running_loss += loss.item() * inputs.size(0)
            running_correct += int((predictions == targets).sum().item())
            running_items += int(inputs.size(0))

        train_metrics = {
            "loss": running_loss / max(running_items, 1),
            "accuracy": running_correct / max(running_items, 1),
        }
        val_metrics = evaluate(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        if val_metrics["accuracy"] >= best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            settings.model_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": TISSUE_CLASSES,
                    "val_accuracy": val_metrics["accuracy"],
                    "settings": {
                        "image_size": settings.image_size,
                        "patch_size": settings.patch_size,
                        "sequence_length": settings.sequence_length,
                    },
                },
                settings.model_checkpoint,
            )

        print(
            f"Epoch {epoch}: train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_loss={val_metrics['loss']:.4f}"
        )

    test_metrics = evaluate(model, test_loader, criterion, device)
    log_dir = Path("model_train/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "training_summary.json").write_text(
        json.dumps({"history": history, "test": test_metrics}, indent=2),
        encoding="utf-8",
    )
    print(f"Training complete. Test accuracy: {test_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
