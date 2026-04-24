import json
from pathlib import Path

import numpy as np

from app.core.config import settings
from app.ml.constants import CANCER_WEIGHTS, CLASS_COLORS, CLASS_LABELS, TISSUE_CLASSES
from app.ml.preprocessing import image_signature, image_to_patch_sequence, load_rgb_image, softmax

try:
    import torch

    from app.ml.model import GastricCancerRNN

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - allows the app to start before full ML deps are installed
    TORCH_AVAILABLE = False
    torch = None
    GastricCancerRNN = None


class InferenceService:
    def __init__(self) -> None:
        self.class_names = TISSUE_CLASSES
        self.prototypes = self._build_prototypes()
        self.model_loaded = False
        self.model_source = "prototype-baseline"
        self.checkpoint_validation_accuracy = 0.0
        self.device = None
        self.model = None
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = GastricCancerRNN(num_classes=len(self.class_names)).to(self.device)
            self._load_checkpoint()

    def _build_prototypes(self) -> dict[str, np.ndarray]:
        prototypes: dict[str, np.ndarray] = {}
        dataset_root = settings.histology_dataset_dir
        if not dataset_root.exists():
            return prototypes

        for class_name in self.class_names:
            class_dir = dataset_root / class_name
            if not class_dir.exists():
                continue
            signatures: list[np.ndarray] = []
            for image_path in sorted(class_dir.glob("*.png"))[: settings.prototype_samples_per_class]:
                image = load_rgb_image(image_path, settings.image_size)
                signatures.append(image_signature(image))
            if signatures:
                prototypes[class_name] = np.stack(signatures)
        return prototypes

    def _load_checkpoint(self) -> None:
        if not TORCH_AVAILABLE or self.model is None:
            return

        checkpoint_path = settings.model_checkpoint
        if checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location=self.device)
            self.checkpoint_validation_accuracy = float(state.get("val_accuracy", 0.0))
            if self.checkpoint_validation_accuracy >= settings.minimum_checkpoint_accuracy:
                self.model.load_state_dict(state["model_state_dict"])
                self.model.eval()
                self.model_loaded = True
                self.model_source = "trained-checkpoint"

    def status(self) -> dict:
        return {
            "mode": self.model_source,
            "checkpoint_available": self.model_loaded,
            "checkpoint_found": settings.model_checkpoint.exists(),
            "checkpoint_validation_accuracy": round(self.checkpoint_validation_accuracy, 4),
            "torch_available": TORCH_AVAILABLE,
            "prototype_ready": bool(self.prototypes),
            "checkpoint_path": str(settings.model_checkpoint),
        }

    def _format_result(self, probabilities: dict[str, float]) -> dict:
        sorted_items = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        predicted_class, top_probability = sorted_items[0]
        cancer_percentage = sum(probabilities[class_name] * CANCER_WEIGHTS[class_name] for class_name in self.class_names)

        if cancer_percentage < 20:
            risk_band = "Low"
        elif cancer_percentage < 55:
            risk_band = "Moderate"
        else:
            risk_band = "High"

        class_probabilities = [
            {
                "class_code": class_name,
                "label": CLASS_LABELS[class_name],
                "color": CLASS_COLORS[class_name],
                "probability": round(probabilities[class_name] * 100.0, 2),
            }
            for class_name, _ in sorted_items
        ]

        return {
            "predicted_class": predicted_class,
            "predicted_label": CLASS_LABELS[predicted_class],
            "cancer_percentage": round(cancer_percentage, 2),
            "risk_band": risk_band,
            "confidence_score": round(top_probability * 100.0, 2),
            "top_probability": round(top_probability * 100.0, 2),
            "class_probabilities": class_probabilities,
            "model_source": self.model_source,
        }

    def _predict_with_checkpoint(self, image_path: Path) -> dict:
        image = load_rgb_image(image_path, settings.image_size)
        sequence = image_to_patch_sequence(image, settings.patch_size, settings.sequence_length)
        sequence = sequence.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, attention = self.model(sequence)
            probabilities_tensor = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        probabilities = {
            class_name: float(probability)
            for class_name, probability in zip(self.class_names, probabilities_tensor.tolist(), strict=True)
        }
        result = self._format_result(probabilities)
        result["attention_summary"] = [round(float(value), 4) for value in attention.squeeze(0).cpu().numpy().tolist()]
        return result

    def _predict_with_prototypes(self, image_path: Path) -> dict:
        if not self.prototypes:
            raise FileNotFoundError(
                "No trained model checkpoint or prototype dataset could be loaded. Ensure the dataset is present."
            )

        image = load_rgb_image(image_path, settings.image_size)
        signature = image_signature(image)

        similarities = []
        for class_name in self.class_names:
            exemplars = self.prototypes[class_name]
            distances = np.linalg.norm(exemplars - signature, axis=1)
            score = -float(np.mean(np.sort(distances)[:5]))
            similarities.append(score)

        probabilities_list = softmax(similarities, temperature=0.05)
        probabilities = {
            class_name: probability
            for class_name, probability in zip(self.class_names, probabilities_list, strict=True)
        }
        return self._format_result(probabilities)

    def predict(self, image_path: Path) -> dict:
        if self.model_loaded:
            return self._predict_with_checkpoint(image_path)
        return self._predict_with_prototypes(image_path)


inference_service = InferenceService()
