from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image


def load_rgb_image(image_path: Path, image_size: int | tuple[int, int]) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    if isinstance(image_size, int):
        target_size = (image_size, image_size)
    else:
        target_size = image_size
    return image.resize(target_size)


def image_to_patch_sequence(
    image: Image.Image,
    patch_size: int,
    sequence_length: int,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> torch.Tensor:
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    patches: list[torch.Tensor] = []

    for top in range(0, image_array.shape[0], patch_size):
        for left in range(0, image_array.shape[1], patch_size):
            patch = image_array[top : top + patch_size, left : left + patch_size]
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue
            patches.append(torch.from_numpy(patch).permute(2, 0, 1))

    if not patches:
        raise ValueError("No valid patches were extracted from the image.")

    if len(patches) >= sequence_length:
        patches = patches[:sequence_length]
    else:
        while len(patches) < sequence_length:
            patches.append(patches[-1].clone())

    sequence = torch.stack(patches)
    if mean is not None or std is not None:
        if mean is None or std is None:
            raise ValueError("Both mean and std are required when normalizing patch sequences.")
        mean_tensor = torch.tensor(mean, dtype=sequence.dtype).view(1, 3, 1, 1)
        std_tensor = torch.tensor(std, dtype=sequence.dtype).clamp_min(1e-6).view(1, 3, 1, 1)
        sequence = (sequence - mean_tensor) / std_tensor
    return sequence


def image_signature(image: Image.Image, signature_size: int = 24) -> np.ndarray:
    reduced = image.resize((signature_size, signature_size))
    signature = np.asarray(reduced, dtype=np.float32) / 255.0
    return signature.reshape(-1)


def softmax(values: Iterable[float], temperature: float = 1.0) -> list[float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return []
    shifted = array - array.max()
    exp_values = np.exp(shifted / max(temperature, 1e-6))
    probs = exp_values / exp_values.sum()
    return probs.astype(np.float64).tolist()
