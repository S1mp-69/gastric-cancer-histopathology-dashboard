from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from torch.utils.data import Dataset

from app.core.config import settings
from app.ml.constants import TISSUE_CLASSES
from app.ml.preprocessing import image_to_patch_sequence, load_rgb_image


class HistopathologySequenceDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        image_transform: Callable | None = None,
        normalization: dict[str, list[float]] | None = None,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.class_to_idx = {class_name: index for index, class_name in enumerate(TISSUE_CLASSES)}
        self.image_transform = image_transform
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[index]
        image_path = Path(row["image_path"])
        image = load_rgb_image(image_path, settings.image_size)
        if self.image_transform is not None:
            image = self.image_transform(image)
        mean = tuple(self.normalization["mean"]) if self.normalization else None
        std = tuple(self.normalization["std"]) if self.normalization else None
        patch_sequence = image_to_patch_sequence(
            image,
            settings.patch_size,
            settings.sequence_length,
            mean=mean,
            std=std,
        )
        label = torch.tensor(self.class_to_idx[row["class_code"]], dtype=torch.long)
        return patch_sequence, label
