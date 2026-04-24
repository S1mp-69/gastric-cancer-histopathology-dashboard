from typing import Tuple

import torch
from torch import nn


class PatchFeatureEncoder(nn.Module):
    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Linear(128, feature_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        features = self.encoder(patches).flatten(start_dim=1)
        return self.projection(features)


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.energy = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, sequence_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        energy = torch.tanh(self.energy(sequence_features))
        attention_scores = self.score(energy).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context = torch.sum(sequence_features * attention_weights.unsqueeze(-1), dim=1)
        return context, attention_weights


class GastricCancerRNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.patch_encoder = PatchFeatureEncoder(feature_dim=feature_dim)
        self.rnn = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.attention = AdditiveAttention(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, patch_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, channels, height, width = patch_sequence.shape
        patches = patch_sequence.view(batch_size * sequence_length, channels, height, width)
        encoded = self.patch_encoder(patches)
        encoded = encoded.view(batch_size, sequence_length, -1)
        sequence_output, _ = self.rnn(encoded)
        context, attention = self.attention(sequence_output)
        logits = self.classifier(context)
        return logits, attention
