"""Neural network architectures used by the classifier."""

from __future__ import annotations

import torch
from torch import nn


class HybridCNNGRU(nn.Module):
    """A mask-aware 1D-CNN plus GRU classifier."""

    def __init__(self, feature_count: int = 6, hidden_size: int = 64, class_count: int = 3):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv1d(feature_count, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.recurrent = nn.GRU(32, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, class_count)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        features = self.convolution(x.transpose(1, 2)).transpose(1, 2)
        lengths = mask.sum(dim=1).to(torch.long).clamp_min(1)
        packed = nn.utils.rnn.pack_padded_sequence(
            features, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.recurrent(packed)
        return self.classifier(hidden[-1])


class CNNClassifier(nn.Module):
    def __init__(self, feature_count: int = 6, class_count: int = 3):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv1d(feature_count, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(32, class_count)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        features = self.convolution(x.transpose(1, 2)).transpose(1, 2)
        mask = mask.unsqueeze(-1)
        pooled = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.classifier(pooled)


class GRUClassifier(nn.Module):
    def __init__(self, feature_count: int = 6, hidden_size: int = 64, class_count: int = 3):
        super().__init__()
        self.recurrent = nn.GRU(feature_count, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, class_count)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).to(torch.long).clamp_min(1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.recurrent(packed)
        return self.classifier(hidden[-1])
