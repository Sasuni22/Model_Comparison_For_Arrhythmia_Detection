"""
models/cnn_baseline/model.py
============================
Simple 1-D CNN baseline for ECG arrhythmia classification.

Architecture
------------
  Conv1d(1, 32, k=7)  → BN → ReLU → MaxPool
  Conv1d(32, 64, k=5) → BN → ReLU → MaxPool
  Conv1d(64, 128, k=3)→ BN → ReLU → MaxPool
  Global Average Pool
  FC(128, 64)  → ReLU → Dropout
  FC(64, num_classes)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv1d → BatchNorm → ReLU → MaxPool block."""

    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int, pool_size: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNBaseline(nn.Module):
    """
    Baseline 1-D CNN for ECG beat classification.

    Parameters
    ----------
    num_classes   : Number of output classes (default 5 – AAMI)
    input_length  : Length of the input ECG segment (default 200)
    dropout_rate  : Dropout probability in the classifier head
    """

    def __init__(self, num_classes: int = 5,
                 input_length: int = 200,
                 dropout_rate: float = 0.5):
        super().__init__()

        # ------------------------------------------------------------------ #
        # Feature extractor
        # ------------------------------------------------------------------ #
        self.features = nn.Sequential(
            ConvBlock(1, 32, kernel_size=7, pool_size=2),   # L/2
            ConvBlock(32, 64, kernel_size=5, pool_size=2),  # L/4
            ConvBlock(64, 128, kernel_size=3, pool_size=2), # L/8
        )

        # Global average pooling collapses the time dimension → (B, 128)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # ------------------------------------------------------------------ #
        # Classifier head
        # ------------------------------------------------------------------ #
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )

    # ---------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, 1, L)

        Returns
        -------
        logits : Tensor, shape (B, num_classes)
        """
        x = self.features(x)   # (B, 128, L')
        x = self.gap(x)        # (B, 128, 1)
        logits = self.classifier(x)
        return logits

    # ---------------------------------------------------------------------- #
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNBaseline(num_classes=5, input_length=200).to(device)
    dummy = torch.randn(16, 1, 200).to(device)
    out = model(dummy)
    print(f"Output shape : {out.shape}")          # (16, 5)
    print(f"Parameters   : {model.count_parameters():,}")