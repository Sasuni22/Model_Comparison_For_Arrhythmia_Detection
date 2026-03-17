"""
models/resnet/model.py
======================
1-D Residual Network (ResNet) for ECG arrhythmia classification.

Architecture follows He et al. (2016) adapted for 1-D signals:
  Stem → Stage1 × N_blocks → Stage2 × N_blocks → Stage3 × N_blocks
       → Global Avg Pool → FC head

Each ResidualBlock uses the pre-activation design (BN→ReLU→Conv)
with a projection shortcut when the channel count or stride changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """
    Pre-activation 1-D Residual Block.

      BN → ReLU → Conv1d → BN → ReLU → Conv1d
          +shortcut (projection if needed)
    """

    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = 3, stride: int = 1,
                 dropout_rate: float = 0.2):
        super().__init__()
        pad = kernel_size // 2

        self.bn1 = nn.BatchNorm1d(in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               stride=stride, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               stride=1, padding=pad, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

        # Shortcut projection when shape changes
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = out + self.shortcut(x)
        return out


# ---------------------------------------------------------------------------
# ResNet
# ---------------------------------------------------------------------------
class ResNet1D(nn.Module):
    """
    1-D ResNet for ECG beat classification.

    Parameters
    ----------
    num_classes   : Number of output classes
    input_length  : ECG segment length in samples
    base_filters  : Number of filters in the first ResNet stage (default 64)
    num_blocks    : Number of residual blocks per stage (default 3)
    dropout_rate  : Dropout inside residual blocks
    """

    def __init__(self, num_classes: int = 5,
                 input_length: int = 200,
                 base_filters: int = 64,
                 num_blocks: int = 3,
                 dropout_rate: float = 0.2):
        super().__init__()

        # ------------------------------------------------------------------ #
        # Stem
        # ------------------------------------------------------------------ #
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_filters, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # ------------------------------------------------------------------ #
        # Residual stages: filters double, time halves at each stage boundary
        # ------------------------------------------------------------------ #
        self.stage1 = self._make_stage(base_filters,     base_filters,
                                       num_blocks, stride=1, dropout_rate=dropout_rate)
        self.stage2 = self._make_stage(base_filters,     base_filters * 2,
                                       num_blocks, stride=2, dropout_rate=dropout_rate)
        self.stage3 = self._make_stage(base_filters * 2, base_filters * 4,
                                       num_blocks, stride=2, dropout_rate=dropout_rate)

        # ------------------------------------------------------------------ #
        # Head
        # ------------------------------------------------------------------ #
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_filters * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    # ---------------------------------------------------------------------- #
    @staticmethod
    def _make_stage(in_ch: int, out_ch: int,
                    num_blocks: int, stride: int,
                    dropout_rate: float) -> nn.Sequential:
        layers = [ResidualBlock(in_ch, out_ch, stride=stride,
                                dropout_rate=dropout_rate)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_ch, out_ch,
                                        dropout_rate=dropout_rate))
        return nn.Sequential(*layers)

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
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        return self.classifier(x)

    # ---------------------------------------------------------------------- #
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet1D(num_classes=5, input_length=200).to(device)
    dummy = torch.randn(16, 1, 200).to(device)
    out = model(dummy)
    print(f"Output shape : {out.shape}")
    print(f"Parameters   : {model.count_parameters():,}")