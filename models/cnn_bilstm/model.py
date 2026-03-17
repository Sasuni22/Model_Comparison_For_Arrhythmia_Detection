"""
models/cnn_bilstm/model.py
==========================
Hybrid CNN + Bidirectional LSTM model for ECG arrhythmia classification.

Architecture
------------
  CNN Feature Extractor
    Conv1d(1,  32, k=7) → BN → ReLU → MaxPool(2)
    Conv1d(32, 64, k=5) → BN → ReLU → MaxPool(2)
    Conv1d(64,128, k=3) → BN → ReLU

  BiLSTM Temporal Modelling
    BiLSTM(128, hidden=128, layers=2)  → output: (B, T, 256)

  Classifier
    Last hidden state → FC(256, 64) → ReLU → Dropout → FC(64, num_classes)

The CNN captures local morphological features (QRS complex shape, etc.)
while the BiLSTM captures long-range temporal dependencies in both forward
and backward directions.
"""

import torch
import torch.nn as nn


class CNNBiLSTM(nn.Module):
    """
    Hybrid CNN + Bidirectional LSTM for ECG classification.

    Parameters
    ----------
    num_classes   : Number of output classes (default 5)
    input_length  : ECG segment length (default 200)
    cnn_filters   : Filter counts for 3 CNN stages
    lstm_hidden   : Hidden units per LSTM direction
    lstm_layers   : Number of stacked BiLSTM layers
    dropout_rate  : Dropout probability
    """

    def __init__(self, num_classes: int = 5,
                 input_length: int = 200,
                 cnn_filters: tuple = (32, 64, 128),
                 lstm_hidden: int = 128,
                 lstm_layers: int = 2,
                 dropout_rate: float = 0.5):
        super().__init__()

        c1, c2, c3 = cnn_filters

        # ------------------------------------------------------------------ #
        # CNN feature extractor
        # ------------------------------------------------------------------ #
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv1d(1, c1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Block 2
            nn.Conv1d(c1, c2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Block 3  (no pooling — preserve temporal resolution for LSTM)
            nn.Conv1d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(c3),
            nn.ReLU(inplace=True),
        )
        # After 2× MaxPool(2): time dim = input_length // 4

        # ------------------------------------------------------------------ #
        # Bidirectional LSTM
        # ------------------------------------------------------------------ #
        self.bilstm = nn.LSTM(
            input_size=c3,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,        # input shape: (B, T, features)
            bidirectional=True,
            dropout=dropout_rate if lstm_layers > 1 else 0.0,
        )

        # ------------------------------------------------------------------ #
        # Classifier head  (uses the final time-step hidden state)
        # ------------------------------------------------------------------ #
        bilstm_out_dim = lstm_hidden * 2   # × 2 because bidirectional

        self.classifier = nn.Sequential(
            nn.Linear(bilstm_out_dim, 64),
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
        # CNN: (B, 1, L) → (B, C, T)
        x = self.cnn(x)

        # Permute for LSTM: (B, C, T) → (B, T, C)
        x = x.permute(0, 2, 1)

        # BiLSTM: (B, T, C) → (B, T, 2*H)
        lstm_out, _ = self.bilstm(x)

        # Take the last time-step representation
        x = lstm_out[:, -1, :]    # (B, 2*H)

        return self.classifier(x)

    # ---------------------------------------------------------------------- #
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNBiLSTM(num_classes=5, input_length=200).to(device)
    dummy = torch.randn(16, 1, 200).to(device)
    out = model(dummy)
    print(f"Output shape : {out.shape}")
    print(f"Parameters   : {model.count_parameters():,}")