"""
models/cnn_bilstm_attention/model.py
=====================================
CNN + Bidirectional LSTM + Multi-Head Self-Attention model for
ECG arrhythmia classification.

Architecture
------------
  1. CNN Feature Extractor  (local morphological features)
       Conv1d(1,  32, k=7) → BN → ReLU → MaxPool(2)
       Conv1d(32, 64, k=5) → BN → ReLU → MaxPool(2)
       Conv1d(64,128, k=3) → BN → ReLU

  2. Bidirectional LSTM  (sequential / temporal context)
       BiLSTM(128 → 256 output)   ×2 layers

  3. Multi-Head Self-Attention  (global dependencies)
       PyTorch MultiheadAttention(embed_dim=256, num_heads=8)

  4. Classifier Head
       Weighted context vector → FC → ReLU → Dropout → FC → logits

The attention layer learns to focus on the most discriminative time-steps
of the BiLSTM output, producing a context vector that summarises the
entire sequence with learnt importance weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Additive (Bahdanau-style) attention – single head, lighter weight
# ---------------------------------------------------------------------------
class AdditiveAttention(nn.Module):
    """
    Bahdanau-style additive attention that produces a weighted sum over
    the time dimension.

    score(h_t) = v · tanh(W · h_t + b)
    α_t        = softmax(score(h_t))
    context    = Σ α_t · h_t
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor):
        """
        Parameters
        ----------
        lstm_out : Tensor, shape (B, T, H)

        Returns
        -------
        context  : Tensor, shape (B, H)
        weights  : Tensor, shape (B, T)   – for interpretability
        """
        # Energy scores
        energy = torch.tanh(self.W(lstm_out))  # (B, T, H)
        scores = self.v(energy).squeeze(-1)     # (B, T)

        # Normalised attention weights
        weights = torch.softmax(scores, dim=1)  # (B, T)

        # Weighted context vector
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # (B, H)
        return context, weights


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class CNNBiLSTMAttention(nn.Module):
    """
    CNN + Bidirectional LSTM + Attention for ECG beat classification.

    Parameters
    ----------
    num_classes   : Number of output classes (default 5)
    input_length  : ECG segment length in samples (default 200)
    cnn_filters   : Tuple of filter sizes for 3 CNN blocks
    lstm_hidden   : Hidden size per LSTM direction
    lstm_layers   : Number of stacked BiLSTM layers
    num_heads     : Number of heads in multi-head self-attention
    dropout_rate  : Dropout probability
    """

    def __init__(self, num_classes: int = 5,
                 input_length: int = 200,
                 cnn_filters: tuple = (32, 64, 128),
                 lstm_hidden: int = 128,
                 lstm_layers: int = 2,
                 num_heads: int = 8,
                 dropout_rate: float = 0.5):
        super().__init__()

        c1, c2, c3 = cnn_filters

        # ------------------------------------------------------------------ #
        # 1. CNN Feature Extractor
        # ------------------------------------------------------------------ #
        self.cnn = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(c1, c2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(c3),
            nn.ReLU(inplace=True),
        )
        # Output: (B, c3, L//4)

        # ------------------------------------------------------------------ #
        # 2. Bidirectional LSTM
        # ------------------------------------------------------------------ #
        self.bilstm = nn.LSTM(
            input_size=c3,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if lstm_layers > 1 else 0.0,
        )
        bilstm_out_dim = lstm_hidden * 2   # 256 by default

        # ------------------------------------------------------------------ #
        # 3a. Multi-Head Self-Attention (global sequence-level dependencies)
        # ------------------------------------------------------------------ #
        self.self_attn = nn.MultiheadAttention(
            embed_dim=bilstm_out_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(bilstm_out_dim)

        # 3b. Additive Attention (learnt time-step weighting → context vector)
        self.additive_attn = AdditiveAttention(hidden_dim=bilstm_out_dim)

        # ------------------------------------------------------------------ #
        # 4. Classifier
        # ------------------------------------------------------------------ #
        self.classifier = nn.Sequential(
            nn.Linear(bilstm_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

        # Store for external interpretability access
        self.attention_weights = None

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
        # --- CNN ---
        x = self.cnn(x)               # (B, C, T)
        x = x.permute(0, 2, 1)        # (B, T, C)

        # --- BiLSTM ---
        lstm_out, _ = self.bilstm(x)  # (B, T, 2*H)

        # --- Multi-head self-attention ---
        sa_out, _ = self.self_attn(lstm_out, lstm_out, lstm_out)
        lstm_out = self.attn_norm(lstm_out + sa_out)  # residual + norm

        # --- Additive attention → context ---
        context, self.attention_weights = self.additive_attn(lstm_out)
        # context : (B, 2*H)

        return self.classifier(context)

    # ---------------------------------------------------------------------- #
    def get_attention_weights(self) -> torch.Tensor | None:
        """Return last batch's additive attention weights for visualisation."""
        return self.attention_weights

    # ---------------------------------------------------------------------- #
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNBiLSTMAttention(num_classes=5, input_length=200).to(device)
    dummy = torch.randn(16, 1, 200).to(device)
    out = model(dummy)
    print(f"Output shape    : {out.shape}")
    print(f"Attention shape : {model.get_attention_weights().shape}")
    print(f"Parameters      : {model.count_parameters():,}")