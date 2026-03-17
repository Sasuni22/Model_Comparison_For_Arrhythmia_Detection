"""
models/cnn_bilstm_attention/evaluate.py
========================================
Evaluate the saved CNN+BiLSTM+Attention checkpoint and visualise
the attention weights for a set of sample beats.

Usage
-----
    cd experiments/
    python models/cnn_bilstm_attention/evaluate.py \
        --signals data/MITBIH/processed/signals.npy \
        --labels  data/MITBIH/processed/labels.npy  \
        --ckpt    models/cnn_bilstm_attention/saved_model/best_model.pth
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.cnn_bilstm_attention.model import CNNBiLSTMAttention
from utils.data_loader import get_dataloaders
from utils.metrics import (
    compute_metrics, print_metrics,
    plot_confusion_matrix, plot_roc_curves,
)

SHORT_NAMES = ['N', 'S', 'V', 'F', 'Q']


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate CNN+BiLSTM+Attention')
    parser.add_argument('--signals',     type=str, default='data/MITBIH/processed/signals.npy')
    parser.add_argument('--labels',      type=str, default='data/MITBIH/processed/labels.npy')
    parser.add_argument('--ckpt',        type=str,
                        default='models/cnn_bilstm_attention/saved_model/best_model.pth')
    parser.add_argument('--batch_size',  type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--vis_samples', type=int, default=5,
                        help='Number of attention visualisation samples per class')
    return parser.parse_args()


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        all_probs.append(torch.softmax(logits, 1).cpu().numpy())
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(y.numpy())
    return (np.concatenate(all_labels),
            np.concatenate(all_preds),
            np.concatenate(all_probs, axis=0))


@torch.no_grad()
def visualise_attention(model, signals: np.ndarray, labels: np.ndarray,
                         num_classes: int, samples_per_class: int,
                         save_path: str, device: torch.device) -> None:
    """
    For each class, plot a random sample beat overlaid with the
    additive attention weights produced by the model.
    """
    model.eval()
    short = SHORT_NAMES[:num_classes]

    fig, axes = plt.subplots(num_classes, samples_per_class,
                              figsize=(4 * samples_per_class, 3 * num_classes))
    if num_classes == 1:
        axes = axes[np.newaxis, :]

    for cls in range(num_classes):
        idx = np.where(labels == cls)[0]
        chosen = np.random.choice(idx, min(samples_per_class, len(idx)), replace=False)

        for col, beat_idx in enumerate(chosen):
            raw = signals[beat_idx]
            x = torch.tensor(raw, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            _ = model(x)
            attn_weights = model.get_attention_weights().squeeze(0).cpu().numpy()

            ax = axes[cls, col]
            t = np.arange(len(raw))

            # Plot ECG signal
            ax.plot(t, raw, color='steelblue', linewidth=1.2, zorder=2, label='ECG')

            # Overlay attention as shaded region (upscale to signal length)
            attn_len = len(attn_weights)
            attn_up = np.interp(t, np.linspace(0, len(t) - 1, attn_len), attn_weights)
            attn_up = (attn_up - attn_up.min()) / (attn_up.max() - attn_up.min() + 1e-8)

            ax2 = ax.twinx()
            ax2.fill_between(t, 0, attn_up, color='tomato', alpha=0.35, zorder=1)
            ax2.set_ylim(0, 1.5)
            ax2.set_yticks([])

            ax.set_title(f'Class {short[cls]}', fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle('ECG Beats with Attention Weights (red shading)', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Attention visualisation saved → {save_path}")
    plt.show()
    plt.close()


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice : {device}")

    _, _, test_loader, seg_len, num_classes = get_dataloaders(
        signals_path=args.signals,
        labels_path=args.labels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    ckpt = torch.load(args.ckpt, map_location=device)
    model = CNNBiLSTMAttention(
        num_classes=ckpt.get('num_classes', num_classes),
        input_length=ckpt.get('seg_len', seg_len),
        lstm_hidden=ckpt.get('lstm_hidden', 128),
        lstm_layers=ckpt.get('lstm_layers', 2),
        num_heads=ckpt.get('num_heads', 8),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded checkpoint – epoch {ckpt['epoch']}  "
          f"(val_loss={ckpt['val_loss']:.4f})")

    # Standard test-set evaluation
    y_true, y_pred, y_prob = run_inference(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, num_classes=num_classes)

    print("\n=== CNN + BiLSTM + Attention – Test Set Results ===")
    print_metrics(metrics, num_classes=num_classes)

    save_dir = os.path.dirname(args.ckpt)

    plot_confusion_matrix(y_true, y_pred, num_classes=num_classes,
                          save_path=os.path.join(save_dir, 'confusion_matrix.png'),
                          title='CNN + BiLSTM + Attention')
    plot_roc_curves(y_true, y_prob, num_classes=num_classes,
                    save_path=os.path.join(save_dir, 'roc_curves.png'),
                    title='CNN + BiLSTM + Attention – ROC')

    # Attention visualisation on raw signals
    signals = np.load(args.signals)
    labels = np.load(args.labels)

    visualise_attention(
        model, signals, labels,
        num_classes=num_classes,
        samples_per_class=args.vis_samples,
        save_path=os.path.join(save_dir, 'attention_visualisation.png'),
        device=device,
    )

    print(f"Evaluation complete. Figures saved to → {save_dir}\n")


if __name__ == '__main__':
    main()