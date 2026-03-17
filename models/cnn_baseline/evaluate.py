"""
models/cnn_baseline/evaluate.py
================================
Load the best checkpoint and evaluate on the held-out test set.

Usage
-----
    cd experiments/
    python models/cnn_baseline/evaluate.py \
        --signals  data/MITBIH/processed/signals.npy \
        --labels   data/MITBIH/processed/labels.npy  \
        --ckpt     models/cnn_baseline/saved_model/best_model.pth
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.cnn_baseline.model import CNNBaseline
from utils.data_loader import get_dataloaders
from utils.metrics import (
    compute_metrics, print_metrics,
    plot_confusion_matrix, plot_roc_curves
)


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate CNN Baseline')
    parser.add_argument('--signals',  type=str,
                        default='data/MITBIH/processed/signals.npy')
    parser.add_argument('--labels',   type=str,
                        default='data/MITBIH/processed/labels.npy')
    parser.add_argument('--ckpt',     type=str,
                        default='models/cnn_baseline/saved_model/best_model.pth')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=2)
    return parser.parse_args()


@torch.no_grad()
def run_inference(model, loader, device):
    """Collect all predictions, probabilities and ground-truth labels."""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.numpy())

    return (np.concatenate(all_labels),
            np.concatenate(all_preds),
            np.concatenate(all_probs, axis=0))


def main():
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice : {device}")

    # Load data (we only need the test_loader here)
    _, _, test_loader, seg_len, num_classes = get_dataloaders(
        signals_path=args.signals,
        labels_path=args.labels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    model = CNNBaseline(
        num_classes=ckpt.get('num_classes', num_classes),
        input_length=ckpt.get('seg_len', seg_len),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}  "
          f"(val_loss={ckpt['val_loss']:.4f})")

    # Inference
    y_true, y_pred, y_prob = run_inference(model, test_loader, device)

    # Metrics
    metrics = compute_metrics(y_true, y_pred, num_classes=num_classes)
    print("\n=== CNN Baseline – Test Set Results ===")
    print_metrics(metrics, num_classes=num_classes)

    # Save figures alongside the checkpoint
    save_dir = os.path.dirname(args.ckpt)

    plot_confusion_matrix(
        y_true, y_pred,
        num_classes=num_classes,
        save_path=os.path.join(save_dir, 'confusion_matrix.png'),
        title='CNN Baseline'
    )

    plot_roc_curves(
        y_true, y_prob,
        num_classes=num_classes,
        save_path=os.path.join(save_dir, 'roc_curves.png'),
        title='CNN Baseline – ROC'
    )

    print(f"Evaluation complete. Figures saved to → {save_dir}\n")


if __name__ == '__main__':
    main()