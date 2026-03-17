"""
models/cnn_baseline/train.py
============================
Training script for the CNN Baseline model.
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Allow imports from the project root
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.cnn_baseline.model import CNNBaseline
from utils.data_loader import get_dataloaders
from utils.metrics import plot_training_curves


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description='Train CNN Baseline')
    parser.add_argument('--signals',    type=str, default='data/MITBIH/processed/signals.npy')
    parser.add_argument('--labels',     type=str, default='data/MITBIH/processed/labels.npy')
    parser.add_argument('--save_dir',   type=str, default='models/cnn_baseline/saved_model')
    parser.add_argument('--epochs',     type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--dropout',    type=float, default=0.5)
    parser.add_argument('--num_workers',type=int, default=2)
    parser.add_argument('--seed',       type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# One epoch helpers
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += X.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)

        running_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += X.size(0)

    return running_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    args = get_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice : {device}")

    # Data
    train_loader, val_loader, _, seg_len, num_classes = get_dataloaders(
        signals_path=args.signals,
        labels_path=args.labels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Model
    model = CNNBaseline(
        num_classes=num_classes,
        input_length=seg_len,
        dropout_rate=args.dropout
    ).to(device)

    print(f"Model parameters : {model.count_parameters():,}\n")

    # Loss, optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 🔧 FIXED: Removed verbose=True (for older PyTorch)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # Output directory
    os.makedirs(args.save_dir, exist_ok=True)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    best_val_loss = float('inf')
    best_epoch = 0

    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>9} | {'Val Acc':>8} | {'LR':>8} | {'Time':>6}")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        vl_loss, vl_acc = evaluate_epoch(
            model, val_loader, criterion, device
        )

        # Manual LR tracking (since verbose not supported)
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(vl_loss)
        new_lr = optimizer.param_groups[0]['lr']

        if new_lr != old_lr:
            print(f"Learning rate reduced to {new_lr}")

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        elapsed = time.time() - t0

        print(f"{epoch:>6} | {tr_loss:>10.4f} | {tr_acc*100:>8.2f}% | "
              f"{vl_loss:>9.4f} | {vl_acc*100:>7.2f}% | "
              f"{new_lr:>8.6f} | {elapsed:>5.1f}s")

        # Save best checkpoint
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_epoch = epoch

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': vl_loss,
                'val_acc': vl_acc,
                'num_classes': num_classes,
                'seg_len': seg_len,
            }, os.path.join(args.save_dir, 'best_model.pth'))

    print(f"\nBest epoch : {best_epoch}  (val_loss={best_val_loss:.4f})")

    # Save history
    history_path = os.path.join(args.save_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Plot curves
    plot_training_curves(
        history['train_loss'],
        history['val_loss'],
        history['train_acc'],
        history['val_acc'],
        save_path=os.path.join(args.save_dir, 'training_curves.png'),
        title='CNN Baseline'
    )

    print(f"\nTraining complete. Saved to → {args.save_dir}\n")


if __name__ == '__main__':
    main()