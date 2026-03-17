"""
models/resnet/train.py
======================
Training script for the 1-D ResNet model.

Usage
-----
    cd experiments/
    python models/resnet/train.py \
        --signals   data/MITBIH/processed/signals.npy \
        --labels    data/MITBIH/processed/labels.npy  \
        --epochs    60  --lr 0.001
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
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.resnet.model import ResNet1D
from utils.data_loader import get_dataloaders
from utils.metrics import compute_metrics, print_metrics, plot_training_curves


def get_args():
    parser = argparse.ArgumentParser(description='Train 1-D ResNet')
    parser.add_argument('--signals',      type=str, default='data/MITBIH/processed/signals.npy')
    parser.add_argument('--labels',       type=str, default='data/MITBIH/processed/labels.npy')
    parser.add_argument('--save_dir',     type=str, default='models/resnet/saved_model')
    parser.add_argument('--epochs',       type=int, default=60)
    parser.add_argument('--batch_size',   type=int, default=64)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--base_filters', type=int, default=64)
    parser.add_argument('--num_blocks',   type=int, default=3)
    parser.add_argument('--dropout',      type=float, default=0.2)
    parser.add_argument('--num_workers',  type=int, default=2)
    parser.add_argument('--seed',         type=int, default=42)
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        # Gradient clipping helps training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * X.size(0)
        correct += (model(X).argmax(1) == y).sum().item()
        total += X.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        running_loss += criterion(logits, y).item() * X.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += X.size(0)
    return running_loss / total, correct / total


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice : {device}")

    train_loader, val_loader, _, seg_len, num_classes = get_dataloaders(
        signals_path=args.signals,
        labels_path=args.labels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = ResNet1D(
        num_classes=num_classes,
        input_length=seg_len,
        base_filters=args.base_filters,
        num_blocks=args.num_blocks,
        dropout_rate=args.dropout,
    ).to(device)
    print(f"Model parameters : {model.count_parameters():,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    os.makedirs(args.save_dir, exist_ok=True)
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_epoch = 0

    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>9} | {'Val Acc':>8} | {'LR':>9} | {'Time':>6}")
    print("-" * 75)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        lr_now = optimizer.param_groups[0]['lr']
        print(f"{epoch:>6} | {tr_loss:>10.4f} | {tr_acc*100:>8.2f}% | "
              f"{vl_loss:>9.4f} | {vl_acc*100:>7.2f}% | {lr_now:>9.2e} | {time.time()-t0:>5.1f}s")

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

    with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    plot_training_curves(
        history['train_loss'], history['val_loss'],
        history['train_acc'], history['val_acc'],
        save_path=os.path.join(args.save_dir, 'training_curves.png'),
        title='1-D ResNet'
    )
    print(f"Training complete. Artefacts saved to → {args.save_dir}\n")


if __name__ == '__main__':
    main()