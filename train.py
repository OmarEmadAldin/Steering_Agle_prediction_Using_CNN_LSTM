"""
train.py
Trains the CNN(frozen ImageNet)+LSTM steering model.

Usage:
    python train.py --csv driving_log.csv --img_dir data/images
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import SteeringSequenceDataset, make_train_val_split
from model import CNNLSTMSteering
from metrics import evaluate


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df, val_df = make_train_val_split(args.csv, val_fraction=0.15)
    train_df.to_csv("train_split.csv", index=False)
    val_df.to_csv("val_split.csv", index=False)

    train_ds = SteeringSequenceDataset(
        "train_split.csv", args.img_dir, seq_len=args.seq_len, train=True
    )
    val_ds = SteeringSequenceDataset(
        "val_split.csv", args.img_dir, seq_len=args.seq_len, train=False
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = CNNLSTMSteering(
        backbone=args.backbone,
        hidden_size=args.hidden_size,
        freeze_backbone=True,
        unfreeze_last_block=False,
    ).to(device)

    criterion = nn.MSELoss()
    # Only optimize params that require grad (frozen backbone params excluded)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    train_loss_history = []
    val_loss_history = []

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]", leave=False)
        for seqs, angles in train_bar:
            seqs, angles = seqs.to(device), angles.to(device)

            optimizer.zero_grad()
            preds = model(seqs)
            loss = criterion(preds, angles)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_bar.set_postfix(loss=f"{loss.item():.5f}")

        model.eval()
        val_losses = []
        all_preds, all_targets = [], []
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [val]", leave=False)
        with torch.no_grad():
            for seqs, angles in val_bar:
                seqs, angles = seqs.to(device), angles.to(device)
                preds = model(seqs)
                loss = criterion(preds, angles)
                val_losses.append(loss.item())
                all_preds.append(preds.cpu().numpy())
                all_targets.append(angles.cpu().numpy())
                val_bar.set_postfix(loss=f"{loss.item():.5f}")

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        m = evaluate(all_preds, all_targets, turn_threshold_deg=args.turn_threshold_deg)

        print(
            f"Epoch {epoch+1}/{args.epochs} | train_loss: {train_loss:.5f} | val_loss: {val_loss:.5f} | "
            f"MAE: {m['mae_deg']:.2f} deg | RMSE: {m['rmse_deg']:.2f} deg | "
            f"straight_MAE: {m['straight_mae_deg']:.2f} deg (n={m['num_straight']}) | "
            f"turn_MAE: {m['turn_mae_deg']:.2f} deg (n={m['num_turn']})"
        )

        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.out)
            print(f"  -> saved best model to {args.out}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping: no val_loss improvement for {args.patience} epochs "
                    f"(best: {best_val_loss:.5f})"
                )
                break

    print("Training complete. Best val loss:", best_val_loss)

    plt.figure(figsize=(7, 4))
    plt.plot(train_loss_history, label="train_loss")
    plt.plot(val_loss_history, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    print("Saved loss curve to loss_curve.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to driving_log.csv")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "mobilenet_v2"])
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--out", type=str, default="steering_model_best.pt")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs)")
    parser.add_argument("--min_delta", type=float, default=1e-5, help="Minimum val_loss improvement to reset patience")
    parser.add_argument("--turn_threshold_deg", type=float, default=5.0, help="Angle threshold (deg) separating 'straight' from 'turn' frames for metrics")
    args = parser.parse_args()

    train(args)