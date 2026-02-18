"""
Full training pipeline for the molecular GNN on Tox21.

Usage:
    python src/train.py
    python src/train.py --epochs 100 --batch-size 64 --hidden-dim 128
"""

import argparse
import pathlib
import sys
import time

import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score

from dataset import load_raw_dataframe, build_dataset, split_dataset, TOX21_TASKS
from model import MolecularGNN, masked_bce_loss, compute_pos_weights
from evaluate import (
    collect_predictions,
    evaluate_predictions,
    save_metrics,
    plot_auroc_bar,
    plot_roc_curves,
    plot_confusion_matrices,
    print_metrics_table,
)

MODELS_DIR = pathlib.Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "tox21_gnn_model.pt"


# ── Training helpers ─────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, pos_weight, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = masked_bce_loss(logits, batch.y, pos_weight=pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def val_auroc(model, loader, device) -> float:
    """Mean AUROC across tasks with ≥2 label classes."""
    model.eval()
    y_true, y_prob = collect_predictions(model, loader, device)
    aurocs = []
    for i in range(len(TOX21_TASKS)):
        known = ~np.isnan(y_true[:, i])
        yt = y_true[known, i]
        yp = y_prob[known, i]
        if len(np.unique(yt)) >= 2:
            aurocs.append(roc_auc_score(yt, yp))
    return float(np.mean(aurocs)) if aurocs else 0.0


# ── Main pipeline ────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── 1. Load & featurize ────────────────────────────────────────────
    print("\n[1/6] Loading and featurizing Tox21 dataset …")
    df, smiles_col = load_raw_dataframe()
    print(f"  Loaded {len(df)} molecules from CSV")

    dataset, invalid = build_dataset(df, smiles_col, verbose=True)
    print(f"  Valid graphs: {len(dataset)} | Invalid SMILES: {len(invalid)}")

    # ── 2. Split ───────────────────────────────────────────────────────
    print("\n[2/6] Splitting into train / val / test (80 / 10 / 10) …")
    train_ds, val_ds, test_ds = split_dataset(dataset, seed=args.seed)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=0)

    # ── 3. Class weights ───────────────────────────────────────────────
    print("\n[3/6] Computing class weights …")
    pos_weight = compute_pos_weights(train_ds, num_tasks=len(TOX21_TASKS))
    pos_weight = pos_weight.to(device)
    for i, (t, w) in enumerate(zip(TOX21_TASKS, pos_weight.cpu().tolist())):
        print(f"  {t:<18} pos_weight={w:.2f}")

    # ── 4. Model & optimiser ───────────────────────────────────────────
    print("\n[4/6] Building GNN model …")
    model = MolecularGNN(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_tasks=len(TOX21_TASKS),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")
    print(f"  Architecture: {args.num_layers} GATv2 layers, "
          f"hidden={args.hidden_dim}, heads={args.num_heads}")

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                  patience=10, min_lr=1e-5)

    # ── 5. Training loop ───────────────────────────────────────────────
    print(f"\n[5/6] Training for up to {args.epochs} epochs "
          f"(early stop patience={args.patience}) …\n")

    best_val_auroc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, pos_weight, device)
        v_auroc = val_auroc(model, val_loader, device)
        scheduler.step(v_auroc)
        elapsed = time.time() - t0

        history.append({"epoch": epoch, "train_loss": train_loss, "val_auroc": v_auroc})

        improved = v_auroc > best_val_auroc
        if improved:
            best_val_auroc = v_auroc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "val_auroc": v_auroc,
            }, MODEL_PATH)
        else:
            patience_counter += 1

        marker = " *" if improved else ""
        print(f"  Epoch {epoch:03d}/{args.epochs}  "
              f"loss={train_loss:.4f}  val_auroc={v_auroc:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"[{elapsed:.1f}s]{marker}")

        if patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(best epoch {best_epoch}, val_auroc={best_val_auroc:.4f})")
            break

    # ── 6. Evaluate best model ─────────────────────────────────────────
    print(f"\n[6/6] Loading best model (epoch {best_epoch}) and evaluating …")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    for split, loader in [("validation", val_loader), ("test", test_loader)]:
        print(f"\n  ── {split.upper()} ──")
        y_true, y_prob = collect_predictions(model, loader, device)
        metrics_df = evaluate_predictions(y_true, y_prob)
        print_metrics_table(metrics_df, split)
        save_metrics(metrics_df, split)
        plot_auroc_bar(metrics_df, split)
        plot_roc_curves(y_true, y_prob, split)
        plot_confusion_matrices(y_true, y_prob, split)

    # ── Summary ────────────────────────────────────────────────────────
    import pandas as pd
    val_df  = pd.read_csv(pathlib.Path(__file__).parent.parent / "results" / "metrics_validation.csv")
    test_df = pd.read_csv(pathlib.Path(__file__).parent.parent / "results" / "metrics_test.csv")

    val_mean  = val_df["AUROC"].mean()
    test_mean = test_df["AUROC"].mean()
    val_above = (val_df["AUROC"] >= 0.65).sum()
    test_above = (test_df["AUROC"] >= 0.65).sum()

    print("\n" + "═"*55)
    print("  FINAL SUMMARY — GNN Toxicity Predictor")
    print("═"*55)
    print(f"  Model         : GATv2 ({args.num_layers} layers, h={args.hidden_dim})")
    print(f"  Best epoch    : {best_epoch} / {args.epochs}")
    print(f"  Val  AUROC    : {val_mean:.4f}  ({val_above}/12 ≥ 0.65)")
    print(f"  Test AUROC    : {test_mean:.4f}  ({test_above}/12 ≥ 0.65)")
    print(f"  Model saved   : {MODEL_PATH}")
    print("═"*55)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train GNN on Tox21")
    p.add_argument("--epochs",       type=int,   default=150)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--hidden-dim",   type=int,   default=128)
    p.add_argument("--num-heads",    type=int,   default=4)
    p.add_argument("--num-layers",   type=int,   default=3)
    p.add_argument("--dropout",      type=float, default=0.2)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience",     type=int,   default=25)
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args)
