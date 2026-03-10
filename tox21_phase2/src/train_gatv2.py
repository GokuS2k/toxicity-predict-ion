"""
GATv2 training pipeline for Tox21 Phase 2.

Supports both random and scaffold splitting.
Saves best model checkpoint and training curves.
"""

import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score

# Add src directory to path when running standalone
SRC_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from data_loading import TOX21_TASKS
from models.gatv2 import GATv2Model, masked_bce_loss, compute_pos_weights

MODELS_DIR = pathlib.Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ── Training helpers ─────────────────────────────────────────────────────────

def _train_epoch(model, loader, optimizer, pos_weight, device):
    model.train()
    total_loss = 0.0
    n_batches  = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        targets = batch.y  # [B, 12]
        loss = masked_bce_loss(logits, targets, pos_weight=pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / max(n_batches, 1)


def _compute_val_auroc(model, loader, device) -> float:
    """Mean AUROC across tasks with at least 2 label classes present."""
    model.eval()
    all_true, all_prob = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            probs = torch.sigmoid(model(batch)).cpu().numpy()
            labels = batch.y.cpu().numpy()
            all_true.append(labels)
            all_prob.append(probs)

    y_true = np.vstack(all_true)
    y_prob = np.vstack(all_prob)

    aurocs = []
    for i in range(len(TOX21_TASKS)):
        known = ~np.isnan(y_true[:, i])
        yt = y_true[known, i]
        yp = y_prob[known, i]
        if len(np.unique(yt)) >= 2:
            try:
                aurocs.append(roc_auc_score(yt, yp))
            except Exception:
                pass
    return float(np.mean(aurocs)) if aurocs else 0.0


# ── Main training function ───────────────────────────────────────────────────

def train_gatv2(
    train_graphs: list,
    val_graphs: list,
    split_name: str,
    epochs: int = 100,
    batch_size: int = 64,
    hidden_dim: int = 128,
    num_heads: int = 8,
    num_layers: int = 3,
    dropout: float = 0.3,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 15,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[GATv2Model, list[dict]]:
    """
    Train GATv2 model.

    Parameters
    ----------
    train_graphs : list of PyG Data objects (training set)
    val_graphs   : list of PyG Data objects (validation set)
    split_name   : 'random' or 'scaffold' — used for checkpoint filename
    ...

    Returns
    -------
    model   : best GATv2Model (loaded from checkpoint)
    history : list of dicts with epoch, train_loss, val_auroc
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"  Device: {device}")

    # Dataloaders
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_graphs,   batch_size=batch_size, shuffle=False, num_workers=0)

    # Class weights from training set
    pos_weight = compute_pos_weights(train_graphs).to(device)

    # Model
    model = GATv2Model(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"  GATv2 parameters: {n_params:,}")

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                  patience=10, min_lr=1e-5)

    ckpt_path = MODELS_DIR / f"gatv2_{split_name}_best.pt"
    best_val_auroc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []

    if verbose:
        print(f"\n  Training GATv2 ({split_name} split) for up to {epochs} epochs …")
        print(f"  {'Epoch':>6}  {'Loss':>8}  {'Val AUROC':>10}  {'LR':>9}")
        print(f"  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*9}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = _train_epoch(model, train_loader, optimizer, pos_weight, device)
        v_auroc    = _compute_val_auroc(model, val_loader, device)
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
                "epoch": epoch,
                "val_auroc": v_auroc,
                "split": split_name,
            }, ckpt_path)
        else:
            patience_counter += 1

        if verbose:
            marker = " *" if improved else ""
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"  {epoch:>6}  {train_loss:>8.4f}  {v_auroc:>10.4f}  {cur_lr:>9.2e}{marker}")

        if patience_counter >= patience:
            if verbose:
                print(f"\n  Early stopping: epoch {epoch} "
                      f"(best epoch {best_epoch}, val_auroc={best_val_auroc:.4f})")
            break

    # Reload best checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if verbose:
        print(f"\n  Best epoch: {best_epoch}  |  Best val AUROC: {best_val_auroc:.4f}")
        print(f"  Checkpoint saved: {ckpt_path}")

    return model, history, device


def collect_predictions_gatv2(
    model: GATv2Model,
    graph_list: list,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference and collect (y_true, y_prob) arrays.

    Returns
    -------
    y_true : [N, 12] float array (NaN where missing)
    y_prob : [N, 12] float array (sigmoid probabilities)
    """
    loader = DataLoader(graph_list, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    all_true, all_prob = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            probs  = torch.sigmoid(model(batch)).cpu().numpy()
            labels = batch.y.cpu().numpy()
            all_true.append(labels)
            all_prob.append(probs)
    return np.vstack(all_true), np.vstack(all_prob)
