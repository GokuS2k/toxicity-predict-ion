"""
Evaluation: AUROC / AUPRC / balanced accuracy + plots (bar, ROC, confusion).
"""

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_curve,
    confusion_matrix,
)
from torch_geometric.loader import DataLoader

from dataset import TOX21_TASKS

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Metric computation ───────────────────────────────────────────────────────

def collect_predictions(model, loader: DataLoader, device: torch.device):
    """
    Run inference over `loader` and collect predictions and labels.

    Returns
    -------
    y_true : np.ndarray  [N, 12]  (NaN where label missing)
    y_prob : np.ndarray  [N, 12]  predicted probabilities
    """
    model.eval()
    all_true, all_prob = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = batch.y.cpu().numpy()
            all_true.append(labels)
            all_prob.append(probs)

    return np.vstack(all_true), np.vstack(all_prob)


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """
    Compute per-task metrics.

    Returns a DataFrame with columns:
      Endpoint, AUROC, AUPRC, Balanced_Acc, N_samples, N_positive, N_negative
    """
    rows = []
    for i, task in enumerate(TOX21_TASKS):
        known = ~np.isnan(y_true[:, i])
        yt = y_true[known, i]
        yp = y_prob[known, i]

        if len(np.unique(yt)) < 2:
            rows.append({
                "Endpoint": task, "AUROC": np.nan, "AUPRC": np.nan,
                "Balanced_Acc": np.nan,
                "N_samples": int(known.sum()), "N_positive": int(yt.sum()),
                "N_negative": int((1 - yt).sum()),
            })
            continue

        auroc = roc_auc_score(yt, yp)
        auprc = average_precision_score(yt, yp)
        bal_acc = balanced_accuracy_score(yt, (yp >= 0.5).astype(int))
        rows.append({
            "Endpoint": task,
            "AUROC": round(auroc, 4),
            "AUPRC": round(auprc, 4),
            "Balanced_Acc": round(bal_acc, 4),
            "N_samples": int(known.sum()),
            "N_positive": int(yt.sum()),
            "N_negative": int((1 - yt).sum()),
        })

    return pd.DataFrame(rows)


def save_metrics(df: pd.DataFrame, split: str):
    path = RESULTS_DIR / f"metrics_{split}.csv"
    df.to_csv(path, index=False)
    print(f"  Saved {path}")


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_auroc_bar(df: pd.DataFrame, split: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    aurocs = df["AUROC"].fillna(0).values
    colors = ["#2ecc71" if v >= 0.65 else "#e74c3c" for v in aurocs]
    bars = ax.bar(df["Endpoint"], aurocs, color=colors, edgecolor="white", linewidth=0.8)

    ax.axhline(0.65, color="#e67e22", linewidth=1.5, linestyle="--", label="Target (0.65)")
    ax.axhline(0.50, color="#95a5a6", linewidth=1.0, linestyle=":", label="Random (0.50)")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Toxicity Endpoint", fontsize=11)
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_title(f"AUROC by Endpoint — {split.capitalize()} Set  (GNN)", fontsize=13)
    plt.xticks(rotation=40, ha="right", fontsize=9)
    ax.legend()

    # Annotate bars
    for bar, val in zip(bars, aurocs):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

    mean_val = df["AUROC"].mean()
    ax.text(0.99, 0.97, f"Mean AUROC: {mean_val:.4f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))

    plt.tight_layout()
    path = RESULTS_DIR / f"auroc_bar_{split}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, split: str):
    n_tasks = len(TOX21_TASKS)
    cols = 4
    rows = (n_tasks + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = axes.flatten()

    for i, task in enumerate(TOX21_TASKS):
        ax = axes[i]
        known = ~np.isnan(y_true[:, i])
        yt = y_true[known, i]
        yp = y_prob[known, i]

        if len(np.unique(yt)) >= 2:
            fpr, tpr, _ = roc_curve(yt, yp)
            auc = roc_auc_score(yt, yp)
            ax.plot(fpr, tpr, lw=1.8, color="#3498db", label=f"AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(task, fontsize=9)
        ax.set_xlabel("FPR", fontsize=8)
        ax.set_ylabel("TPR", fontsize=8)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(n_tasks, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"ROC Curves — {split.capitalize()} Set  (GNN)", fontsize=13)
    plt.tight_layout()
    path = RESULTS_DIR / f"roc_curves_{split}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_confusion_matrices(y_true: np.ndarray, y_prob: np.ndarray, split: str):
    import seaborn as sns
    n_tasks = len(TOX21_TASKS)
    cols = 4
    rows = (n_tasks + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3))
    axes = axes.flatten()

    for i, task in enumerate(TOX21_TASKS):
        ax = axes[i]
        known = ~np.isnan(y_true[:, i])
        yt = y_true[known, i].astype(int)
        yp = (y_prob[known, i] >= 0.5).astype(int)

        if len(np.unique(yt)) >= 2:
            cm = confusion_matrix(yt, yp, normalize="true")
            sns.heatmap(cm, annot=True, fmt=".2f", ax=ax, cmap="Blues",
                        cbar=False, vmin=0, vmax=1,
                        xticklabels=["Pred 0", "Pred 1"],
                        yticklabels=["True 0", "True 1"])
        ax.set_title(task, fontsize=9)
        ax.tick_params(labelsize=7)

    for j in range(n_tasks, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Confusion Matrices (normalised) — {split.capitalize()} Set  (GNN)", fontsize=12)
    plt.tight_layout()
    path = RESULTS_DIR / f"confusion_matrices_{split}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def print_metrics_table(df: pd.DataFrame, split: str):
    print(f"\n{'─'*65}")
    print(f"  Results on {split.upper()} set")
    print(f"{'─'*65}")
    print(f"  {'Endpoint':<18} {'AUROC':>7}  {'AUPRC':>7}  {'Bal.Acc':>8}  {'N':>5}")
    print(f"{'─'*65}")
    for _, row in df.iterrows():
        auroc = f"{row['AUROC']:.4f}" if pd.notna(row['AUROC']) else "  N/A "
        auprc = f"{row['AUPRC']:.4f}" if pd.notna(row['AUPRC']) else "  N/A "
        bacc  = f"{row['Balanced_Acc']:.4f}" if pd.notna(row['Balanced_Acc']) else "  N/A "
        print(f"  {row['Endpoint']:<18} {auroc:>7}  {auprc:>7}  {bacc:>8}  {row['N_samples']:>5}")
    print(f"{'─'*65}")
    mean_auroc = df["AUROC"].mean()
    mean_auprc = df["AUPRC"].mean()
    print(f"  {'MEAN':<18} {mean_auroc:>7.4f}  {mean_auprc:>7.4f}")
    print(f"{'─'*65}\n")
