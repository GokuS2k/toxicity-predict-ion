"""
evaluation.py
-------------
Evaluation metrics and reporting for the Tox21 toxicity model.

Metrics used:
  - AUROC: Area Under the ROC Curve (primary metric for toxicity prediction).
            Robust to class imbalance, widely used in cheminformatics.
  - AUPRC: Area Under the Precision-Recall Curve. More informative than
            AUROC when positive class is rare (as in many Tox21 endpoints).
  - Balanced accuracy: accuracy adjusted for class imbalance.
  - Confusion matrix: per-endpoint TP/FP/TN/FN breakdown.

Missing labels are handled by computing metrics only on samples with known
labels for each endpoint.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, balanced_accuracy_score,
    roc_curve, precision_recall_curve,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def evaluate_predictions(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    task_names: list[str],
    split_name: str = "validation",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compute per-task evaluation metrics.

    Args:
        y_true     : True labels, shape (n_samples, n_tasks), NaN = missing.
        y_proba    : Predicted probabilities, shape (n_samples, n_tasks).
        task_names : List of task/endpoint names.
        split_name : Label for logging (e.g., 'validation', 'test').
        threshold  : Probability threshold for binary predictions.

    Returns:
        DataFrame with per-task metrics: AUROC, AUPRC, balanced_acc,
        n_samples, n_positive, n_negative.
    """
    rows = []

    for i, task in enumerate(task_names):
        # Only evaluate on samples with known labels
        known = ~np.isnan(y_true[:, i]) & ~np.isnan(y_proba[:, i])
        y_t = y_true[known, i].astype(int)
        y_p = y_proba[known, i]
        y_pred = (y_p >= threshold).astype(int)

        n_samples = known.sum()
        n_pos = (y_t == 1).sum()
        n_neg = (y_t == 0).sum()

        if len(np.unique(y_t)) < 2:
            logger.warning(f"[{task}] Only one class present — AUROC/AUPRC undefined.")
            rows.append({
                "Endpoint": task,
                "AUROC": np.nan,
                "AUPRC": np.nan,
                "Balanced_Acc": np.nan,
                "N_samples": n_samples,
                "N_positive": n_pos,
                "N_negative": n_neg,
            })
            continue

        auroc = roc_auc_score(y_t, y_p)
        auprc = average_precision_score(y_t, y_p)
        bal_acc = balanced_accuracy_score(y_t, y_pred)

        rows.append({
            "Endpoint": task,
            "AUROC": round(auroc, 4),
            "AUPRC": round(auprc, 4),
            "Balanced_Acc": round(bal_acc, 4),
            "N_samples": n_samples,
            "N_positive": n_pos,
            "N_negative": n_neg,
        })

    metrics_df = pd.DataFrame(rows)

    # Print summary table
    print(f"\n{'=' * 70}")
    print(f"EVALUATION RESULTS ({split_name.upper()})")
    print(f"{'=' * 70}")
    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "  N/A ",
        )
    )

    valid_auroc = metrics_df["AUROC"].dropna()
    if len(valid_auroc) > 0:
        above_065 = (valid_auroc >= 0.65).sum()
        print(f"\nMean AUROC: {valid_auroc.mean():.4f}")
        print(f"Endpoints with AUROC >= 0.65: {above_065}/{len(valid_auroc)}")

    return metrics_df


def plot_auroc_bar(
    metrics_df: pd.DataFrame,
    split_name: str = "validation",
    save_path: str | None = None,
) -> str:
    """
    Bar chart of AUROC scores across all 12 Tox21 endpoints.

    Returns the path where the figure was saved.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, f"auroc_bar_{split_name}.png")

    df = metrics_df.copy().sort_values("AUROC", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#2ecc71" if v >= 0.65 else "#e74c3c"
              for v in df["AUROC"].fillna(0)]

    bars = ax.bar(df["Endpoint"], df["AUROC"], color=colors, edgecolor="white",
                  linewidth=0.8, zorder=3)

    # Value labels on bars
    for bar, val in zip(bars, df["AUROC"]):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold"
            )

    # Threshold line at 0.65
    ax.axhline(0.65, color="navy", linestyle="--", linewidth=1.5,
               label="Target AUROC (0.65)", zorder=4)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0,
               label="Random baseline (0.50)", zorder=4)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_xlabel("Toxicity Endpoint", fontsize=12)
    ax.set_title(
        f"Tox21 Random Forest — AUROC per Endpoint ({split_name.capitalize()} Set)",
        fontsize=13, fontweight="bold"
    )
    ax.set_xticklabels(df["Endpoint"], rotation=35, ha="right", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # Add green/red legend patches
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color="#2ecc71", label="AUROC ≥ 0.65"),
        Patch(color="#e74c3c", label="AUROC < 0.65"),
    ]
    ax.legend(handles=legend_patches + ax.get_legend_handles_labels()[0][2:],
              labels=["AUROC ≥ 0.65", "AUROC < 0.65"] + ax.get_legend_handles_labels()[1][2:],
              fontsize=9, loc="lower left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"AUROC bar chart saved to: {save_path}")
    return save_path


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    task_names: list[str],
    split_name: str = "validation",
    save_path: str | None = None,
) -> str:
    """
    Plot ROC curves for all tasks in a grid layout.

    Returns the path where the figure was saved.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, f"roc_curves_{split_name}.png")

    n_tasks = len(task_names)
    ncols = 4
    nrows = (n_tasks + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 4))
    axes = axes.flatten()

    for i, task in enumerate(task_names):
        ax = axes[i]
        known = ~np.isnan(y_true[:, i]) & ~np.isnan(y_proba[:, i])
        y_t = y_true[known, i].astype(int)
        y_p = y_proba[known, i]

        if len(np.unique(y_t)) < 2:
            ax.text(0.5, 0.5, "Single class\n(no ROC)",
                    ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.set_title(task, fontsize=10)
            continue

        fpr, tpr, _ = roc_curve(y_t, y_p)
        auc_val = roc_auc_score(y_t, y_p)

        ax.plot(fpr, tpr, color="#3498db", lw=2, label=f"AUC = {auc_val:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.fill_between(fpr, tpr, alpha=0.1, color="#3498db")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel("FPR", fontsize=8)
        ax.set_ylabel("TPR", fontsize=8)
        ax.set_title(task, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"ROC Curves — Tox21 Random Forest ({split_name.capitalize()} Set)",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC curves saved to: {save_path}")
    return save_path


def plot_confusion_matrices(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    task_names: list[str],
    split_name: str = "validation",
    threshold: float = 0.5,
    save_path: str | None = None,
) -> str:
    """
    Plot normalized confusion matrices for all tasks in a grid layout.

    Returns the path where the figure was saved.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, f"confusion_matrices_{split_name}.png")

    n_tasks = len(task_names)
    ncols = 4
    nrows = (n_tasks + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 4))
    axes = axes.flatten()

    for i, task in enumerate(task_names):
        ax = axes[i]
        known = ~np.isnan(y_true[:, i]) & ~np.isnan(y_proba[:, i])
        y_t = y_true[known, i].astype(int)
        y_pred = (y_proba[known, i] >= threshold).astype(int)

        if len(np.unique(y_t)) < 2:
            ax.text(0.5, 0.5, "Single class", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_title(task, fontsize=10)
            continue

        cm = confusion_matrix(y_t, y_pred, normalize="true")
        sns.heatmap(
            cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
            ax=ax, cbar=False, annot_kws={"size": 10}
        )
        ax.set_title(task, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Confusion Matrices (normalized) — {split_name.capitalize()} Set",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrices saved to: {save_path}")
    return save_path


def save_metrics(metrics_df: pd.DataFrame, split_name: str = "validation") -> str:
    """Save the metrics DataFrame to a CSV file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"metrics_{split_name}.csv")
    metrics_df.to_csv(path, index=False)
    logger.info(f"Metrics saved to: {path}")
    return path
