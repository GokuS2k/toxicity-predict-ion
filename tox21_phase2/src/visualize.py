"""
Visualization for Tox21 Phase 2.

Generates all required plots:
  1. auroc_comparison_barplot.png       — grouped bars for all 5 model configs
  2. random_vs_scaffold_delta.png       — AUROC drop from random → scaffold
  3. roc_curves_best_model.png          — 4x3 ROC curve grid (best scaffold model)
  4. confusion_matrices_best_model.png  — 4x3 confusion matrix grid (best scaffold model)
  5. training_curves_gatv2.png          — GATv2 loss & val AUROC per epoch
"""

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Clean color palette for 5 model configurations
COLORS = {
    "RF (random)":        "#636EFA",
    "GATv2 (random)":     "#EF553B",
    "GATv2 (scaffold)":   "#00CC96",
    "D-MPNN (random)":    "#AB63FA",
    "D-MPNN (scaffold)":  "#FFA15A",
}


# ── Plot 1: Grouped AUROC bar chart ──────────────────────────────────────────

def plot_auroc_comparison(auroc_df: pd.DataFrame):
    """
    Grouped bar chart: X = endpoint, Y = AUROC, grouped by model config.
    """
    from data_loading import TOX21_TASKS

    # Remove Mean row for plotting
    df = auroc_df[auroc_df["Endpoint"] != "Mean"].copy()
    tasks = TOX21_TASKS

    configs = [
        ("RF_Random_AUROC",       "RF (random)"),
        ("GATv2_Random_AUROC",    "GATv2 (random)"),
        ("GATv2_Scaffold_AUROC",  "GATv2 (scaffold)"),
        ("DMPNN_Random_AUROC",    "D-MPNN (random)"),
        ("DMPNN_Scaffold_AUROC",  "D-MPNN (scaffold)"),
    ]
    # Filter to columns that actually exist
    configs = [(col, lbl) for col, lbl in configs if col in df.columns]

    n_groups = len(tasks)
    n_bars   = len(configs)
    width    = 0.15
    x        = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(18, 6))

    for i, (col, label) in enumerate(configs):
        vals = df.set_index("Endpoint").reindex(tasks)[col].fillna(0).values
        offset = (i - n_bars / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=label,
               color=COLORS.get(label, f"C{i}"), edgecolor="white", linewidth=0.5, alpha=0.9)

    ax.axhline(0.5,  color="gray",   linewidth=1.0, linestyle=":",  label="Random baseline (0.50)")
    ax.axhline(0.65, color="orange", linewidth=1.2, linestyle="--", label="Target threshold (0.65)")

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=40, ha="right", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("Toxicity Endpoint", fontsize=11)
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_title("AUROC Comparison — All Models & Splitting Strategies", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = RESULTS_DIR / "auroc_comparison_barplot.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 2: Random vs scaffold AUROC delta ───────────────────────────────────

def plot_random_vs_scaffold_delta(auroc_df: pd.DataFrame):
    """
    Per-endpoint AUROC drop when moving from random → scaffold split.
    Shows which endpoints suffer most from data leakage under random splitting.
    """
    from data_loading import TOX21_TASKS

    df = auroc_df[auroc_df["Endpoint"] != "Mean"].set_index("Endpoint")
    tasks = TOX21_TASKS

    model_pairs = []
    if "GATv2_Random_AUROC" in df.columns and "GATv2_Scaffold_AUROC" in df.columns:
        model_pairs.append(("GATv2", "GATv2_Random_AUROC", "GATv2_Scaffold_AUROC"))
    if "DMPNN_Random_AUROC" in df.columns and "DMPNN_Scaffold_AUROC" in df.columns:
        model_pairs.append(("D-MPNN", "DMPNN_Random_AUROC", "DMPNN_Scaffold_AUROC"))

    if not model_pairs:
        print("  Warning: no scaffold results found, skipping delta plot.")
        return

    n_models = len(model_pairs)
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 5), sharey=False)
    if n_models == 1:
        axes = [axes]

    x = np.arange(len(tasks))
    for ax, (name, rand_col, scaf_col) in zip(axes, model_pairs):
        rand_vals = df.reindex(tasks)[rand_col].fillna(0).values
        scaf_vals = df.reindex(tasks)[scaf_col].fillna(0).values
        delta = rand_vals - scaf_vals  # positive = scaffold is harder

        colors = ["#e74c3c" if d > 0 else "#2ecc71" for d in delta]
        bars = ax.bar(x, delta, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Endpoint", fontsize=10)
        ax.set_ylabel("AUROC drop (random − scaffold)", fontsize=10)
        ax.set_title(f"{name}: Random vs Scaffold AUROC Gap", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, delta):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.003 if val >= 0 else -0.018),
                f"{val:+.3f}", ha="center", va="bottom", fontsize=7.5,
            )

        red_patch  = mpatches.Patch(color="#e74c3c", label="Scaffold harder")
        green_patch = mpatches.Patch(color="#2ecc71", label="Scaffold easier")
        ax.legend(handles=[red_patch, green_patch], fontsize=9)

    plt.suptitle("Data Leakage Effect: AUROC Drop from Random → Scaffold Split",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = RESULTS_DIR / "random_vs_scaffold_delta.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 3: ROC curves (best scaffold model) ─────────────────────────────────

def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    tasks: list[str],
):
    """4x3 ROC curve grid for all 12 endpoints."""
    n_tasks = len(tasks)
    cols = 4
    rows = (n_tasks + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = axes.flatten()

    for i, task in enumerate(tasks):
        ax = axes[i]
        known = ~np.isnan(y_true[:, i])
        yt = y_true[known, i]
        yp = y_prob[known, i]

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        if len(np.unique(yt)) >= 2:
            fpr, tpr, _ = roc_curve(yt, yp)
            auc_val = roc_auc_score(yt, yp)
            ax.plot(fpr, tpr, lw=1.8, color="#3498db", label=f"AUC = {auc_val:.3f}")
            ax.legend(fontsize=8)
        ax.set_title(task, fontsize=9, fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=7)
        ax.set_ylabel("True Positive Rate", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)

    for j in range(n_tasks, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"ROC Curves — {model_name} (Best Scaffold Model)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = RESULTS_DIR / "roc_curves_best_model.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 4: Confusion matrices (best scaffold model) ─────────────────────────

def plot_confusion_matrices(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    tasks: list[str],
):
    """Normalized 4x3 confusion matrix grid."""
    n_tasks = len(tasks)
    cols = 4
    rows = (n_tasks + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.2))
    axes = axes.flatten()

    for i, task in enumerate(tasks):
        ax = axes[i]
        known = ~np.isnan(y_true[:, i])
        yt = y_true[known, i].astype(int)
        yp = (y_prob[known, i] >= 0.5).astype(int)

        if len(np.unique(yt)) >= 2:
            cm = confusion_matrix(yt, yp, normalize="true")
            sns.heatmap(
                cm, annot=True, fmt=".2f", ax=ax,
                cmap="Blues", cbar=False, vmin=0, vmax=1,
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"],
            )
        else:
            ax.text(0.5, 0.5, "Insufficient\nClasses",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_title(task, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)

    for j in range(n_tasks, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Confusion Matrices (Normalized) — {model_name} (Best Scaffold Model)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = RESULTS_DIR / "confusion_matrices_best_model.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 5: Training curves ───────────────────────────────────────────────────

def plot_training_curves(
    history_random: list[dict],
    history_scaffold: list[dict],
):
    """
    Dual-panel training curve plot for GATv2:
    Left: training loss, Right: validation AUROC.
    Both random and scaffold split curves shown.
    """
    fig, (ax_loss, ax_auroc) = plt.subplots(1, 2, figsize=(14, 5))

    for history, label, color in [
        (history_random,   "Random split",   "#3498db"),
        (history_scaffold, "Scaffold split", "#e74c3c"),
    ]:
        if not history:
            continue
        epochs     = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_auroc  = [h["val_auroc"]  for h in history]

        ax_loss.plot(epochs, train_loss, color=color, lw=1.8, label=label)
        ax_auroc.plot(epochs, val_auroc, color=color, lw=1.8, label=label)

    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("Masked BCE Loss", fontsize=11)
    ax_loss.set_title("GATv2 Training Loss", fontsize=12, fontweight="bold")
    ax_loss.legend(fontsize=10)
    ax_loss.grid(alpha=0.3)

    ax_auroc.set_xlabel("Epoch", fontsize=11)
    ax_auroc.set_ylabel("Mean Validation AUROC", fontsize=11)
    ax_auroc.set_title("GATv2 Validation AUROC", fontsize=12, fontweight="bold")
    ax_auroc.axhline(0.821, color="green", linestyle="--", lw=1.2, alpha=0.7,
                     label="RF baseline (0.821)")
    ax_auroc.legend(fontsize=10)
    ax_auroc.grid(alpha=0.3)
    ax_auroc.set_ylim(0, 1.0)

    plt.suptitle("GATv2 Training Curves — Random vs Scaffold Split",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = RESULTS_DIR / "training_curves_gatv2.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
