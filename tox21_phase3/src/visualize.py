#!/usr/bin/env python3
"""
Tox21 Phase 3 — Visualizations

Generates all comparison plots for the 1D → 2D → 3D toxicity prediction
progression. All plots are saved to results/.

Plots:
  1. all_phases_mean_auroc.png       — Bar chart of mean AUROC per model×split
  2. all_phases_heatmap.png          — Heatmap (endpoints × models)
  3. 3d_vs_2d_improvement.png        — Uni-Mol v1 minus D-MPNN (scaffold)
  4. binding_vs_stress_comparison.png — NR vs SR group comparison
  5. scaffold_drop_all_models.png    — Random−scaffold delta per model
  6. progression_chart.png           — 1D→2D→3D mean AUROC progression
"""

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

NR_TASKS = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
            "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma"]
SR_TASKS = ["SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]

# Display names and colors
MODEL_DISPLAY = {
    "rf_random": ("RF (Random)", "#718093"),
    "gatv2_random": ("GATv2 (Random)", "#e17055"),
    "gatv2_scaffold": ("GATv2 (Scaffold)", "#d63031"),
    "dmpnn_random": ("D-MPNN (Random)", "#0984e3"),
    "dmpnn_scaffold": ("D-MPNN (Scaffold)", "#0652DD"),
    "unimol_v1_random": ("Uni-Mol v1 (Random)", "#00b894"),
    "unimol_v1_scaffold": ("Uni-Mol v1 (Scaffold)", "#00695c"),
    "unimol_v2_random": ("Uni-Mol v2 (Random)", "#a29bfe"),
    "unimol_v2_scaffold": ("Uni-Mol v2 (Scaffold)", "#6c5ce7"),
}


def _load_comparison():
    """Load the master comparison CSV."""
    csv = RESULTS_DIR / "all_phases_comparison.csv"
    if not csv.exists():
        print(f"  ERROR: {csv} not found. Run compare_all_phases.py first.")
        sys.exit(1)
    return pd.read_csv(csv)


def _get_model_cols(df):
    """Return model columns present in the comparison table."""
    return [c for c in df.columns if c != "endpoint"]


def _savefig(fig, name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] Saved: {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Mean AUROC bar chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_mean_auroc_bar(df: pd.DataFrame):
    """Bar chart of mean AUROC across all models and splits — the hero chart."""
    mean_row = df[df["endpoint"] == "Mean"]
    if mean_row.empty:
        print("  [!] No 'Mean' row found in comparison table.")
        return
    cols = _get_model_cols(df)
    models = [c for c in cols if c in mean_row.columns and pd.notna(mean_row[c].values[0])]
    values = [float(mean_row[c].values[0]) for c in models]
    labels = [MODEL_DISPLAY.get(c, (c, "#555"))[0] for c in models]
    colors = [MODEL_DISPLAY.get(c, (c, "#555"))[1] for c in models]

    fig, ax = plt.subplots(figsize=(max(10, len(models)*1.2), 6))
    bars = ax.bar(range(len(models)), values, color=colors, edgecolor="white",
                  linewidth=0.8, width=0.7, zorder=3)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Mean AUROC", fontsize=12)
    ax.set_title("Mean AUROC Across All Models & Splits — Tox21", fontsize=14, fontweight="bold")
    ax.set_ylim(0.65, min(max(values) + 0.05, 1.0))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    _savefig(fig, "all_phases_mean_auroc.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_heatmap(df: pd.DataFrame):
    """Heatmap with rows = 12 endpoints, columns = all model×split combinations."""
    data = df[df["endpoint"] != "Mean"].copy()
    cols = _get_model_cols(df)
    avail_cols = [c for c in cols if c in data.columns]
    labels = [MODEL_DISPLAY.get(c, (c, "#555"))[0] for c in avail_cols]

    heatmap_data = data.set_index("endpoint")[avail_cols].astype(float)

    fig, ax = plt.subplots(figsize=(max(10, len(avail_cols)*1.5), 8))
    sns.heatmap(
        heatmap_data, annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=0.5, vmax=1.0, linewidths=0.5, linecolor="white",
        cbar_kws={"label": "AUROC", "shrink": 0.8},
        xticklabels=labels, ax=ax,
    )
    ax.set_title("AUROC Heatmap — All Models × All Endpoints", fontsize=14, fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.yticks(fontsize=10)
    fig.tight_layout()
    _savefig(fig, "all_phases_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: 3D vs 2D improvement (scaffold split)
# ─────────────────────────────────────────────────────────────────────────────
def plot_3d_vs_2d_improvement(df: pd.DataFrame):
    """Per-endpoint AUROC difference: Uni-Mol v1 - D-MPNN (scaffold split)."""
    data = df[df["endpoint"] != "Mean"].copy()
    if "unimol_v1_scaffold" not in data.columns or "dmpnn_scaffold" not in data.columns:
        print("  [!] Cannot plot 3D vs 2D: missing unimol_v1_scaffold or dmpnn_scaffold columns.")
        return

    data["delta"] = data["unimol_v1_scaffold"].astype(float) - data["dmpnn_scaffold"].astype(float)
    data = data.sort_values("delta", ascending=True)

    colors = ["#00b894" if d >= 0 else "#d63031" for d in data["delta"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(data)), data["delta"], color=colors,
                   edgecolor="white", linewidth=0.5, height=0.7)

    for i, (val, ep) in enumerate(zip(data["delta"], data["endpoint"])):
        offset = 0.003 if val >= 0 else -0.003
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, i, f"{val:+.4f}", ha=ha, va="center", fontsize=9)

    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data["endpoint"], fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xlabel("AUROC Difference (Uni-Mol v1 − D-MPNN)", fontsize=11)
    ax.set_title("3D vs 2D: Per-Endpoint Improvement (Scaffold Split)", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#00b894", label="3D better"),
        Patch(facecolor="#d63031", label="2D better"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    fig.tight_layout()
    _savefig(fig, "3d_vs_2d_improvement.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Binding vs Stress Response comparison
# ─────────────────────────────────────────────────────────────────────────────
def plot_binding_vs_stress(df: pd.DataFrame):
    """Mean AUROC for NR (binding) vs SR (stress) groups across scaffold-split models."""
    data = df[df["endpoint"] != "Mean"].copy()

    scaffold_cols = [c for c in _get_model_cols(df)
                     if "scaffold" in c and c in data.columns]
    # Also include rf_random since there's no scaffold version
    if "rf_random" in data.columns:
        scaffold_cols = ["rf_random"] + scaffold_cols

    if not scaffold_cols:
        print("  [!] No scaffold columns found for binding vs stress plot.")
        return

    nr_data = data[data["endpoint"].isin(NR_TASKS)]
    sr_data = data[data["endpoint"].isin(SR_TASKS)]

    models = [MODEL_DISPLAY.get(c, (c, "#555"))[0] for c in scaffold_cols]
    nr_means = [nr_data[c].astype(float).mean() for c in scaffold_cols]
    sr_means = [sr_data[c].astype(float).mean() for c in scaffold_cols]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(models)*1.5), 6))
    bars1 = ax.bar(x - width/2, nr_means, width, label="Nuclear Receptor (Binding)",
                   color="#0984e3", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, sr_means, width, label="Stress Response",
                   color="#e17055", edgecolor="white", linewidth=0.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Mean AUROC", fontsize=11)
    ax.set_title("Binding vs Stress Response — Scaffold Split Comparison", fontsize=13, fontweight="bold")
    ax.set_ylim(0.6, 1.0)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    _savefig(fig, "binding_vs_stress_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: Scaffold drop across models
# ─────────────────────────────────────────────────────────────────────────────
def plot_scaffold_drop(df: pd.DataFrame):
    """For each model, plot (random - scaffold) AUROC per endpoint."""
    data = df[df["endpoint"] != "Mean"].copy()

    # Identify model pairs (random, scaffold)
    pairs = []
    for base in ["gatv2", "dmpnn", "unimol_v1", "unimol_v2"]:
        rand_col = f"{base}_random"
        scaf_col = f"{base}_scaffold"
        if rand_col in data.columns and scaf_col in data.columns:
            display = base.replace("_", " ").upper()
            if base == "unimol_v1":
                display = "Uni-Mol v1"
            elif base == "unimol_v2":
                display = "Uni-Mol v2"
            elif base == "dmpnn":
                display = "D-MPNN"
            elif base == "gatv2":
                display = "GATv2"
            pairs.append((display, rand_col, scaf_col))

    if not pairs:
        print("  [!] No random/scaffold pairs found for scaffold drop plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(TOX21_TASKS))
    width = 0.8 / len(pairs)
    colors_cycle = ["#e17055", "#0984e3", "#00b894", "#6c5ce7"]

    for i, (label, rand_col, scaf_col) in enumerate(pairs):
        drops = (data[rand_col].astype(float) - data[scaf_col].astype(float)).values
        offset = (i - len(pairs)/2 + 0.5) * width
        bars = ax.bar(x + offset, drops, width, label=label,
                     color=colors_cycle[i % len(colors_cycle)],
                     edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(TOX21_TASKS, rotation=45, ha="right", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("AUROC Drop (Random − Scaffold)", fontsize=11)
    ax.set_title("Scaffold Split Robustness — Random vs Scaffold AUROC Gap", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    _savefig(fig, "scaffold_drop_all_models.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6: Progression chart (1D → 2D → 3D)
# ─────────────────────────────────────────────────────────────────────────────
def plot_progression(df: pd.DataFrame):
    """Mean scaffold-split AUROC progression: RF → GATv2 → D-MPNN → Uni-Mol."""
    mean_row = df[df["endpoint"] == "Mean"]
    if mean_row.empty:
        print("  [!] No Mean row found.")
        return

    # Build progression sequence
    progression = []

    # RF (random only — no scaffold version)
    if "rf_random" in mean_row.columns and pd.notna(mean_row["rf_random"].values[0]):
        progression.append(("RF\n(1D Fingerprints)", float(mean_row["rf_random"].values[0]), "#718093"))

    # GATv2 scaffold
    if "gatv2_scaffold" in mean_row.columns and pd.notna(mean_row["gatv2_scaffold"].values[0]):
        progression.append(("GATv2\n(2D Graph)", float(mean_row["gatv2_scaffold"].values[0]), "#d63031"))

    # D-MPNN scaffold
    if "dmpnn_scaffold" in mean_row.columns and pd.notna(mean_row["dmpnn_scaffold"].values[0]):
        progression.append(("D-MPNN\n(2D Graph)", float(mean_row["dmpnn_scaffold"].values[0]), "#0652DD"))

    # Uni-Mol v1 scaffold
    if "unimol_v1_scaffold" in mean_row.columns and pd.notna(mean_row["unimol_v1_scaffold"].values[0]):
        progression.append(("Uni-Mol v1\n(3D Pretrained)", float(mean_row["unimol_v1_scaffold"].values[0]), "#00695c"))

    # Uni-Mol v2 scaffold
    if "unimol_v2_scaffold" in mean_row.columns and pd.notna(mean_row["unimol_v2_scaffold"].values[0]):
        progression.append(("Uni-Mol v2\n(3D Pretrained)", float(mean_row["unimol_v2_scaffold"].values[0]), "#6c5ce7"))

    if len(progression) < 2:
        print("  [!] Not enough models for progression chart.")
        return

    labels, values, colors = zip(*progression)

    fig, ax = plt.subplots(figsize=(max(8, len(labels)*2), 6))
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="white",
                  linewidth=1, width=0.6, zorder=3)

    # Connecting line through bar tops
    ax.plot(range(len(labels)), values, color="#2d3436", linewidth=2,
            marker="o", markersize=8, zorder=4)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Dimension annotation arrows
    dims = ["1D", "2D", "2D", "3D", "3D"][:len(labels)]
    for i, dim in enumerate(dims):
        ax.text(i, 0.62, dim, ha="center", va="center", fontsize=14,
                fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.85))

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean AUROC (Scaffold Split)", fontsize=12)
    ax.set_title("Molecular Representation Progression: 1D → 2D → 3D",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0.58, min(max(values) + 0.06, 1.0))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    _savefig(fig, "progression_chart.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def generate_all_plots():
    """Generate all Phase 3 comparison visualizations."""
    print("\n" + "=" * 60)
    print("  Generating Phase 3 Visualizations")
    print("=" * 60)

    df = _load_comparison()
    print(f"  Loaded comparison table: {len(df)} rows × {len(df.columns)} columns")
    print(f"  Models: {_get_model_cols(df)}")

    plot_mean_auroc_bar(df)
    plot_heatmap(df)
    plot_3d_vs_2d_improvement(df)
    plot_binding_vs_stress(df)
    plot_scaffold_drop(df)
    plot_progression(df)

    print(f"\n  [OK] All visualizations saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    generate_all_plots()
