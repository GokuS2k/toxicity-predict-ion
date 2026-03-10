"""
Evaluation: per-endpoint metrics, master comparison table.

Metrics computed:
  - AUROC           (primary)
  - AUPRC           (secondary)
  - Balanced Accuracy at threshold 0.5
"""

import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
)

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Per-endpoint metric computation ─────────────────────────────────────────

def evaluate_predictions(
    y_true: np.ndarray,  # [N, 12]  (NaN where missing)
    y_prob: np.ndarray,  # [N, 12]  predicted probabilities
    tasks: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute per-task AUROC, AUPRC, and balanced accuracy.

    Only samples where the label is not NaN are used for each endpoint.

    Returns a DataFrame with columns:
      Endpoint, AUROC, AUPRC, Balanced_Accuracy, Test_Samples, Test_Positives
    """
    if tasks is None:
        from data_loading import TOX21_TASKS
        tasks = TOX21_TASKS

    rows = []
    for i, task in enumerate(tasks):
        known = ~np.isnan(y_true[:, i])
        yt = y_true[known, i]
        yp = y_prob[known, i]
        n_pos = int(yt.sum())
        n_samples = int(known.sum())

        if len(np.unique(yt)) < 2:
            rows.append({
                "Endpoint": task,
                "AUROC": np.nan, "AUPRC": np.nan, "Balanced_Accuracy": np.nan,
                "Test_Samples": n_samples, "Test_Positives": n_pos,
            })
            continue

        auroc   = round(roc_auc_score(yt, yp), 4)
        auprc   = round(average_precision_score(yt, yp), 4)
        bal_acc = round(balanced_accuracy_score(yt, (yp >= 0.5).astype(int)), 4)

        rows.append({
            "Endpoint": task,
            "AUROC": auroc, "AUPRC": auprc, "Balanced_Accuracy": bal_acc,
            "Test_Samples": n_samples, "Test_Positives": n_pos,
        })

    return pd.DataFrame(rows)


def save_results(df: pd.DataFrame, filename: str) -> pathlib.Path:
    """Save per-endpoint results CSV to results/ directory."""
    path = RESULTS_DIR / filename
    df.to_csv(path, index=False)
    print(f"  Saved {path}")
    return path


def print_results_table(df: pd.DataFrame, title: str = ""):
    """Pretty-print per-endpoint metrics to console."""
    print(f"\n{'─'*70}")
    if title:
        print(f"  {title}")
    print(f"{'─'*70}")
    print(f"  {'Endpoint':<18} {'AUROC':>7}  {'AUPRC':>7}  {'Bal.Acc':>8}  {'N':>5}  {'Pos':>5}")
    print(f"{'─'*70}")
    for _, row in df.iterrows():
        auroc = f"{row['AUROC']:.4f}"   if pd.notna(row['AUROC'])   else "   N/A"
        auprc = f"{row['AUPRC']:.4f}"   if pd.notna(row['AUPRC'])   else "   N/A"
        bacc  = f"{row['Balanced_Accuracy']:.4f}" if pd.notna(row['Balanced_Accuracy']) else "   N/A"
        print(f"  {row['Endpoint']:<18} {auroc:>7}  {auprc:>7}  {bacc:>8}  "
              f"{row['Test_Samples']:>5}  {row['Test_Positives']:>5}")
    print(f"{'─'*70}")
    mean_auroc = df["AUROC"].mean()
    mean_auprc = df["AUPRC"].mean()
    print(f"  {'MEAN':<18} {mean_auroc:>7.4f}  {mean_auprc:>7.4f}")
    print(f"{'─'*70}\n")


# ── Master comparison table ───────────────────────────────────────────────────

def build_comparison_table(result_files: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the master AUROC and AUPRC comparison tables.

    Parameters
    ----------
    result_files : dict mapping column name → CSV filepath
        e.g. {
            "RF_Random": None,   # uses hard-coded Phase 1 values
            "GATv2_Random": "results_gatv2_random.csv",
            "GATv2_Scaffold": "results_gatv2_scaffold.csv",
            "DMPNN_Random": "results_dmpnn_random.csv",
            "DMPNN_Scaffold": "results_dmpnn_scaffold.csv",
        }

    Returns
    -------
    auroc_table : DataFrame with columns [Endpoint, RF_Random_AUROC, ...]
    auprc_table : DataFrame with columns [Endpoint, RF_Random_AUPRC, ...]
    """
    from data_loading import TOX21_TASKS, RF_RANDOM_AUROC, RF_RANDOM_AUPRC

    tasks = TOX21_TASKS

    # Seed the comparison tables with RF baseline
    auroc_data = {
        "Endpoint": tasks,
        "RF_Random_AUROC": [RF_RANDOM_AUROC.get(t, np.nan) for t in tasks],
    }
    auprc_data = {
        "Endpoint": tasks,
        "RF_Random_AUPRC": [RF_RANDOM_AUPRC.get(t, np.nan) for t in tasks],
    }

    # Add each GNN model's results
    for col_name, csv_path in result_files.items():
        if csv_path is None:
            continue
        full_path = RESULTS_DIR / csv_path
        if not full_path.exists():
            print(f"  Warning: {full_path} not found, skipping column {col_name}")
            continue
        df = pd.read_csv(full_path)
        task_to_auroc = dict(zip(df["Endpoint"], df["AUROC"]))
        task_to_auprc = dict(zip(df["Endpoint"], df["AUPRC"]))
        auroc_data[f"{col_name}_AUROC"] = [task_to_auroc.get(t, np.nan) for t in tasks]
        auprc_data[f"{col_name}_AUPRC"] = [task_to_auprc.get(t, np.nan) for t in tasks]

    auroc_df = pd.DataFrame(auroc_data)
    auprc_df = pd.DataFrame(auprc_data)

    # Add Mean row
    auroc_means = {"Endpoint": "Mean"}
    for col in auroc_df.columns:
        if col != "Endpoint":
            auroc_means[col] = round(auroc_df[col].mean(), 4)
    auroc_df = pd.concat([auroc_df, pd.DataFrame([auroc_means])], ignore_index=True)

    auprc_means = {"Endpoint": "Mean"}
    for col in auprc_df.columns:
        if col != "Endpoint":
            auprc_means[col] = round(auprc_df[col].mean(), 4)
    auprc_df = pd.concat([auprc_df, pd.DataFrame([auprc_means])], ignore_index=True)

    return auroc_df, auprc_df


def save_comparison_tables(auroc_df: pd.DataFrame, auprc_df: pd.DataFrame):
    """Save both comparison tables as CSV."""
    auroc_path = RESULTS_DIR / "comparison_all_models_auroc.csv"
    auprc_path = RESULTS_DIR / "comparison_all_models_auprc.csv"
    auroc_df.to_csv(auroc_path, index=False)
    auprc_df.to_csv(auprc_path, index=False)
    print(f"  Saved {auroc_path}")
    print(f"  Saved {auprc_path}")

    # Print AUROC table to console
    print("\n" + "═"*80)
    print("  MASTER AUROC COMPARISON TABLE")
    print("═"*80)
    print(auroc_df.to_string(index=False))
    print("═"*80 + "\n")
