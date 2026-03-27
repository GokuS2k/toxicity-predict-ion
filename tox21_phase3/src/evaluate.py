#!/usr/bin/env python3
"""
Tox21 Phase 3 — Evaluation

Loads trained Uni-Mol checkpoints, generates predictions on the test split,
and computes per-endpoint metrics: AUROC, AUPRC, Balanced Accuracy.
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_CSV = PROJECT_ROOT / "data" / "tox21_prepared.csv"
EXP_DIR = PROJECT_ROOT / "exp"
RESULTS_DIR = PROJECT_ROOT / "results"

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


def evaluate_experiment(
    exp_name: str,
    exp_path: pathlib.Path,
    data_csv: pathlib.Path = DATA_CSV,
) -> pd.DataFrame | None:
    """
    Evaluate a trained Uni-Mol experiment.

    Uses MolPredict to load the checkpoint and generate predictions,
    then computes AUROC, AUPRC, and Balanced Accuracy per endpoint.

    Parameters
    ----------
    exp_name : str
        Identifier like 'unimol_v1_scaffold'
    exp_path : pathlib.Path
        Path to the experiment directory (saved by MolTrain)
    data_csv : pathlib.Path
        Path to the prepared data CSV

    Returns
    -------
    DataFrame with columns: endpoint, auroc, auprc, balanced_accuracy,
                            num_samples, num_positives
    """
    if not exp_path.exists():
        print(f"  [!] Experiment directory not found: {exp_path}")
        return None

    try:
        from unimol_tools import MolPredict
    except ImportError:
        print("  ERROR: unimol_tools not installed.")
        return None

    print(f"\n  Evaluating: {exp_name}")
    print(f"  Checkpoint: {exp_path}")

    # Load predictions
    clf = MolPredict(load_model=str(exp_path))
    predictions = clf.predict(data=str(data_csv))

    # Load original data to get ground truth labels
    df = pd.read_csv(data_csv)

    # unimol_tools returns predictions as a numpy array or DataFrame
    # Shape: (n_samples, n_tasks)
    if isinstance(predictions, pd.DataFrame):
        pred_probs = predictions.values
    else:
        pred_probs = np.array(predictions)

    # Compute metrics per endpoint
    rows = []
    for i, task in enumerate(TOX21_TASKS):
        y_true = df[task].values
        y_prob = pred_probs[:, i]

        # Mask for non-NaN labels
        mask = ~np.isnan(y_true)
        y_true_valid = y_true[mask]
        y_prob_valid = y_prob[mask]

        n_samples = int(mask.sum())
        n_pos = int((y_true_valid == 1).sum())

        if n_pos == 0 or n_pos == n_samples:
            print(f"    {task}: Skipped (no positive/negative samples in test set)")
            auroc = float("nan")
            auprc = float("nan")
            bal_acc = float("nan")
        else:
            auroc = roc_auc_score(y_true_valid, y_prob_valid)
            auprc = average_precision_score(y_true_valid, y_prob_valid)
            y_pred_binary = (y_prob_valid >= 0.5).astype(int)
            bal_acc = balanced_accuracy_score(y_true_valid, y_pred_binary)

        rows.append({
            "endpoint": task,
            "auroc": round(auroc, 4) if not np.isnan(auroc) else np.nan,
            "auprc": round(auprc, 4) if not np.isnan(auprc) else np.nan,
            "balanced_accuracy": round(bal_acc, 4) if not np.isnan(bal_acc) else np.nan,
            "num_samples": n_samples,
            "num_positives": n_pos,
        })
        if not np.isnan(auroc):
            print(f"    {task:<18} AUROC={auroc:.4f}  AUPRC={auprc:.4f}  BalAcc={bal_acc:.4f}  (n={n_samples}, pos={n_pos})")

    results_df = pd.DataFrame(rows)

    # Print mean
    mean_auroc = results_df["auroc"].mean()
    mean_auprc = results_df["auprc"].mean()
    mean_balacc = results_df["balanced_accuracy"].mean()
    print(f"\n    {'MEAN':<18} AUROC={mean_auroc:.4f}  AUPRC={mean_auprc:.4f}  BalAcc={mean_balacc:.4f}")

    return results_df


def save_results(results_df: pd.DataFrame, filename: str):
    """Save evaluation results to a CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / filename
    results_df.to_csv(out_path, index=False)
    print(f"    [OK] Saved: {out_path}")


def evaluate_all():
    """Evaluate all available Uni-Mol experiments."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    experiments = [
        ("unimol_v1_scaffold", EXP_DIR / "unimol_v1_scaffold"),
        ("unimol_v1_random",   EXP_DIR / "unimol_v1_random"),
        ("unimol_v2_scaffold", EXP_DIR / "unimol_v2_scaffold"),
        ("unimol_v2_random",   EXP_DIR / "unimol_v2_random"),
    ]

    evaluated = {}
    for exp_name, exp_path in experiments:
        results = evaluate_experiment(exp_name, exp_path)
        if results is not None:
            save_results(results, f"{exp_name}_results.csv")
            evaluated[exp_name] = results

    if not evaluated:
        print("\n  [!] No experiments found to evaluate.")
        print("    Run train_unimol_v1.py first.")
    else:
        print(f"\n  [OK] Evaluated {len(evaluated)} experiment(s).")

    return evaluated


if __name__ == "__main__":
    evaluate_all()
