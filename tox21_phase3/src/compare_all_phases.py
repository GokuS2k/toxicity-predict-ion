#!/usr/bin/env python3
"""
Tox21 Phase 3 — Cross-Phase Comparison

Loads Phase 1/2 hardcoded results and Phase 3 Uni-Mol results,
builds the master comparison table across all models and splits.

Output: results/all_phases_comparison.csv
"""

import pathlib
import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
PHASE2_CSV = PROJECT_ROOT / "phase2_results" / "phase2_results.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


def load_phase2_results() -> pd.DataFrame:
    """Load hardcoded Phase 1/2 AUROC results."""
    if not PHASE2_CSV.exists():
        raise FileNotFoundError(f"Phase 2 results not found: {PHASE2_CSV}")

    df = pd.read_csv(PHASE2_CSV)
    # Filter out the 'Mean' row for per-endpoint data
    df = df[df["endpoint"] != "Mean"].reset_index(drop=True)
    return df


def load_phase3_results() -> dict[str, pd.DataFrame]:
    """Load all available Phase 3 (Uni-Mol) result CSVs."""
    experiments = {}
    candidates = [
        "unimol_v1_scaffold_results.csv",
        "unimol_v1_random_results.csv",
        "unimol_v2_scaffold_results.csv",
        "unimol_v2_random_results.csv",
    ]
    for fname in candidates:
        fpath = RESULTS_DIR / fname
        if fpath.exists():
            key = fname.replace("_results.csv", "")
            experiments[key] = pd.read_csv(fpath)
            print(f"  Loaded: {fname}")
        else:
            print(f"  Not found (skipped): {fname}")
    return experiments


def build_master_comparison() -> pd.DataFrame:
    """
    Build the master comparison table with all phases.

    Returns DataFrame with columns:
        endpoint, rf_random, gatv2_random, gatv2_scaffold,
        dmpnn_random, dmpnn_scaffold,
        unimol_v1_random, unimol_v1_scaffold,
        [unimol_v2_random, unimol_v2_scaffold]
    Plus a Mean row.
    """
    print("\n  Building master comparison table …")

    # Phase 1/2 results
    p2 = load_phase2_results()
    master = pd.DataFrame({"endpoint": TOX21_TASKS})

    # Merge Phase 2 columns
    for col in ["rf_random", "gatv2_random", "gatv2_scaffold",
                 "dmpnn_random", "dmpnn_scaffold"]:
        if col in p2.columns:
            master = master.merge(
                p2[["endpoint", col]], on="endpoint", how="left"
            )

    # Phase 3 results
    p3 = load_phase3_results()
    for exp_name, exp_df in p3.items():
        col_name = exp_name  # e.g., 'unimol_v1_scaffold'
        exp_auroc = exp_df[["endpoint", "auroc"]].rename(
            columns={"auroc": col_name}
        )
        master = master.merge(exp_auroc, on="endpoint", how="left")

    # Add Mean row
    numeric_cols = [c for c in master.columns if c != "endpoint"]
    means = {"endpoint": "Mean"}
    for col in numeric_cols:
        means[col] = round(master[col].mean(), 4)
    mean_row = pd.DataFrame([means])
    master = pd.concat([master, mean_row], ignore_index=True)

    return master


def save_master_comparison(master_df: pd.DataFrame):
    """Save the master comparison table."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "all_phases_comparison.csv"
    master_df.to_csv(out_path, index=False)
    print(f"\n  [OK] Saved master comparison: {out_path}")

    # Pretty print
    print(f"\n{'='*100}")
    print("  MASTER COMPARISON TABLE — AUROC (All Phases)")
    print(f"{'='*100}")
    print(master_df.to_string(index=False))
    print()


def main():
    master = build_master_comparison()
    save_master_comparison(master)
    return master


if __name__ == "__main__":
    main()
