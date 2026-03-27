#!/usr/bin/env python3
"""
Tox21 Phase 3 — Data Preparation

Loads the Tox21 dataset, validates SMILES with RDKit, and exports a clean CSV
formatted for unimol_tools (smiles + 12 endpoint columns, NaN for missing).

Output: data/tox21_prepared.csv  (~7823 rows)
"""

import pathlib
import sys
import urllib.request

import pandas as pd
from rdkit import Chem

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_CSV = DATA_DIR / "tox21_prepared.csv"

# Try parent project first (shared data), fall back to local
PARENT_DATA = PROJECT_ROOT.parent / "data" / "tox21.csv.gz"
LOCAL_DATA = DATA_DIR / "tox21.csv.gz"

DOWNLOAD_URLS = [
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
    "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
]

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

_SMILES_COLS = ["smiles", "SMILES", "Smiles", "canonical_smiles"]


def _find_smiles_col(df: pd.DataFrame) -> str:
    for col in _SMILES_COLS:
        if col in df.columns:
            return col
    raise ValueError(f"No SMILES column found. Columns: {list(df.columns)}")


def _get_raw_csv() -> pathlib.Path:
    """Locate or download the Tox21 CSV."""
    if PARENT_DATA.exists():
        print(f"  Using shared dataset at {PARENT_DATA}")
        return PARENT_DATA
    if LOCAL_DATA.exists():
        print(f"  Using cached dataset at {LOCAL_DATA}")
        return LOCAL_DATA

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for url in DOWNLOAD_URLS:
        try:
            print(f"  Downloading from {url} …")
            urllib.request.urlretrieve(url, LOCAL_DATA)
            print(f"  Saved to {LOCAL_DATA}")
            return LOCAL_DATA
        except Exception as exc:
            print(f"  Failed ({exc}), trying next URL …")
    raise RuntimeError("Could not download Tox21 dataset. Check network access.")


def prepare_data() -> pd.DataFrame:
    """Load, validate, and export Tox21 data for unimol_tools."""
    print("=" * 60)
    print("  Tox21 Data Preparation for Phase 3 (Uni-Mol)")
    print("=" * 60)

    # Load raw CSV
    csv_path = _get_raw_csv()
    raw_df = pd.read_csv(csv_path, compression="gzip")
    smiles_col = _find_smiles_col(raw_df)
    print(f"\n  Raw rows: {len(raw_df)}")
    print(f"  SMILES column: '{smiles_col}'")

    # Validate SMILES with RDKit
    print("\n  Validating SMILES with RDKit …")
    valid_mask = raw_df[smiles_col].apply(
        lambda s: Chem.MolFromSmiles(str(s)) is not None
    )
    n_invalid = (~valid_mask).sum()
    print(f"  Invalid SMILES (dropped): {n_invalid}")

    df = raw_df[valid_mask].reset_index(drop=True)
    print(f"  Valid molecules: {len(df)}")

    # Build output DataFrame: smiles + 12 task columns
    out = pd.DataFrame()
    out["smiles"] = df[smiles_col].values

    for task in TOX21_TASKS:
        if task in df.columns:
            out[task] = df[task].values
        else:
            out[task] = float("nan")

    # Report statistics
    print(f"\n  {'Endpoint':<18} {'Miss%':>6}  {'Pos':>5}  {'Neg':>5}  {'Total':>6}")
    print(f"  {'-'*48}")
    for task in TOX21_TASKS:
        n_total = out[task].notna().sum()
        n_pos = (out[task] == 1).sum()
        n_neg = (out[task] == 0).sum()
        miss_pct = out[task].isna().mean() * 100
        print(f"  {task:<18} {miss_pct:>5.1f}%  {n_pos:>5}  {n_neg:>5}  {n_total:>6}")

    # Verify no accidental imputation — NaN must stay NaN
    total_nan = out[TOX21_TASKS].isna().sum().sum()
    print(f"\n  Total NaN values across all endpoints: {total_nan}")
    assert total_nan > 0, "ERROR: No missing values found — data may be imputed!"

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  [OK] Saved prepared CSV to {OUTPUT_CSV}")
    print(f"    Shape: {out.shape[0]} rows x {out.shape[1]} columns")

    return out


if __name__ == "__main__":
    df = prepare_data()
    print(f"\n  Done. {len(df)} molecules ready for Uni-Mol training.")
