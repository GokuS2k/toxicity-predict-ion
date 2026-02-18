"""
data_acquisition.py
-------------------
Downloads and inspects the Tox21 dataset from MoleculeNet (via direct URLs).

The Tox21 dataset contains ~7,831 compounds with binary labels across 12 toxicity
assays. Labels can be missing (NaN), which is handled in preprocessing.

Assay descriptions:
  NR-AR        Nuclear Receptor - Androgen Receptor
  NR-AR-LBD    Nuclear Receptor - Androgen Receptor Ligand Binding Domain
  NR-AhR       Nuclear Receptor - Aryl hydrocarbon Receptor
  NR-Aromatase Nuclear Receptor - Aromatase
  NR-ER        Nuclear Receptor - Estrogen Receptor
  NR-ER-LBD   Nuclear Receptor - Estrogen Receptor Ligand Binding Domain
  NR-PPAR-gamma Nuclear Receptor - Peroxisome Proliferator-Activated Receptor gamma
  SR-ARE       Stress Response - Antioxidant Response Element
  SR-ATAD5     Stress Response - ATAD5
  SR-HSE       Stress Response - Heat Shock Factor Response Element
  SR-MMP       Stress Response - Mitochondrial Membrane Potential
  SR-p53       Stress Response - p53
"""

import os
import requests
import pandas as pd
import numpy as np


# Tox21 data URLs from MoleculeNet / NCATS
TOX21_URLS = {
    "train": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
}

# Fallback: direct NCATS Tox21 challenge data
FALLBACK_URL = "https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_data_all&tp=sdf"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]


def download_tox21(force_download: bool = False) -> str:
    """
    Download the Tox21 dataset CSV from MoleculeNet S3.

    Returns the local file path to the downloaded (and cached) CSV.
    Uses gzip-compressed CSV for efficiency.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    local_path = os.path.join(DATA_DIR, "tox21.csv.gz")

    if os.path.exists(local_path) and not force_download:
        print(f"[INFO] Dataset already cached at: {local_path}")
        return local_path

    url = TOX21_URLS["train"]
    print(f"[INFO] Downloading Tox21 dataset from:\n  {url}")

    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total_bytes = 0
    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            total_bytes += len(chunk)

    print(f"[INFO] Downloaded {total_bytes / 1024:.1f} KB -> {local_path}")
    return local_path


def load_tox21(file_path: str | None = None) -> pd.DataFrame:
    """
    Load the Tox21 CSV into a DataFrame.

    Expected columns: smiles, mol_id, + 12 toxicity label columns.
    Returns a DataFrame with all columns; labels may contain NaN.
    """
    if file_path is None:
        file_path = download_tox21()

    df = pd.read_csv(file_path, compression="gzip")
    print(f"\n[INFO] Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def inspect_dataset(df: pd.DataFrame) -> None:
    """Print a comprehensive summary of the dataset structure and label distribution."""
    print("\n" + "=" * 60)
    print("DATASET INSPECTION SUMMARY")
    print("=" * 60)

    print(f"\nShape: {df.shape}")
    print(f"\nColumns:\n  {list(df.columns)}")

    # Identify SMILES column
    smiles_col = _find_smiles_column(df)
    if smiles_col:
        print(f"\nSMILES column detected: '{smiles_col}'")
        print(f"  Example SMILES: {df[smiles_col].iloc[0]}")

    # Label analysis
    print("\n--- Toxicity Label Distribution ---")
    label_cols = [c for c in df.columns if c in TOX21_TASKS]

    summary_rows = []
    for col in label_cols:
        total = len(df)
        missing = df[col].isna().sum()
        available = total - missing
        positives = (df[col] == 1).sum()
        negatives = (df[col] == 0).sum()
        pos_rate = positives / available * 100 if available > 0 else 0

        summary_rows.append({
            "Endpoint": col,
            "Total": total,
            "Missing": missing,
            "Available": available,
            "Positives": positives,
            "Negatives": negatives,
            "Pos_Rate%": round(pos_rate, 1),
        })
        print(
            f"  {col:<18} | Available: {available:5d} | "
            f"Pos: {positives:4d} ({pos_rate:5.1f}%) | "
            f"Neg: {negatives:5d} | Missing: {missing}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(DATA_DIR, "label_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[INFO] Label summary saved to: {summary_path}")

    print("\n--- Missing Value Counts ---")
    print(df[label_cols].isna().sum().to_string())


def _find_smiles_column(df: pd.DataFrame) -> str | None:
    """Heuristically identify the SMILES column in a DataFrame."""
    for candidate in ["smiles", "SMILES", "Smiles", "canonical_smiles"]:
        if candidate in df.columns:
            return candidate
    # Fallback: look for a column containing ring notation
    for col in df.columns:
        if df[col].dtype == object and df[col].str.contains(r"[cCnNoO]", na=False).mean() > 0.5:
            return col
    return None


if __name__ == "__main__":
    file_path = download_tox21()
    df = load_tox21(file_path)
    inspect_dataset(df)
