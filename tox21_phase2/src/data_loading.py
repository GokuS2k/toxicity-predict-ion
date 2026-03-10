"""
Data loading for Tox21 Phase 2.

Downloads the Tox21 CSV from MoleculeNet, validates SMILES with RDKit,
and returns a clean DataFrame with the 12 toxicity endpoint columns.
"""

import pathlib
import urllib.request

import pandas as pd
from rdkit import Chem

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
TOX21_PATH = DATA_DIR / "tox21.csv.gz"

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

_SMILES_COLS = ["smiles", "SMILES", "Smiles", "canonical_smiles"]

# Phase 1 baseline results (random split, test set) — hard-coded for comparison table
RF_RANDOM_AUROC = {
    "NR-AR": 0.820,
    "NR-AR-LBD": 0.841,
    "NR-AhR": 0.896,
    "NR-Aromatase": 0.818,
    "NR-ER": 0.768,
    "NR-ER-LBD": 0.797,
    "NR-PPAR-gamma": 0.908,
    "SR-ARE": 0.776,
    "SR-ATAD5": 0.834,
    "SR-HSE": 0.690,
    "SR-MMP": 0.901,
    "SR-p53": 0.804,
}
RF_RANDOM_AUPRC = {
    "NR-AR": 0.380,
    "NR-AR-LBD": 0.422,
    "NR-AhR": 0.676,
    "NR-Aromatase": 0.488,
    "NR-ER": 0.472,
    "NR-ER-LBD": 0.398,
    "NR-PPAR-gamma": 0.440,
    "SR-ARE": 0.540,
    "SR-ATAD5": 0.440,
    "SR-HSE": 0.336,
    "SR-MMP": 0.803,
    "SR-p53": 0.450,
}


def _find_smiles_col(df: pd.DataFrame) -> str:
    for col in _SMILES_COLS:
        if col in df.columns:
            return col
    raise ValueError(f"No SMILES column found. Columns: {list(df.columns)}")


def download_tox21() -> pathlib.Path:
    """Download Tox21 CSV from MoleculeNet if not already cached."""
    if TOX21_PATH.exists():
        print(f"  Using cached dataset at {TOX21_PATH}")
        return TOX21_PATH

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    urls = [
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
        "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
    ]
    for url in urls:
        try:
            print(f"  Downloading from {url} …")
            urllib.request.urlretrieve(url, TOX21_PATH)
            print(f"  Saved to {TOX21_PATH}")
            return TOX21_PATH
        except Exception as exc:
            print(f"  Failed ({exc}), trying next URL …")
    raise RuntimeError("Could not download Tox21 dataset. Check network access.")


def load_tox21_df() -> tuple[pd.DataFrame, str]:
    """
    Load Tox21 CSV, validate SMILES with RDKit, drop invalid rows.

    Returns
    -------
    df         : DataFrame with columns [smiles_col] + TOX21_TASKS
    smiles_col : name of the SMILES column in df
    """
    download_tox21()
    raw_df = pd.read_csv(TOX21_PATH, compression="gzip")
    smiles_col = _find_smiles_col(raw_df)

    print(f"  Raw rows: {len(raw_df)}")

    # Validate each SMILES; drop any that RDKit cannot parse
    valid_mask = raw_df[smiles_col].apply(
        lambda s: Chem.MolFromSmiles(str(s)) is not None
    )
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} rows with invalid SMILES")

    df = raw_df[valid_mask].reset_index(drop=True)
    print(f"  Valid molecules: {len(df)}")

    # Report missingness per endpoint
    for task in TOX21_TASKS:
        if task not in df.columns:
            df[task] = float("nan")
        miss_pct = df[task].isna().mean() * 100
        n_pos = (df[task] == 1).sum()
        n_neg = (df[task] == 0).sum()
        print(f"    {task:<18} miss={miss_pct:.1f}%  pos={n_pos}  neg={n_neg}")

    return df, smiles_col
