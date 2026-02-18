"""
preprocessing.py
----------------
Converts SMILES strings to Morgan fingerprints and prepares
train/validation/test splits for the Tox21 dataset.

Key decisions:
  - Morgan fingerprints (radius=2, 2048 bits): standard ECFP4 equivalent,
    widely used in cheminformatics and well-suited for Random Forests.
  - Missing labels (NaN): kept as NaN; each model is trained only on
    samples where that endpoint has a label (masked training).
  - Stratified split: uses the first non-all-missing label column for
    stratification; falls back to random split if stratification fails.
  - Invalid SMILES: logged and dropped — they cannot be featurized.
"""

import os
import logging
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from sklearn.model_selection import train_test_split

# Suppress RDKit C++ warnings; we handle errors ourselves
RDLogger.DisableLog("rdApp.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

FINGERPRINT_RADIUS = 2
FINGERPRINT_NBITS = 2048


def smiles_to_morgan(smiles: str, radius: int = FINGERPRINT_RADIUS,
                     n_bits: int = FINGERPRINT_NBITS) -> np.ndarray | None:
    """
    Convert a SMILES string to a Morgan fingerprint bit vector.

    Returns:
        numpy array of shape (n_bits,) with dtype uint8, or None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    # Convert RDKit ExplicitBitVect to numpy array efficiently
    arr = np.zeros(n_bits, dtype=np.uint8)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return arr


def featurize_smiles(smiles_series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a Series of SMILES to Morgan fingerprints.

    Returns:
        fingerprints : np.ndarray, shape (n_valid, FINGERPRINT_NBITS)
        valid_mask   : np.ndarray[bool], shape (n_total,)
            True where SMILES was successfully converted.
    """
    n = len(smiles_series)
    fingerprints = []
    valid_mask = np.zeros(n, dtype=bool)
    invalid_count = 0

    for i, smi in enumerate(smiles_series):
        fp = smiles_to_morgan(smi)
        if fp is not None:
            fingerprints.append(fp)
            valid_mask[i] = True
        else:
            invalid_count += 1
            logger.warning(f"Invalid SMILES at index {smiles_series.index[i]}: {smi!r}")

    if invalid_count > 0:
        logger.warning(f"Skipped {invalid_count}/{n} invalid SMILES strings.")

    X = np.vstack(fingerprints) if fingerprints else np.empty((0, FINGERPRINT_NBITS), dtype=np.uint8)
    logger.info(f"Featurized {len(fingerprints)}/{n} molecules -> shape {X.shape}")
    return X, valid_mask


def find_smiles_column(df: pd.DataFrame) -> str:
    """Return the name of the SMILES column in the DataFrame."""
    for candidate in ["smiles", "SMILES", "Smiles", "canonical_smiles"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"No SMILES column found. Available columns: {list(df.columns)}")


def prepare_data(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    random_state: int = 42,
) -> dict:
    """
    Full preprocessing pipeline: SMILES -> fingerprints -> train/val/test splits.

    Args:
        df           : Raw Tox21 DataFrame with SMILES and label columns.
        train_frac   : Fraction of data for training.
        val_frac     : Fraction of data for validation.
        test_frac    : Fraction of data for testing.
        random_state : Random seed for reproducibility.

    Returns:
        dict with keys: X_train, X_val, X_test, y_train, y_val, y_test,
                        smiles_train, smiles_val, smiles_test, task_names.
        y arrays have shape (n_samples, 12) with NaN for missing labels.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
        "Fractions must sum to 1.0"

    smiles_col = find_smiles_column(df)
    label_cols = [c for c in TOX21_TASKS if c in df.columns]

    logger.info(f"SMILES column: '{smiles_col}'")
    logger.info(f"Label columns found: {label_cols}")
    logger.info(f"Total rows: {len(df)}")

    # --- Step 1: Generate fingerprints ---
    logger.info("Generating Morgan fingerprints (radius=2, 2048 bits)...")
    X, valid_mask = featurize_smiles(df[smiles_col])

    # Filter DataFrame to valid molecules only
    df_valid = df[valid_mask].reset_index(drop=True)
    smiles_valid = df_valid[smiles_col].values

    # Extract label matrix — shape (n_valid, n_tasks), float to preserve NaN
    Y = df_valid[label_cols].values.astype(float)

    logger.info(f"Valid molecules: {len(df_valid)}")
    logger.info(f"Label matrix shape: {Y.shape}")
    logger.info(f"Missing label rate: {np.isnan(Y).mean():.1%}")

    # --- Step 2: Train/Val/Test split ---
    # We do an 80/20 first split, then split the 20% into val/test (50/50).
    # Stratification uses the first label column with enough positives.
    stratify_col = _get_stratify_column(Y, label_cols)

    # Primary split: train vs (val+test)
    val_test_frac = val_frac + test_frac
    indices = np.arange(len(X))

    if stratify_col is not None:
        logger.info(f"Stratified split using endpoint: '{stratify_col[1]}'")
        strat_labels = stratify_col[0]
        try:
            train_idx, valtest_idx = train_test_split(
                indices, test_size=val_test_frac,
                stratify=strat_labels, random_state=random_state
            )
        except ValueError:
            logger.warning("Stratified split failed, falling back to random split.")
            train_idx, valtest_idx = train_test_split(
                indices, test_size=val_test_frac, random_state=random_state
            )
    else:
        logger.warning("No suitable stratification column found. Using random split.")
        train_idx, valtest_idx = train_test_split(
            indices, test_size=val_test_frac, random_state=random_state
        )

    # Secondary split: val vs test (equal halves of the 20%)
    val_share = val_frac / val_test_frac  # e.g., 0.5
    val_idx, test_idx = train_test_split(
        valtest_idx, test_size=(1 - val_share), random_state=random_state
    )

    logger.info(
        f"Split sizes — Train: {len(train_idx)}, "
        f"Val: {len(val_idx)}, Test: {len(test_idx)}"
    )

    result = {
        "X_train": X[train_idx],
        "X_val": X[val_idx],
        "X_test": X[test_idx],
        "y_train": Y[train_idx],
        "y_val": Y[val_idx],
        "y_test": Y[test_idx],
        "smiles_train": smiles_valid[train_idx],
        "smiles_val": smiles_valid[val_idx],
        "smiles_test": smiles_valid[test_idx],
        "task_names": label_cols,
    }

    _log_split_summary(result)
    return result


def _get_stratify_column(Y: np.ndarray, label_cols: list) -> tuple | None:
    """
    Find the best column to use for stratified splitting.

    Picks the first column that:
      - Has no NaN values (or we fill NaN with -1 for stratification)
      - Has at least 10 positives
    Returns (labels_array, column_name) or None.
    """
    for i, col in enumerate(label_cols):
        col_vals = Y[:, i]
        known = ~np.isnan(col_vals)
        if known.sum() == len(col_vals):  # no missing
            positives = (col_vals == 1).sum()
            if positives >= 10:
                return col_vals.astype(int), col

    # Fallback: use the column with fewest missing values
    missing_counts = np.isnan(Y).sum(axis=0)
    best_i = np.argmin(missing_counts)
    col_vals = Y[:, best_i].copy()
    # Fill NaN with -1 for stratification purposes
    col_vals[np.isnan(col_vals)] = -1
    n_classes = len(np.unique(col_vals))
    if n_classes >= 2:
        return col_vals.astype(int), label_cols[best_i]

    return None


def _log_split_summary(data: dict) -> None:
    """Log per-task label counts for each split."""
    task_names = data["task_names"]
    print("\n--- Split Label Summary ---")
    header = f"{'Endpoint':<18} {'Train+':>7} {'Train-':>7} {'Val+':>6} {'Val-':>6} {'Test+':>6} {'Test-':>6}"
    print(header)
    print("-" * len(header))

    for i, task in enumerate(task_names):
        def counts(y, i):
            col = y[:, i]
            col = col[~np.isnan(col)]
            return int((col == 1).sum()), int((col == 0).sum())

        tp, tn = counts(data["y_train"], i)
        vp, vn = counts(data["y_val"], i)
        ep, en = counts(data["y_test"], i)
        print(f"  {task:<16} {tp:>7} {tn:>7} {vp:>6} {vn:>6} {ep:>6} {en:>6}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_acquisition import load_tox21

    df = load_tox21()
    data = prepare_data(df)
    print(f"\nX_train shape: {data['X_train'].shape}")
    print(f"y_train shape: {data['y_train'].shape}")
