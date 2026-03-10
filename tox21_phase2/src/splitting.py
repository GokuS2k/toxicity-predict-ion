"""
Train/val/test splitting strategies for Tox21.

Strategy 1: Random split  (stratified, 80/10/10)
Strategy 2: Scaffold split (Murcko, 80/10/10)
"""

import random
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split


# ── Random split ─────────────────────────────────────────────────────────────

def random_split(
    df: pd.DataFrame,
    smiles_col: str,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified random split on the first endpoint with sufficient coverage.

    Returns (train_df, val_df, test_df).
    """
    from data_loading import TOX21_TASKS

    n = len(df)
    indices = np.arange(n)

    # Build stratification from first well-covered task
    strat_labels = None
    for task in TOX21_TASKS:
        if task not in df.columns:
            continue
        col = df[task].values
        known = ~pd.isna(col)
        if known.sum() > 50 and col[known].sum() > 10:
            # Use 0/1 for known labels, 2 for missing — safe int cast after NaN check
            strat_labels = np.where(
                pd.isna(col),
                2,
                np.where(col == 1.0, 1, 0),
            )
            break

    idx_train, idx_tmp = train_test_split(
        indices,
        test_size=val_frac + test_frac,
        random_state=seed,
        stratify=strat_labels,
    )
    strat_tmp = strat_labels[idx_tmp] if strat_labels is not None else None
    idx_val, idx_test = train_test_split(
        idx_tmp,
        test_size=test_frac / (val_frac + test_frac),
        random_state=seed,
        stratify=strat_tmp,
    )

    return (
        df.iloc[idx_train].reset_index(drop=True),
        df.iloc[idx_val].reset_index(drop=True),
        df.iloc[idx_test].reset_index(drop=True),
    )


# ── Scaffold split ────────────────────────────────────────────────────────────

def _get_scaffold(smiles: str, generic: bool = True) -> str:
    """
    Compute the (generic) Murcko scaffold for a SMILES string.

    generic=True removes side-chain atom types, giving better grouping
    across heteroatom variants.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles  # fallback: use original SMILES as its own "scaffold"
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if generic:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        smi = Chem.MolToSmiles(scaffold)
        return smi if smi else smiles
    except Exception:
        return smiles


def scaffold_split(
    df: pd.DataFrame,
    smiles_col: str,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Murcko scaffold split.

    Algorithm:
    1. Compute generic Murcko scaffold for every molecule.
    2. Group molecule indices by scaffold.
    3. Sort scaffold groups largest→smallest (ensures large clusters go to train).
    4. Assign entire groups to train/val/test sequentially until size targets hit.

    Returns (train_df, val_df, test_df) — no scaffold appears in >1 split.
    """
    n = len(df)
    print(f"  Computing scaffolds for {n} molecules …")

    # Map: scaffold_smiles → list of row indices
    scaffold_to_indices: dict[str, List[int]] = defaultdict(list)
    for i, smi in enumerate(df[smiles_col]):
        scaffold = _get_scaffold(str(smi))
        scaffold_to_indices[scaffold].append(i)

    n_scaffolds = len(scaffold_to_indices)
    print(f"  Unique scaffolds: {n_scaffolds}")

    # Sort by group size (largest first), shuffle ties with seeded RNG
    rng = random.Random(seed)
    groups = list(scaffold_to_indices.values())
    groups.sort(key=lambda g: (-len(g), rng.random()))

    # Assign groups to splits
    train_cutoff = int(n * (1 - val_frac - test_frac))
    val_cutoff   = int(n * (1 - test_frac))

    train_idx: List[int] = []
    val_idx:   List[int] = []
    test_idx:  List[int] = []

    for group in groups:
        if len(train_idx) < train_cutoff:
            train_idx.extend(group)
        elif len(train_idx) + len(val_idx) < val_cutoff:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    print(
        f"  Scaffold split sizes — "
        f"train: {len(train_idx)} | val: {len(val_idx)} | test: {len(test_idx)}"
    )

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )
