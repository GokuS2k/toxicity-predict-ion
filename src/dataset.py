"""
Tox21 graph dataset: loads the CSV, converts every SMILES to a
PyTorch Geometric Data object, and attaches the 12 toxicity labels.
"""

import pathlib
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from featurization import smiles_to_graph

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
TOX21_PATH = DATA_DIR / "tox21.csv.gz"

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

_POSSIBLE_SMILES_COLS = ["smiles", "SMILES", "Smiles", "canonical_smiles"]


def _find_smiles_column(df: pd.DataFrame) -> str:
    for col in _POSSIBLE_SMILES_COLS:
        if col in df.columns:
            return col
    raise ValueError(f"Cannot find SMILES column. Got: {list(df.columns)}")


def download_tox21() -> pathlib.Path:
    """Download Tox21 CSV if not already cached."""
    if TOX21_PATH.exists():
        return TOX21_PATH
    import urllib.request
    urls = [
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
        "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
    ]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for url in urls:
        try:
            print(f"Downloading Tox21 from {url} …")
            urllib.request.urlretrieve(url, TOX21_PATH)
            print(f"Saved to {TOX21_PATH}")
            return TOX21_PATH
        except Exception as e:
            print(f"  Failed: {e}")
    raise RuntimeError("Could not download Tox21 dataset.")


class Tox21GraphDataset(Dataset):
    """
    PyTorch Dataset wrapping the Tox21 molecular graph data.

    Each item is a PyG `Data` object augmented with:
      - `.y`       : float32 tensor [12] — NaN where label is missing
      - `.smiles`  : original SMILES string
    """

    def __init__(self, data_list: list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_raw_dataframe() -> tuple[pd.DataFrame, str]:
    """Load Tox21 CSV and return (df, smiles_col_name)."""
    download_tox21()
    df = pd.read_csv(TOX21_PATH, compression="gzip")
    smiles_col = _find_smiles_column(df)
    return df, smiles_col


def build_dataset(df: pd.DataFrame, smiles_col: str,
                  verbose: bool = True) -> tuple[Tox21GraphDataset, list[int]]:
    """
    Convert every row in df to a molecular graph.

    Returns:
        dataset      : Tox21GraphDataset
        invalid_idxs : row indices that could not be featurized
    """
    data_list = []
    invalid_idxs = []

    iterator = tqdm(df.iterrows(), total=len(df), desc="Featurizing") if verbose else df.iterrows()

    for row_idx, row in iterator:
        graph = smiles_to_graph(str(row[smiles_col]))
        if graph is None:
            invalid_idxs.append(row_idx)
            continue

        # 12 toxicity labels (NaN → float('nan'))
        labels = []
        for task in TOX21_TASKS:
            val = row.get(task, float("nan"))
            labels.append(float(val) if pd.notna(val) else float("nan"))

        # unsqueeze so PyG batching produces [B, 12] not [B*12]
        graph.y = torch.tensor(labels, dtype=torch.float).unsqueeze(0)
        graph.smiles = str(row[smiles_col])
        data_list.append(graph)

    return Tox21GraphDataset(data_list), invalid_idxs


def split_dataset(dataset: Tox21GraphDataset,
                  val_frac: float = 0.1,
                  test_frac: float = 0.1,
                  seed: int = 42) -> tuple[Tox21GraphDataset, Tox21GraphDataset, Tox21GraphDataset]:
    """Random train / val / test split (stratified by first task with enough labels)."""
    import numpy as np
    from sklearn.model_selection import train_test_split

    n = len(dataset)
    indices = np.arange(n)

    # Build stratification labels from the first task column with sufficient coverage
    y_all = torch.cat([dataset[i].y for i in range(n)], dim=0)  # [N, 12]
    strat_col = None
    for col in range(12):
        col_vals = y_all[:, col]
        known_mask = ~torch.isnan(col_vals)
        if known_mask.sum() > 50 and col_vals[known_mask].sum() > 10:
            strat_col = col
            break

    strat_labels = None
    if strat_col is not None:
        col_vals = y_all[:, strat_col].numpy()
        strat_labels = np.where(np.isnan(col_vals), 2, col_vals.astype(int))

    # First split: train vs (val + test)
    idx_train, idx_tmp = train_test_split(
        indices,
        test_size=val_frac + test_frac,
        random_state=seed,
        stratify=strat_labels,
    )

    # Second split: val vs test from the remainder
    strat_tmp = strat_labels[idx_tmp] if strat_labels is not None else None
    idx_val, idx_test = train_test_split(
        idx_tmp,
        test_size=test_frac / (val_frac + test_frac),
        random_state=seed,
        stratify=strat_tmp,
    )

    def _subset(idxs):
        return Tox21GraphDataset([dataset[i] for i in idxs])

    return _subset(idx_train), _subset(idx_val), _subset(idx_test)
