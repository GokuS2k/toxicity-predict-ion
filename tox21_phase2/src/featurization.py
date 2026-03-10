"""
Molecular graph featurization: SMILES → PyG Data objects.

Reuses the same atom/bond feature scheme from Phase 1, which produces:
  Node features : 75 dimensions
  Edge features : 12 dimensions (bidirectional)

These dimensions are used by GATv2Conv(edge_dim=12).
"""

import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdchem

# ── Atom feature constants ───────────────────────────────────────────────────

ATOM_TYPES = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca",
    "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag",
    "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni",
    "Cd", "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb",
]  # 43 elements + "other" bucket = 44

HYBRIDIZATION_TYPES = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]  # 5 types + "other" = 6

CHIRALITY_TYPES = [
    rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]  # 2 types + "other" = 3

BOND_TYPES = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]

STEREO_TYPES = [
    rdchem.BondStereo.STEREONONE,
    rdchem.BondStereo.STEREOANY,
    rdchem.BondStereo.STEREOZ,
    rdchem.BondStereo.STEREOE,
]

NUM_NODE_FEATURES = 75   # 44 + 6 + 3 + 10 + 7 + 1 + 1 + 1 + 1 = 74... wait, let's count:
# 44 (atom type) + 6 (hybridization) + 3 (chirality) + 10 (H count: 0-8+other=10)
# + 8 (degree: 0-6+other=8) + 1 (formal charge) + 1 (aromatic) + 1 (in ring) + 1 (radicals) = 75
NUM_EDGE_FEATURES = 12   # 5 (bond type: 4+other) + 5 (stereo: 4+other) + 1 + 1 = 12
# Actually bond type has 4 items (SINGLE/DOUBLE/TRIPLE/AROMATIC) + other = 5
# Stereo has 4 items + other = 5; is_conjugated + is_in_ring = 2 → total 12


def _one_hot(value, choices: list) -> list:
    """One-hot encoding with an 'other' bucket at the end."""
    enc = [0] * (len(choices) + 1)
    if value in choices:
        enc[choices.index(value)] = 1
    else:
        enc[-1] = 1
    return enc


def atom_features(atom: rdchem.Atom) -> np.ndarray:
    """75-dimensional atom feature vector."""
    feats = []
    feats += _one_hot(atom.GetSymbol(), ATOM_TYPES)                       # 44
    feats += _one_hot(atom.GetHybridization(), HYBRIDIZATION_TYPES)       # 6
    feats += _one_hot(atom.GetChiralTag(), CHIRALITY_TYPES)               # 3
    feats += _one_hot(int(atom.GetTotalNumHs()), list(range(9)))          # 10 (0-8 + other)
    feats += _one_hot(int(atom.GetDegree()), list(range(7)))              # 8  (0-6 + other)
    feats += [atom.GetFormalCharge() / 4.0]                               # 1
    feats += [float(atom.GetIsAromatic())]                                # 1
    feats += [float(atom.IsInRing())]                                     # 1
    feats += [min(atom.GetNumRadicalElectrons(), 4) / 4.0]               # 1
    return np.array(feats, dtype=np.float32)


def bond_features(bond: rdchem.Bond) -> np.ndarray:
    """12-dimensional bond feature vector."""
    feats = []
    feats += _one_hot(bond.GetBondType(), BOND_TYPES)    # 5 (4 + other)
    feats += _one_hot(bond.GetStereo(), STEREO_TYPES)    # 5 (4 + other)
    feats += [float(bond.GetIsConjugated())]             # 1
    feats += [float(bond.IsInRing())]                    # 1
    return np.array(feats, dtype=np.float32)


def smiles_to_graph(smiles: str) -> Data | None:
    """
    Convert SMILES to a PyTorch Geometric Data object.

    Returns None if parsing fails or molecule has no atoms.

    Graph format:
      data.x          : float32 [N, 75]
      data.edge_index : int64   [2, 2*B]   (bidirectional)
      data.edge_attr  : float32 [2*B, 12]
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    node_feats = [atom_features(a) for a in mol.GetAtoms()]
    if not node_feats:
        return None
    x = torch.tensor(np.stack(node_feats), dtype=torch.float)

    # Edge features (bidirectional)
    src_list, dst_list, edge_feat_list = [], [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat = bond_features(bond)
        src_list += [i, j]
        dst_list += [j, i]
        edge_feat_list += [feat, feat]

    if not src_list:
        # Single-atom molecule: add TWO self-loop edges so the total edge
        # count stays even. The D-MPNN uses reverse_idx = k ^ 1 (XOR-1),
        # which assumes edges come in directed pairs (k, k^1). A single
        # self-loop would make the batch edge count odd and cause an
        # IndexError when the last edge tries to access index == size.
        self_feat = np.zeros(NUM_EDGE_FEATURES, dtype=np.float32)
        src_list = [0, 0]
        dst_list = [0, 0]
        edge_feat_list = [self_feat, self_feat]

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(np.stack(edge_feat_list), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def build_graph_list(
    df,
    smiles_col: str,
    verbose: bool = True,
) -> tuple[list[Data], list[int]]:
    """
    Convert all rows in df to PyG Data objects.

    Returns (graph_list, invalid_indices).
    Labels are NOT attached here — they are added in the dataset wrapper.
    """
    from tqdm import tqdm
    from data_loading import TOX21_TASKS

    graph_list = []
    invalid = []

    iterator = (
        tqdm(df.iterrows(), total=len(df), desc="  Featurizing")
        if verbose else df.iterrows()
    )

    for row_idx, row in iterator:
        smi = str(row[smiles_col])
        g = smiles_to_graph(smi)
        if g is None:
            invalid.append(row_idx)
            continue

        # Attach labels: NaN preserved as float('nan')
        import pandas as _pd
        labels = [
            float(row[task]) if _pd.notna(row.get(task, float("nan"))) else float("nan")
            for task in TOX21_TASKS
        ]

        g.y = torch.tensor(labels, dtype=torch.float).unsqueeze(0)  # [1, 12]
        g.smiles = smi
        graph_list.append(g)

    return graph_list, invalid
