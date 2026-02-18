"""
Molecular graph featurization: SMILES → PyG Data objects.

Atoms  → nodes  (73-dim feature vector)
Bonds  → edges  (10-dim feature vector, bidirectional)
"""

import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdchem


# ── Atom feature helpers ────────────────────────────────────────────────────

ATOM_TYPES = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca",
    "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag",
    "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni",
    "Cd", "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb",
]  # 43 elements + "other" = 44

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
]  # 2 types + "unspecified" = 3


def _one_hot(value, choices):
    """Return a one-hot list; last element = 'other' bucket."""
    enc = [0] * (len(choices) + 1)
    if value in choices:
        enc[choices.index(value)] = 1
    else:
        enc[-1] = 1
    return enc


def atom_features(atom: rdchem.Atom) -> np.ndarray:
    """
    Build a 75-dimensional feature vector for a single RDKit atom.

    Breakdown:
      44  atom type      (one-hot, 43 elements + other)
       6  hybridization  (one-hot, SP/SP2/SP3/SP3D/SP3D2 + other)
       3  chirality      (one-hot, CW / CCW / other)
      10  total num Hs   (one-hot, 0-8 + other)
       8  degree         (one-hot, 0-6 + other)
       1  formal charge  (raw, normalised by /4)
       1  is aromatic
       1  is in ring
       1  num radical electrons (raw, clipped /4)
    ─────
      75
    """
    feats = []
    feats += _one_hot(atom.GetSymbol(), ATOM_TYPES)                     # 44
    feats += _one_hot(atom.GetHybridization(), HYBRIDIZATION_TYPES)     # 6
    feats += _one_hot(atom.GetChiralTag(), CHIRALITY_TYPES)             # 3
    feats += _one_hot(int(atom.GetTotalNumHs()), list(range(9)))        # 9  (0-8 + other)
    feats += _one_hot(int(atom.GetDegree()), list(range(7)))            # 7  (0-6 + other)
    feats += [atom.GetFormalCharge() / 4.0]                             # 1
    feats += [float(atom.GetIsAromatic())]                              # 1
    feats += [float(atom.IsInRing())]                                   # 1
    feats += [min(atom.GetNumRadicalElectrons(), 4) / 4.0]              # 1
    return np.array(feats, dtype=np.float32)


# ── Bond feature helpers ────────────────────────────────────────────────────

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


def bond_features(bond: rdchem.Bond) -> np.ndarray:
    """
    Build a 12-dimensional feature vector for a single RDKit bond.

    Breakdown:
      5  bond type (one-hot: SINGLE/DOUBLE/TRIPLE/AROMATIC + other)
      5  stereo    (one-hot: NONE/ANY/Z/E + other)
      1  is conjugated
      1  is in ring
    ──
     12
    """
    feats = []
    feats += _one_hot(bond.GetBondType(), BOND_TYPES)    # 4 (no "other" – all bonds covered)
    feats += _one_hot(bond.GetStereo(), STEREO_TYPES)    # 4 (no "other")
    feats += [float(bond.GetIsConjugated())]             # 1
    feats += [float(bond.IsInRing())]                    # 1
    return np.array(feats, dtype=np.float32)


# ── SMILES → PyG Data ────────────────────────────────────────────────────────

NUM_NODE_FEATURES = 75
NUM_EDGE_FEATURES = 12


def smiles_to_graph(smiles: str) -> Data | None:
    """
    Convert a SMILES string to a PyTorch Geometric Data object.

    Returns None if the SMILES is invalid or the molecule has no atoms.

    Graph conventions:
      - Edges are bidirectional (both directions stored).
      - `data.x`         : float32 tensor [N, 73]
      - `data.edge_index`: int64  tensor [2, 2*B]
      - `data.edge_attr` : float32 tensor [2*B, 10]
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # ── Node features ──
    node_feats = [atom_features(a) for a in mol.GetAtoms()]
    if len(node_feats) == 0:
        return None
    x = torch.tensor(np.stack(node_feats), dtype=torch.float)

    # ── Edge features (bidirectional) ──
    src_list, dst_list, edge_feat_list = [], [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat = bond_features(bond)
        # i → j
        src_list.append(i)
        dst_list.append(j)
        edge_feat_list.append(feat)
        # j → i (same features)
        src_list.append(j)
        dst_list.append(i)
        edge_feat_list.append(feat)

    if len(src_list) == 0:
        # Single-atom molecule: no bonds → add self-loop to avoid isolated node
        src_list = [0]
        dst_list = [0]
        edge_feat_list = [np.zeros(12, dtype=np.float32)]

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(np.stack(edge_feat_list), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
