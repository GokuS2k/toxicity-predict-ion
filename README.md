# Tox21 Molecular Toxicity Prediction — Graph Neural Network

Multi-task toxicity prediction across 12 endpoints using a **Graph Attention Network (GATv2)**. Molecules are represented as graphs where **atoms are nodes** and **bonds are edges**, allowing the model to learn directly from molecular topology and chemistry.

## Model Architecture

```
Molecule (SMILES)
       │
       ▼
  RDKit parsing
       │
       ├─ Atoms → Node features [75-dim]
       └─ Bonds → Edge features [12-dim]
       │
       ▼
  ┌──────────────────────────────┐
  │   GATv2Conv  (Layer 1)       │  128 hidden × 4 heads → 512
  │   BatchNorm + ELU + Dropout  │
  ├──────────────────────────────┤
  │   GATv2Conv  (Layer 2)       │  512 → 512
  │   BatchNorm + ELU + Dropout  │
  ├──────────────────────────────┤
  │   GATv2Conv  (Layer 3)       │  512 → 128 (mean-aggregated)
  │   BatchNorm + ELU            │
  └──────────────────────────────┘
       │
       ▼
  Global Mean Pool ──┐
  Global Max Pool  ──┴─► concat [256-dim]
       │
       ▼
  MLP trunk: Linear(256→256) → ReLU → Dropout
             Linear(256→128) → ReLU → Dropout
       │
       ▼
  12 × Linear(128→1) task heads
       │
       ▼
  12 toxicity logits  →  sigmoid  →  probabilities
```

### Node Features (75 dimensions)

| Feature | Dim | Description |
|---------|-----|-------------|
| Atom type | 44 | One-hot over 43 elements + "other" |
| Hybridization | 6 | SP / SP2 / SP3 / SP3D / SP3D2 + other |
| Chirality | 3 | CW / CCW / other |
| Total H count | 10 | 0–8 + other |
| Degree | 8 | 0–6 + other |
| Formal charge | 1 | Normalised by /4 |
| Is aromatic | 1 | Boolean |
| Is in ring | 1 | Boolean |
| Radical electrons | 1 | Clipped and normalised |

### Edge Features (12 dimensions)

| Feature | Dim | Description |
|---------|-----|-------------|
| Bond type | 5 | SINGLE / DOUBLE / TRIPLE / AROMATIC + other |
| Stereo | 5 | NONE / ANY / Z / E + other |
| Is conjugated | 1 | Boolean |
| Is in ring | 1 | Boolean |

All edges are bidirectional; features are shared in both directions.

## Project Structure

```
toxicity-predict-ion/
├── src/
│   ├── featurization.py   # SMILES → molecular graph (node & edge features)
│   ├── dataset.py         # Tox21GraphDataset + data loading/splitting
│   ├── model.py           # MolecularGNN (GATv2), masked BCE loss
│   ├── train.py           # Full training pipeline (entry point)
│   ├── evaluate.py        # Metrics (AUROC/AUPRC) + plots
│   └── predict.py         # Inference API + CLI demo
├── data/
│   └── tox21.csv.gz       # Auto-downloaded on first run
├── models/
│   └── tox21_gnn_model.pt # Saved best checkpoint
├── results/
│   ├── metrics_validation.csv
│   ├── metrics_test.csv
│   ├── auroc_bar_*.png
│   ├── roc_curves_*.png
│   └── confusion_matrices_*.png
└── requirements.txt
```

## Dataset

**Tox21** — 7,831 compounds across 12 toxicity assays (6 nuclear receptor + 6 stress response):

| Group | Endpoints |
|-------|-----------|
| Nuclear receptor | NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma |
| Stress response  | SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53 |

- Missing label rate: ~17% on average
- Class imbalance: 2.9–16.2% positive rate
- Split: 80 / 10 / 10 (train / val / test)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train

```bash
python src/train.py

# Custom hyperparameters
python src/train.py --epochs 200 --hidden-dim 256 --num-layers 4 --batch-size 32
```

All options:
```
--epochs        int    Max training epochs (default: 150)
--batch-size    int    Mini-batch size (default: 64)
--hidden-dim    int    Hidden dimension per GAT head (default: 128)
--num-heads     int    Number of attention heads (default: 4)
--num-layers    int    Number of GATv2 layers (default: 3)
--dropout       float  Dropout rate (default: 0.2)
--lr            float  Learning rate (default: 1e-3)
--weight-decay  float  L2 regularisation (default: 1e-5)
--patience      int    Early-stopping patience (default: 25)
--seed          int    Random seed (default: 42)
```

### Predict

```bash
# Demo on 5 known compounds
python src/predict.py
```

```python
from src.predict import load_model, predict_toxicity, batch_predict

model, device = load_model()

# Single molecule
result = predict_toxicity("CC(=O)Oc1ccccc1C(=O)O", model, device)
print(result["NR-AhR"])  # {'prediction': 0, 'probability': 0.0312, 'label': 'non-toxic'}

# Batch
results = batch_predict(["CCO", "c1ccccc1", "ClC1=CC(Cl)=C(Cl)C(Cl)=C1Cl"], model, device)
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| GATv2 over GCN | Dynamic attention learns which neighbours matter per molecule |
| Edge features in attention | Bond type/aromaticity/stereo directly informs message passing |
| Mean + Max global pooling | Captures both average and extreme atom environments |
| Shared backbone, per-task heads | Enables multi-task transfer learning across related assays |
| Masked BCE loss | Safely handles missing labels without imputation |
| Per-task positive class weights | Counteracts 5–20× class imbalance in each assay |
| ReduceLROnPlateau on val AUROC | Adapts LR without overfitting on noisy labels |
| Early stopping (patience=25) | Prevents overfitting on the ~7.8K molecule dataset |
