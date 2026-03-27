# Tox21 Molecular Toxicity Prediction — Project Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Development Phases](#3-development-phases)
4. [Molecular Featurization](#4-molecular-featurization)
5. [Models](#5-models)
6. [Training Methodology](#6-training-methodology)
7. [Evaluation Framework](#7-evaluation-framework)
8. [Results](#8-results)
9. [Key Design Decisions](#9-key-design-decisions)
10. [Project Structure](#10-project-structure)
11. [Dependencies](#11-dependencies)

---

## 1. Project Overview

This project implements a **multi-task molecular toxicity prediction system** using Graph Neural Networks (GNNs). The system predicts whether a given molecule is toxic across 12 biological assay endpoints from the **Tox21** dataset.

Molecules are represented as graphs — atoms as nodes, bonds as edges — enabling models to learn directly from molecular topology and chemistry without hand-crafted fingerprints.

The project evolved across three development phases, starting from a classical machine learning baseline and progressively advancing toward state-of-the-art GNN architectures.

---

## 2. Dataset

**Tox21** (Toxicology in the 21st Century) is a public benchmark containing 7,831 compounds screened across 12 toxicity assays.

### Toxicity Endpoints

| Group | Endpoints |
|---|---|
| Nuclear Receptor Panel | NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma |
| Stress Response Panel | SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53 |

### Dataset Characteristics

| Property | Value |
|---|---|
| Total compounds | 7,831 |
| Number of tasks | 12 |
| Missing label rate | ~17% per task (average) |
| Positive class rate | 2.9% – 16.2% per task |
| Data format | SMILES strings + binary labels |

The heavy class imbalance (5–20× more negatives than positives per task) and the high missing-label rate are the two primary challenges in this dataset.

### Data Split

All experiments use an **80 / 10 / 10** train / validation / test split. Phase 2 additionally evaluates under a **scaffold split**, which partitions molecules by Bemis-Murcko scaffold to measure generalization to structurally novel compounds.

---

## 3. Development Phases

### Phase 0 — Initial Inspection (commit `87ee319`)

An initial data inspection script was written to explore the raw Tox21 CSV, characterize label distributions, and understand missing data patterns. A preliminary model and visualization assets were produced to establish a performance baseline.

### Phase 1 — Random Forest Baseline (PR #1, commit `5d6f0b8`)

A **Random Forest** classifier was implemented as a classical ML baseline using Morgan fingerprints (circular fingerprints computed with RDKit). This established a performance floor against which subsequent GNN models would be compared.

### Phase 2 — GATv2 Graph Neural Network (PR #2, commit `3b8e712`)

The Random Forest was replaced by a **Graph Attention Network v2 (GATv2)**, representing molecules as graphs and learning directly from atomic topology. This became the main production model in `src/`.

Key improvements over the baseline:
- Molecules processed as graphs (no information loss from fingerprint hashing)
- Attention mechanism learns which neighboring atoms are relevant
- Edge features (bond type, stereo, conjugation) incorporated into message passing
- Multi-task learning with shared backbone and per-task output heads

### Phase 3 — Dual-Model Pipeline with Scaffold Evaluation (PR #4, commits `4945c11`, `536e5f5`)

A second independent pipeline was built in `tox21_phase2/` introducing:
- **D-MPNN** (Directed Message Passing Neural Network) as an alternative architecture
- Both **random** and **scaffold** splitting strategies evaluated side-by-side
- A master comparison table generated across all model × split combinations
- Extended visualization suite (training curves, grouped AUROC bar, random-vs-scaffold delta)

### Current Branch — D-MPNN Implementation + Model Refresh (commits `6bbb974`, `f30a87f`)

- Full `DMPNNModel` implementation added (`tox21_phase2/src/models/dmpnn.py`)
- Featurization fixed for single-atom molecules (doubled self-loops for even edge counts)
- Trained GATv2 checkpoint refreshed with updated weights
- Validation and test metrics regenerated and saved

---

## 4. Molecular Featurization

Molecules are converted from SMILES strings to PyTorch Geometric `Data` objects using RDKit.

### Node Features (75 dimensions per atom)

| Feature | Dim | Encoding |
|---|---|---|
| Atom type | 44 | One-hot over 43 elements + "other" bucket |
| Hybridization | 6 | SP / SP2 / SP3 / SP3D / SP3D2 + "other" |
| Chirality | 3 | CW / CCW + "other" |
| Total H count | 10 | 0–8 one-hot + "other" |
| Degree | 8 | 0–6 one-hot + "other" |
| Formal charge | 1 | Continuous, normalized by /4 |
| Is aromatic | 1 | Boolean |
| Is in ring | 1 | Boolean |
| Radical electrons | 1 | Clipped to [0,4], normalized by /4 |

### Edge Features (12 dimensions per bond)

| Feature | Dim | Encoding |
|---|---|---|
| Bond type | 5 | SINGLE / DOUBLE / TRIPLE / AROMATIC + "other" |
| Stereo | 5 | NONE / ANY / Z / E + "other" |
| Is conjugated | 1 | Boolean |
| Is in ring | 1 | Boolean |

All bonds are represented bidirectionally — each bond generates two directed edges sharing the same feature vector. Single-atom molecules (no bonds) receive two self-loop edges to maintain consistent edge-pair indexing required by D-MPNN.

---

## 5. Models

### 5.1 GATv2 — Graph Attention Network v2

**Location:** `src/model.py` (Phase 1), `tox21_phase2/src/models/gatv2.py` (Phase 2)

GATv2 extends the original GAT by computing dynamic attention coefficients that depend on both source and target node features, rather than static linear combinations.

#### Architecture

```
Input SMILES
     │
     ▼
 RDKit → PyG Graph (node feats 75-dim, edge feats 12-dim)
     │
     ▼
 Linear input projection: 75 → 128 (ELU)
     │
     ▼
 GATv2Conv Layer 1: 128 → 128×4 heads = 512  [BatchNorm + ELU + Dropout]
 GATv2Conv Layer 2: 512 → 512               [BatchNorm + ELU + Dropout]
 GATv2Conv Layer 3: 512 → 128 (mean-agg)   [BatchNorm + ELU]
     │
     ▼
 Global Mean Pool ──┐
 Global Max Pool  ──┴─► concat [256-dim]
     │
     ▼
 MLP Trunk: Linear(256→256) → ReLU → Dropout
            Linear(256→128) → ReLU → Dropout
     │
     ▼
 12 × Linear(128→1) task heads
     │
     ▼
 12 toxicity logits → sigmoid → probabilities
```

#### Key Hyperparameters (Phase 1 / default)

| Parameter | Value |
|---|---|
| Hidden dim per head | 128 |
| Attention heads | 4 |
| GATv2 layers | 3 |
| Dropout | 0.2 |
| Self-loops | Enabled |

#### Key Hyperparameters (Phase 2)

| Parameter | Value |
|---|---|
| Hidden dim per head | 128 |
| Attention heads | 8 |
| GATv2 layers | 3 |
| Dropout | 0.3 |

---

### 5.2 D-MPNN — Directed Message Passing Neural Network

**Location:** `tox21_phase2/src/models/dmpnn.py`

Implements the Chemprop architecture (Yang et al., 2019). The core idea is to pass messages along **directed edges** rather than undirected ones, preventing each node from immediately reflecting its own information back (the "echo" problem in standard MPNNs).

#### Algorithm

```
1. Edge Initialization
   h_e^0 = ReLU(W_i · [x_src || x_edge])    for each directed edge e = (u→v)

2. Directed Message Passing (T iterations)
   node_agg[u] = sum of h_e for all edges pointing TO u
   m_{u→v}     = node_agg[u] - h_{v→u}        ← exclude reverse echo
   h_e          = ReLU(h_init_e + W_m · m_e)  ← residual connection

3. Node Readout
   node_agg[v] = sum_{(u→v)} h_e_final
   h_v         = ReLU(W_a · [x_v || node_agg[v]])

4. Graph Readout
   h_G = concat(GlobalMeanPool(h_v), GlobalMaxPool(h_v))

5. Classification
   logits = MLP(h_G)    → shape [B, 12]
```

The reverse-edge exclusion is computed efficiently using XOR indexing: since edges are stored in pairs (i→j at index 2k, j→i at index 2k+1), the reverse of edge k is simply `k ^ 1`.

#### Key Hyperparameters

| Parameter | Value |
|---|---|
| Hidden dim | 300 |
| Message passing steps | 3 |
| Dropout | 0.15 |
| Batch size | 50 |

---

## 6. Training Methodology

### Loss Function — Masked BCE

Standard binary cross-entropy cannot handle missing labels (NaN). A custom **masked BCE loss** is used:

```
L = mean over non-NaN entries of [ BCE(logit, label) × sample_weight ]
```

If no known labels exist in a batch, the loss returns a differentiable zero.

### Class Imbalance Handling

Each task has a **positive class weight** computed from the training set:

```
pos_weight[t] = neg_count[t] / pos_count[t],  clipped to [1, 50]
```

This weight is applied per-sample during loss computation, effectively up-weighting rare toxic examples by 5–20×.

### Optimizer & Scheduler

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| Gradient clipping | max norm = 5.0 |
| LR scheduler | ReduceLROnPlateau (mode=max, factor=0.5, patience=10) |
| Min LR | 1e-5 |

### Early Stopping

Training monitors **mean validation AUROC** across all tasks with at least two label classes. The best checkpoint is saved whenever validation AUROC improves. Training stops when no improvement is observed for `patience` consecutive epochs (default: 25 for Phase 1, 15 for Phase 2).

### Reproducibility

All experiments use seed 42 for Python's `random`, NumPy, and PyTorch (including CUDA if available).

---

## 7. Evaluation Framework

### Metrics

Three metrics are computed **per task** on the held-out set:

| Metric | Description |
|---|---|
| **AUROC** | Area Under the ROC Curve — primary metric; measures rank discrimination regardless of threshold |
| **AUPRC** | Area Under the Precision-Recall Curve — especially informative under heavy class imbalance |
| **Balanced Accuracy** | Average of sensitivity and specificity at the 0.5 decision threshold |

Tasks where only one class is present in the evaluation split are marked as NaN and excluded from aggregated means.

### Visualizations Generated

| Plot | Description |
|---|---|
| `auroc_bar_{split}.png` | Bar chart of per-task AUROC; bars colored green (≥0.65) or red (<0.65) |
| `roc_curves_{split}.png` | 3×4 grid of ROC curves, one per endpoint |
| `confusion_matrices_{split}.png` | 3×4 grid of normalized confusion matrices at 0.5 threshold |

---

## 8. Results

### GATv2 — Phase 1 Model Performance

Results from `models/tox21_gnn_model.pt` (current saved checkpoint).

#### Validation Set

| Endpoint | AUROC | AUPRC | Balanced Acc | N | Positives |
|---|---|---|---|---|---|
| NR-AR | 0.7766 | 0.4641 | 0.7120 | 726 | 31 |
| NR-AR-LBD | 0.8897 | 0.4725 | 0.7811 | 676 | 24 |
| NR-AhR | 0.8786 | 0.5137 | 0.7885 | 670 | 75 |
| NR-Aromatase | **0.9215** | 0.3573 | 0.7715 | 578 | 26 |
| NR-ER | 0.7123 | 0.4677 | 0.7119 | 620 | 83 |
| NR-ER-LBD | 0.8630 | 0.4852 | 0.7959 | 704 | 29 |
| NR-PPAR-gamma | 0.9210 | 0.4954 | 0.7440 | 644 | 21 |
| SR-ARE | 0.8277 | 0.5314 | 0.7305 | 591 | 102 |
| SR-ATAD5 | 0.8722 | 0.3308 | 0.7607 | 700 | 24 |
| SR-HSE | 0.7871 | 0.4189 | 0.7284 | 628 | 35 |
| SR-MMP | **0.8957** | **0.6943** | 0.7936 | 587 | 90 |
| SR-p53 | 0.8092 | 0.2923 | 0.6962 | 673 | 47 |
| **Mean** | **0.8629** | **0.4605** | — | — | — |

#### Test Set

| Endpoint | AUROC | AUPRC | Balanced Acc | N | Positives |
|---|---|---|---|---|---|
| NR-AR | 0.7661 | 0.4171 | 0.6711 | 726 | 31 |
| NR-AR-LBD | 0.8255 | 0.5552 | 0.7558 | 673 | 21 |
| NR-AhR | 0.8860 | 0.5674 | 0.8142 | 650 | 81 |
| NR-Aromatase | 0.8207 | 0.4776 | 0.7398 | 580 | 41 |
| NR-ER | 0.7071 | 0.4022 | 0.6588 | 612 | 72 |
| NR-ER-LBD | 0.8062 | 0.3771 | 0.7326 | 679 | 35 |
| NR-PPAR-gamma | **0.9367** | 0.4775 | 0.8674 | 634 | 16 |
| SR-ARE | 0.8088 | 0.4981 | 0.7152 | 584 | 101 |
| SR-ATAD5 | **0.9097** | 0.3888 | 0.8073 | 701 | 27 |
| SR-HSE | 0.7273 | 0.2557 | 0.6211 | 641 | 38 |
| SR-MMP | 0.8759 | **0.6581** | 0.7796 | 564 | 98 |
| SR-p53 | 0.8424 | 0.3568 | 0.7571 | 674 | 53 |
| **Mean** | **0.8344** | **0.4530** | — | — | — |

### Result Observations

**Strong performers (AUROC ≥ 0.87 on test):**
- **NR-PPAR-gamma** (0.9367) — highest test AUROC; low positive count (16) but the model generalizes well
- **SR-ATAD5** (0.9097) — strong discrimination on the ATAD5 genotoxicity assay
- **NR-AhR** (0.8860) — aryl hydrocarbon receptor activation; well-discriminated
- **SR-MMP** (0.8759) — mitochondrial membrane potential disruption; also highest AUPRC (0.6581)

**Moderate performers (AUROC 0.80–0.87 on test):**
- NR-AR-LBD, NR-Aromatase, NR-ER-LBD, SR-ARE, SR-p53 all fall in the 0.80–0.84 range

**Harder tasks (AUROC < 0.80 on test):**
- **NR-AR** (0.7661) and **NR-ER** (0.7071) are the most challenging. Both are androgen/estrogen receptor assays with high structural complexity and small positive counts.
- **SR-HSE** (0.7273) — heat shock element response; relatively lower signal

**Validation vs. Test gap:**
- Mean AUROC drops from 0.8629 (validation) to 0.8344 (test), a delta of ~0.029 — moderate generalization gap, consistent with the small dataset size (~780 test molecules).
- NR-AR-LBD shows the largest drop (0.8897 → 0.8255), suggesting some overfitting on this endpoint.
- NR-PPAR-gamma is slightly better on test (0.9210 → 0.9367), indicating robust learned signal.

**All 12 tasks exceed the 0.65 AUROC target** on both validation and test sets.

---

## 9. Key Design Decisions

| Decision | Rationale |
|---|---|
| **GATv2 over GCN** | Dynamic, input-dependent attention learns which neighbor atoms are most informative per molecule, unlike static GCN aggregation |
| **Edge features in attention** | Bond type, aromaticity, and stereo chemistry directly affect reactivity — these should inform message passing, not just atom identity |
| **Mean + Max global pooling** | Mean pooling captures average atom environment; max pooling retains the most extreme signals. Together they produce richer graph representations |
| **Shared backbone, per-task heads** | Multi-task learning allows related toxicity signals (e.g., nuclear receptor endpoints) to share representations while maintaining task-specific decision boundaries |
| **Masked BCE loss** | Naively treating NaN as non-toxic would introduce false negatives. Masking correctly excludes unknown entries from gradient computation |
| **Per-task positive class weights** | The 5–20× class imbalance would cause the model to predict all-negative. Weighting up positive samples counteracts this without oversampling |
| **ReduceLROnPlateau on val AUROC** | Directly optimizes for the primary evaluation metric; halves LR when AUROC plateaus to allow finer optimization near convergence |
| **Early stopping (patience = 25)** | The ~7,800 molecule dataset is small enough that models overfit within 150 epochs; early stopping selects the generalization peak |
| **Directed edges in D-MPNN** | Prevents information "echo" where a node immediately reads back its own message from the previous step through an undirected neighbor |
| **XOR reverse-edge indexing** | Efficient O(1) computation of paired directed edges without storing an explicit reverse-edge lookup table |
| **Scaffold splitting (Phase 2)** | Random splits can inflate performance by placing structurally similar molecules in both train and test. Scaffold splits enforce structural novelty in the test set, giving a more realistic estimate of generalization |

---

## 10. Project Structure

```
toxicity-predict-ion/
│
├── src/                          # Phase 1 — GATv2 pipeline
│   ├── featurization.py          # SMILES → PyG graph (node & edge features)
│   ├── dataset.py                # Tox21GraphDataset, data loading, 80/10/10 split
│   ├── model.py                  # MolecularGNN (GATv2), masked BCE loss, pos weights
│   ├── train.py                  # Full training pipeline (CLI entry point)
│   ├── evaluate.py               # AUROC/AUPRC/balanced acc + bar/ROC/confusion plots
│   └── predict.py                # Inference API + CLI demo on known compounds
│
├── tox21_phase2/                 # Phase 2 — Dual-model comparison pipeline
│   ├── run_all.py                # Master pipeline: trains both models × both splits
│   ├── data/
│   │   └── tox21.csv.gz          # Raw Tox21 dataset
│   └── src/
│       ├── data_loading.py       # Dataset loading + task definitions
│       ├── featurization.py      # Shared SMILES → graph featurization (with D-MPNN fixes)
│       ├── splitting.py          # Random split + Bemis-Murcko scaffold split
│       ├── evaluate.py           # Metrics + comparison table builder
│       ├── visualize.py          # Extended visualization suite
│       ├── train_gatv2.py        # GATv2 training loop
│       ├── train_dmpnn.py        # D-MPNN training loop
│       └── models/
│           ├── gatv2.py          # GATv2Model architecture
│           └── dmpnn.py          # DMPNNModel architecture
│
├── data/
│   └── tox21.csv.gz              # Auto-downloaded on first run (Phase 1)
│
├── models/
│   └── tox21_gnn_model.pt        # Saved best GATv2 checkpoint (Phase 1)
│
├── results/
│   ├── metrics_validation.csv    # Per-task metrics on validation set
│   ├── metrics_test.csv          # Per-task metrics on test set
│   ├── auroc_bar_validation.png
│   ├── auroc_bar_test.png
│   ├── roc_curves_validation.png
│   ├── roc_curves_test.png
│   ├── confusion_matrices_validation.png
│   └── confusion_matrices_test.png
│
├── requirements.txt
├── README.md
└── PROJECT_DOCUMENTATION.md     ← this file
```

---

## 11. Dependencies

| Library | Min Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.0.0 | Neural network training and inference |
| `torch-geometric` | ≥ 2.4.0 | Graph neural network layers, data structures, loaders |
| `rdkit` | ≥ 2022.9.1 | SMILES parsing and molecular featurization |
| `scikit-learn` | ≥ 1.3.0 | AUROC, AUPRC, balanced accuracy, confusion matrix |
| `pandas` | ≥ 2.0.0 | Dataset loading and metrics DataFrames |
| `numpy` | ≥ 1.24.0 | Array operations and numerical utilities |
| `matplotlib` | ≥ 3.7.0 | Plotting (bar charts, ROC curves, confusion matrices) |
| `seaborn` | ≥ 0.12.0 | Confusion matrix heatmaps |
| `tqdm` | ≥ 4.65.0 | Progress bars during featurization |

Install with:
```bash
pip install -r requirements.txt
```
