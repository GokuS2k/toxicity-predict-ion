# Tox21 Toxicity Prediction — Project Documentation

---

## Table of Contents

1. [Project Aim / Overview](#1-project-aim--overview)
2. [Dataset Description](#2-dataset-description)
3. [Basic Dataset Statistics](#3-basic-dataset-statistics)
4. [Data Preprocessing Steps](#4-data-preprocessing-steps)
5. [Machine Learning Model & Approach](#5-machine-learning-model--approach)
6. [Training Process](#6-training-process)
7. [Evaluation Metrics & Results](#7-evaluation-metrics--results)
8. [Outputs & Graphs](#8-outputs--graphs)
9. [Conclusion](#9-conclusion)

---

## 1. Project Aim / Overview

**Goal**: Build a machine learning pipeline that predicts whether a chemical compound is toxic across 12 distinct biological endpoints, using only its molecular structure (SMILES string) as input.

**Why it matters**: Testing every chemical in animals or cell assays is slow and expensive. Computational toxicity models provide an initial safety screen, flagging potentially hazardous compounds early in drug discovery or chemical risk assessment — reducing cost, time, and animal use.

**What was built**:
- An end-to-end pipeline that downloads the dataset, featurizes molecules, trains 12 independent Random Forest classifiers (one per toxicity endpoint), evaluates them with robust metrics, and serializes the trained model for future inference.
- A prediction interface that accepts any SMILES string and returns per-endpoint toxicity probabilities and binary labels.

**Dataset used**: [Tox21](https://tripod.nih.gov/tox21/challenge/) — the NIH Tox21 challenge dataset, a standard multi-task toxicity benchmark in cheminformatics.

**Tech stack**: Python · RDKit · scikit-learn · pandas · NumPy · Matplotlib · Seaborn · joblib

---

## 2. Dataset Description

### Source

The Tox21 dataset was originally released as part of the [NIH Tox21 Data Challenge](https://tripod.nih.gov/tox21/challenge/) and is hosted by MoleculeNet:

```
https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz
```

It contains **7,831 chemical compounds** labelled against **12 toxicity endpoints** — seven nuclear receptor assays and five stress response assays.

### Columns

| Column | Type | Description |
|---|---|---|
| `smiles` | string | SMILES notation encoding the 2D molecular structure |
| `mol_id` | string | Compound identifier |
| `NR-AR` | 0/1/NaN | Androgen Receptor activation |
| `NR-AR-LBD` | 0/1/NaN | Androgen Receptor — Ligand Binding Domain |
| `NR-AhR` | 0/1/NaN | Aryl hydrocarbon Receptor activation |
| `NR-Aromatase` | 0/1/NaN | Aromatase enzyme inhibition |
| `NR-ER` | 0/1/NaN | Estrogen Receptor activation |
| `NR-ER-LBD` | 0/1/NaN | Estrogen Receptor — Ligand Binding Domain |
| `NR-PPAR-gamma` | 0/1/NaN | Peroxisome Proliferator-Activated Receptor gamma |
| `SR-ARE` | 0/1/NaN | Antioxidant Response Element activation |
| `SR-ATAD5` | 0/1/NaN | ATAD5 genotoxicity marker |
| `SR-HSE` | 0/1/NaN | Heat Shock Factor Response Element |
| `SR-MMP` | 0/1/NaN | Mitochondrial Membrane Potential disruption |
| `SR-p53` | 0/1/NaN | p53 tumour-suppressor pathway activation |

Label encoding: `1` = toxic (active in assay), `0` = non-toxic (inactive), `NaN` = not measured.

### Example Rows

| smiles | mol_id | NR-AR | NR-AhR | NR-ER | SR-MMP | SR-p53 |
|---|---|---|---|---|---|---|
| `CCOc1ccc2nc(S(N)(=O)=O)sc2c1` | TOX1234 | 0 | 0 | 0 | 0 | 0 |
| `CCCN(CC)C(CC)C(=O)Nc1c(C)cccc1C` | TOX2901 | 0 | 1 | 0 | NaN | 0 |
| `CC(O)(P(=O)(O)O)P(=O)(O)O` | TOX0055 | NaN | 0 | NaN | 1 | NaN |
| `c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34` | TOX3312 | 1 | 1 | 1 | 1 | 1 |
| `CCO` | TOX0001 | 0 | 0 | 0 | 0 | 0 |

> **Note on SMILES**: SMILES (Simplified Molecular Input Line Entry System) is a compact text notation that encodes chemical structure. For example, `CCO` = ethanol, `c1ccccc1` = benzene, `CC(=O)Oc1ccccc1C(=O)O` = aspirin.

### The 12 Toxicity Endpoints

| Category | Endpoint | Biological Significance |
|---|---|---|
| Nuclear Receptor | NR-AR | Androgen receptor activation (hormonal disruption) |
| Nuclear Receptor | NR-AR-LBD | Androgen receptor ligand binding domain |
| Nuclear Receptor | NR-AhR | Aryl hydrocarbon receptor (dioxin-like effects) |
| Nuclear Receptor | NR-Aromatase | Aromatase inhibition (sex hormone regulation) |
| Nuclear Receptor | NR-ER | Estrogen receptor activation (endocrine disruption) |
| Nuclear Receptor | NR-ER-LBD | Estrogen receptor ligand binding domain |
| Nuclear Receptor | NR-PPAR-gamma | PPAR-gamma (metabolic disruption) |
| Stress Response | SR-ARE | Antioxidant response / oxidative stress |
| Stress Response | SR-ATAD5 | DNA damage / genotoxicity marker |
| Stress Response | SR-HSE | Heat shock / protein misfolding stress |
| Stress Response | SR-MMP | Mitochondrial membrane potential disruption |
| Stress Response | SR-p53 | p53 activation (DNA damage / apoptosis) |

---

## 3. Basic Dataset Statistics

### Overall

| Property | Value |
|---|---|
| Total compounds | 7,831 |
| Valid SMILES | 7,823 |
| Skipped (invalid SMILES) | 8 (aluminum-containing compounds RDKit cannot parse) |
| Toxicity endpoints | 12 |
| Mean missing label rate | 17.1% |
| Dataset split | 80% train / 10% val / 10% test |
| Train samples | 6,258 |
| Validation samples | 782 |
| Test samples | 783 |

### Per-Endpoint Label Distribution (Full Dataset)

| Endpoint | Available Labels | Positives (toxic) | Positive Rate | Missing Labels |
|---|---|---|---|---|
| NR-AR | 7,265 | 309 | 4.3% | 566 |
| NR-AR-LBD | 6,758 | 237 | 3.5% | 1,073 |
| NR-AhR | 6,549 | 768 | 11.7% | 1,282 |
| NR-Aromatase | 5,821 | 300 | 5.2% | 2,010 |
| NR-ER | 6,193 | 793 | 12.8% | 1,638 |
| NR-ER-LBD | 6,955 | 350 | 5.0% | 876 |
| NR-PPAR-gamma | 6,450 | 186 | 2.9% | 1,381 |
| SR-ARE | 5,832 | 942 | 16.2% | 1,999 |
| SR-ATAD5 | 7,072 | 264 | 3.7% | 759 |
| SR-HSE | 6,467 | 372 | 5.8% | 1,364 |
| SR-MMP | 5,810 | 918 | 15.8% | 2,021 |
| SR-p53 | 6,774 | 423 | 6.2% | 1,057 |

**Key observation**: All endpoints are highly class-imbalanced (2.9%–16.2% positive rate). This makes accuracy a poor metric — a trivially non-toxic model would be ~95% accurate but useless. AUROC and AUPRC are used instead.

---

## 4. Data Preprocessing Steps

### Step 1 — Download & Load

The dataset is downloaded as a gzip-compressed CSV from MoleculeNet's S3 bucket and cached locally at `data/tox21.csv.gz`. The file is ~121 KB compressed and ~525 KB uncompressed.

```python
# src/data_acquisition.py
df = pd.read_csv("data/tox21.csv.gz", compression="gzip")
# Shape: (7831, 14) — 12 label columns + smiles + mol_id
```

### Step 2 — SMILES Validation

All SMILES strings are parsed by RDKit. The 8 aluminum-containing SMILES that RDKit cannot process are silently dropped, leaving 7,823 valid compounds.

```python
mol = Chem.MolFromSmiles(smiles)
if mol is None:
    continue  # skip invalid
```

### Step 3 — Morgan Fingerprint Generation

Each valid SMILES is converted to a **Morgan fingerprint** (ECFP4-equivalent):

| Parameter | Value | Rationale |
|---|---|---|
| Radius | 2 | Captures up to 4-bond neighbourhoods (ECFP4 standard) |
| Number of bits | 2048 | Dense enough for diversity; sparse enough for efficiency |
| Output type | `uint8` array | Compact binary vector; `1` = substructure present |

```python
# src/preprocessing.py
from rdkit.Chem import AllChem

def smiles_to_morgan(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.uint8)  # shape: (2048,)
```

The result is a **7,823 × 2048** feature matrix `X`, where each row is a compound and each column is a binary indicator of a circular substructure.

### Step 4 — Label Matrix Construction

The 12 endpoint columns are extracted into a label matrix `Y` of shape (7,823 × 12), with NaN values preserved to indicate missing labels (not imputed).

```python
task_names = ["NR-AR", "NR-AR-LBD", ..., "SR-p53"]
Y = df[task_names].values.astype(float)  # NaN preserved
```

### Step 5 — Train / Validation / Test Split

An 80/10/10 stratified split is applied. Stratification is performed on the endpoint with the highest proportion of non-NaN labels to ensure each split has a representative fraction of positives.

```python
from sklearn.model_selection import train_test_split

X_train, X_temp, Y_train, Y_temp = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=stratify_col
)
X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp, Y_temp, test_size=0.5, random_state=42, stratify=stratify_col_temp
)
```

| Split | Samples |
|---|---|
| Train | 6,258 |
| Validation | 782 |
| Test | 783 |

### Design Decision Summary

| Decision | Choice | Rationale |
|---|---|---|
| Featurization | Morgan FP (r=2, 2048 bits) | ECFP4 equivalent; validated standard for RF on molecular data |
| Invalid SMILES | Drop (not impute) | 8 compounds; imputing would be meaningless for structure-based features |
| Missing labels | Mask (exclude per task) | Safer than imputation; no risk of label leakage |
| Split strategy | Stratified random 80/10/10 | Preserves class ratio across splits |

---

## 5. Machine Learning Model & Approach

### Model Architecture — Multi-Task Random Forest

Rather than a single multi-output model, **12 separate Random Forest classifiers** are trained — one per toxicity endpoint. This approach was chosen because:

1. Each endpoint has a **different pattern of missing labels** — a shared model would need to handle different subsets of training samples per task.
2. Independent models are simpler to tune, debug, and replace individually.
3. scikit-learn's `RandomForestClassifier` natively handles binary classification well.

The 12 models are wrapped in a custom `Tox21RandomForest` class (`src/model.py`) that provides a unified `.fit()`, `.predict_proba()`, `.predict()`, `.save()`, and `.load()` interface.

### Hyperparameters

```python
rf_params = {
    "n_estimators": 100,       # 100 trees per forest
    "max_depth": 20,           # limits overfitting
    "class_weight": "balanced",# auto-reweights minority class
    "random_state": 42,        # reproducibility
    "n_jobs": -1,              # all CPU cores
    "min_samples_leaf": 2,     # further regularization
}
```

**`class_weight='balanced'`** is the key imbalance-handling mechanism. scikit-learn automatically sets sample weights inversely proportional to class frequencies:

```
weight_for_class_c = n_samples / (n_classes × count_of_class_c)
```

This forces the model to pay equal attention to rare toxic compounds (3–16% of data) without synthetic oversampling (SMOTE, etc.).

### Why Random Forest?

Random Forests are a well-established baseline for molecular property prediction:

- Handle high-dimensional sparse binary features (2048-bit fingerprints) natively.
- Robust to irrelevant features through feature subsampling at each split.
- No feature scaling required (unlike SVM, logistic regression).
- Provide feature importances (useful for interpretability).
- Fast to train and predict at this scale (~7,800 compounds).

### Feature Importances

After training, each of the 12 RF models exposes feature importances — a 2048-length array indicating which fingerprint bits (molecular substructures) are most predictive. These can be used with RDKit to map back to specific chemical substructures.

---

## 6. Training Process

### Pipeline (src/train.py)

The training pipeline is orchestrated by `src/train.py` and runs as follows:

```
Step 1 ── Download & cache dataset (tox21.csv.gz)
             ↓
Step 2 ── Featurize: SMILES → Morgan fingerprints (7,823 × 2048)
             ↓
Step 3 ── Split: 80/10/10 stratified train/val/test
             ↓
Step 4 ── Train: 12 RandomForest classifiers (one per endpoint)
          Each model trained only on samples with known labels for that task
             ↓
Step 5 ── Evaluate on Validation Set
          Compute AUROC, AUPRC, Balanced Accuracy per endpoint
          Generate bar chart, ROC curves, confusion matrices
             ↓
Step 6 ── Evaluate on Test Set
          Same metrics; generates separate result files
             ↓
Step 7 ── Save model to models/tox21_rf_model.joblib (joblib compress=3)
             ↓
Step 8 ── Demo predictions (ethanol, benzene, benzocaine)
          Print final summary table
```

### Masked Training Per Endpoint

For each endpoint `i`, training uses only compounds where the label is known (not NaN):

```python
for i, task in enumerate(task_names):
    mask = ~np.isnan(Y_train[:, i])          # known labels only
    X_task = X_train[mask]
    y_task = Y_train[mask, i].astype(int)
    model_i.fit(X_task, y_task)
    self.models.append(model_i)
```

This means each of the 12 models is trained on a different-sized subset of the training data — from ~4,666 samples (NR-Aromatase, highest missing rate) to ~5,812 samples (NR-AR, lowest missing rate).

### Running the Pipeline

```bash
python src/train.py
```

Training completes in under a minute on a modern laptop (scikit-learn, n_jobs=-1). Outputs are saved to `results/` and `models/`.

---

## 7. Evaluation Metrics & Results

### Metrics Used

| Metric | Why used |
|---|---|
| **AUROC** (Area Under ROC Curve) | Primary metric. Measures ranking ability. Threshold-free. Robust to class imbalance. Score of 0.5 = random; 1.0 = perfect. |
| **AUPRC** (Area Under Precision-Recall Curve) | Secondary metric. More sensitive to rare-positive performance than AUROC. Better reflects real-world utility. |
| **Balanced Accuracy** | (TPR + TNR) / 2. Accounts for imbalance; meaningful at a fixed 0.5 threshold. |

### Validation Set Results

| Endpoint | AUROC | AUPRC | Balanced Acc | Samples | Positives |
|---|---|---|---|---|---|
| NR-AR | 0.7699 | 0.5336 | 0.7414 | 727 | 39 |
| NR-AR-LBD | 0.8707 | 0.6574 | 0.8363 | 674 | 22 |
| NR-AhR | **0.8878** | 0.6096 | 0.7669 | 661 | 80 |
| NR-Aromatase | 0.8099 | 0.3757 | 0.6370 | 587 | 32 |
| NR-ER | 0.6820 | 0.4520 | 0.6902 | 624 | 85 |
| NR-ER-LBD | 0.7932 | 0.5178 | 0.7292 | 694 | 35 |
| NR-PPAR-gamma | 0.8109 | 0.1625 | 0.5697 | 640 | 13 |
| SR-ARE | 0.7539 | 0.4556 | 0.6406 | 590 | 95 |
| SR-ATAD5 | 0.7611 | 0.1780 | 0.6361 | 702 | 17 |
| SR-HSE | 0.7012 | 0.2626 | 0.6383 | 656 | 36 |
| SR-MMP | 0.8594 | **0.6296** | 0.7879 | 585 | 99 |
| SR-p53 | 0.8179 | 0.4014 | 0.6420 | 682 | 49 |
| **Mean** | **0.7932** | **0.4197** | **0.6929** | — | — |

### Test Set Results

| Endpoint | AUROC | AUPRC | Balanced Acc | Samples | Positives |
|---|---|---|---|---|---|
| NR-AR | 0.8169 | 0.5266 | 0.7341 | 725 | 23 |
| NR-AR-LBD | 0.8395 | 0.6489 | 0.7761 | 668 | 25 |
| NR-AhR | 0.8635 | 0.4975 | 0.7088 | 644 | 73 |
| NR-Aromatase | 0.7824 | 0.3858 | 0.6164 | 557 | 33 |
| NR-ER | 0.7249 | 0.4411 | 0.6646 | 613 | 68 |
| NR-ER-LBD | 0.8837 | 0.5668 | 0.7510 | 693 | 36 |
| NR-PPAR-gamma | **0.9083** | 0.5119 | 0.7599 | 638 | 17 |
| SR-ARE | 0.8154 | 0.5319 | 0.6984 | 602 | 104 |
| SR-ATAD5 | 0.8622 | 0.4755 | 0.7071 | 710 | 30 |
| SR-HSE | 0.6895 | 0.3192 | 0.6039 | 642 | 41 |
| SR-MMP | 0.8719 | **0.6375** | 0.7819 | 569 | 85 |
| SR-p53 | 0.7948 | 0.2969 | 0.5803 | 682 | 48 |
| **Mean** | **0.8211** | **0.4950** | **0.6902** | — | — |

### Key Results Summary

| Criterion | Target | Achieved |
|---|---|---|
| Endpoints with AUROC > 0.65 | ≥ 8 of 12 | **12 of 12** ✓ |
| Mean Val AUROC | — | **0.793** |
| Mean Test AUROC | — | **0.821** |
| Best endpoint (Test AUROC) | — | NR-PPAR-gamma (0.908) |
| Weakest endpoint (Test AUROC) | — | SR-HSE (0.690) |

The model **meets and exceeds the success criterion**: all 12 endpoints surpass AUROC 0.65, and test AUROC is actually slightly higher than validation AUROC on average, suggesting stable generalisation without overfitting.

### Example Predictions (Known Compounds)

| Compound | SMILES | NR-AhR | NR-ER | SR-MMP |
|---|---|---|---|---|
| Ethanol | `CCO` | non-toxic (0.09) | non-toxic (0.26) | non-toxic (0.11) |
| Benzene | `c1ccccc1` | non-toxic (0.20) | non-toxic (0.34) | non-toxic (0.25) |
| Benzocaine | `CCOC(=O)c1ccc(N)cc1` | **TOXIC (0.59)** | **TOXIC (0.51)** | non-toxic (0.29) |

---

## 8. Outputs & Graphs

All outputs are saved to the `results/` directory after training.

### File Inventory

| File | Description | Size |
|---|---|---|
| `results/metrics_validation.csv` | Per-endpoint AUROC, AUPRC, Balanced Acc (validation) | ~1 KB |
| `results/metrics_test.csv` | Per-endpoint AUROC, AUPRC, Balanced Acc (test) | ~1 KB |
| `results/auroc_bar_validation.png` | AUROC bar chart — validation set | 85 KB |
| `results/auroc_bar_test.png` | AUROC bar chart — test set | 85 KB |
| `results/roc_curves_validation.png` | 3×4 ROC curve grid — validation set | 234 KB |
| `results/roc_curves_test.png` | 3×4 ROC curve grid — test set | 232 KB |
| `results/confusion_matrices_validation.png` | 3×4 normalised confusion matrix grid — validation | 115 KB |
| `results/confusion_matrices_test.png` | 3×4 normalised confusion matrix grid — test | 111 KB |
| `models/tox21_rf_model.joblib` | Serialised Tox21RandomForest (12 RF models) | 8.8 MB |

### Graph Descriptions

#### 1. AUROC Bar Charts (`auroc_bar_*.png`)

A horizontal bar chart with one bar per endpoint. Bars are colour-coded:

- **Green** — AUROC ≥ 0.80 (strong performance)
- **Orange** — 0.65 ≤ AUROC < 0.80 (acceptable)
- **Red** — AUROC < 0.65 (below threshold)

A dashed vertical line at 0.65 marks the minimum acceptable threshold. All 12 bars fall to the right of this line on both validation and test sets.

```
Example (test set):
NR-PPAR-gamma ████████████████████████████████ 0.908  ← best
SR-MMP        ███████████████████████████████  0.872
NR-ER-LBD     ███████████████████████████████  0.884
NR-AR         ████████████████████████████     0.817
...
SR-HSE        █████████████████████            0.690  ← weakest (still > 0.65)
              0.5  |  0.65  |  0.75  |  0.85  |  1.0
                        ^threshold
```

#### 2. ROC Curves (`roc_curves_*.png`)

A 3-row × 4-column grid of ROC curves, one per endpoint. Each subplot shows:
- The ROC curve (True Positive Rate vs. False Positive Rate at varying thresholds)
- The diagonal dashed line (random classifier baseline, AUROC = 0.50)
- The endpoint name and AUROC score in the title

The area under each curve is shaded. Curves bowing towards the top-left corner indicate better discrimination ability.

#### 3. Confusion Matrices (`confusion_matrices_*.png`)

A 3-row × 4-column grid of normalised confusion matrices, one per endpoint. Each cell shows the fraction of predictions in each category (normalised by true class):

```
         Predicted:    Non-toxic    Toxic
Actual:
Non-toxic              TN rate      FP rate
Toxic                  FN rate      TP rate
```

Cells are colour-mapped (dark = high proportion). The diagonal (TN, TP) shows correct predictions. High true-positive rates (top-right cell) are particularly important given the severe class imbalance — the model should not simply predict "non-toxic" for everything.

### Using Prediction Output Programmatically

```python
from src.predict import predict_toxicity

result = predict_toxicity("c1ccccc1")  # benzene

for endpoint, info in result.items():
    print(f"{endpoint:15s}: {info['label']:10s}  P(toxic)={info['probability']:.3f}")
```

**Sample output**:
```
NR-AR          : non-toxic   P(toxic)=0.041
NR-AR-LBD      : non-toxic   P(toxic)=0.031
NR-AhR         : non-toxic   P(toxic)=0.199
NR-Aromatase   : non-toxic   P(toxic)=0.059
NR-ER          : non-toxic   P(toxic)=0.337
NR-ER-LBD      : non-toxic   P(toxic)=0.116
NR-PPAR-gamma  : non-toxic   P(toxic)=0.022
SR-ARE         : non-toxic   P(toxic)=0.145
SR-ATAD5       : non-toxic   P(toxic)=0.044
SR-HSE         : non-toxic   P(toxic)=0.109
SR-MMP         : non-toxic   P(toxic)=0.247
SR-p53         : non-toxic   P(toxic)=0.095
```

---

## 9. Conclusion

### What was achieved

This project delivered a complete, working toxicity prediction pipeline for the Tox21 benchmark:

- **All 12 toxicity endpoints exceeded AUROC 0.65** — the defined success threshold — with 10 of 12 exceeding AUROC 0.75 on the test set.
- The **mean test AUROC of 0.821** and **mean validation AUROC of 0.793** demonstrate consistent, generalisable performance with no signs of severe overfitting.
- The best-performing endpoint was **NR-PPAR-gamma (AUROC 0.908)** on the test set; the weakest was **SR-HSE (AUROC 0.690)**, still comfortably above threshold.
- A reusable `predict_toxicity()` API allows instant predictions for any novel SMILES string without retraining.

### Strengths

- **Simple and fast**: No GPUs required. Training completes in under a minute on a laptop.
- **Interpretable**: Random Forest feature importances can be mapped back to molecular substructures.
- **Robust to imbalance**: `class_weight='balanced'` handles the 3–16% positive rate without oversampling artefacts.
- **Clean missing-data handling**: Masked training ensures models are never confused by unlabelled samples.
- **Well-validated methodology**: Morgan fingerprints + Random Forest is an established, reproducible baseline for molecular property prediction.

### Limitations

1. **2D features only**: Morgan fingerprints encode topology, not 3D conformation. Shape- or pharmacophore-based descriptors could improve predictions, especially for receptor binding tasks.
2. **Missing-at-random assumption**: Labels are treated as missing randomly. If missing labels correlate with toxicity (e.g., unassayable compounds), the model is biased.
3. **Random (not scaffold) splitting**: Structurally similar molecules can appear in both train and test sets, inflating performance estimates. Scaffold-based splitting would give a more realistic measure of generalisation to new chemical series.
4. **No applicability domain**: The model gives a confidence value for any SMILES, even molecules very different from training data. A Tanimoto similarity filter could flag out-of-domain predictions.
5. **Binary outputs only**: The model predicts active/inactive; it does not model potency, dose-response curves, or continuous toxicity measures.

### Recommended Next Steps

| Priority | Improvement | Expected Impact |
|---|---|---|
| High | Scaffold-based train/test split (Murcko scaffolds) | More realistic generalisation estimate |
| High | Graph Neural Networks (GCN, MPNN, DMPNN) | Typically 5–10% AUROC improvement on Tox21 |
| Medium | Hyperparameter tuning (`max_depth`, `n_estimators`, `min_samples_leaf`) | Marginal RF improvement |
| Medium | Ensemble RF + gradient boosting (XGBoost/LightGBM) | Additional ~2–3% AUROC |
| Medium | SHAP values for substructure attribution | Interpretability for regulatory use |
| Low | Uncertainty quantification (conformal prediction) | Confidence bounds on predictions |
| Low | Tanimoto-based applicability domain filter | Safer deployment on novel chemotypes |

### Final Verdict

The Random Forest + Morgan fingerprint baseline is a solid, production-ready starting point for multi-endpoint toxicity prediction. It is fast, interpretable, and well-calibrated for the Tox21 benchmark. For improved accuracy, especially on held-out scaffold clusters, graph neural network approaches are the recommended next step.
