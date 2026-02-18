# Tox21 Toxicity Prediction — Random Forest Baseline

A baseline model for multi-endpoint molecular toxicity prediction using
Random Forest classifiers on Morgan fingerprints, trained on the
[Tox21 dataset](https://tripod.nih.gov/tox21/challenge/).

## Project Structure

```
toxicity-predict-ion/
├── data/
│   ├── tox21.csv.gz          # Tox21 dataset (auto-downloaded)
│   └── label_summary.csv     # Per-endpoint label distribution stats
├── models/
│   └── tox21_rf_model.joblib # Saved model (12 RF classifiers)
├── results/
│   ├── metrics_validation.csv
│   ├── metrics_test.csv
│   ├── auroc_bar_validation.png
│   ├── auroc_bar_test.png
│   ├── roc_curves_validation.png
│   ├── roc_curves_test.png
│   ├── confusion_matrices_validation.png
│   └── confusion_matrices_test.png
└── src/
    ├── data_acquisition.py   # Dataset download & inspection
    ├── preprocessing.py      # SMILES -> Morgan fingerprints + splits
    ├── model.py              # Tox21RandomForest class
    ├── evaluation.py         # Metrics, charts, confusion matrices
    ├── train.py              # Full training pipeline (entry point)
    └── predict.py            # Prediction interface
```

## Dataset Statistics

| Property | Value |
|---|---|
| Total compounds | 7,831 |
| Valid SMILES | 7,823 (8 skipped — invalid aluminum SMILES) |
| Toxicity endpoints | 12 |
| Mean missing label rate | 17.1% |
| Train / Val / Test split | 6,258 / 782 / 783 (80/10/10) |

**Per-endpoint label distribution** (full dataset):

| Endpoint | Available | Positives | Pos Rate | Missing |
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

All endpoints are highly imbalanced (2.9%–16.2% positive rate).
`class_weight='balanced'` is used in the Random Forest to compensate.

## Model Performance

Random Forest (n_estimators=100, max_depth=20, class_weight='balanced').

| Endpoint | Val AUROC | Test AUROC | Val AUPRC | Test AUPRC |
|---|---|---|---|---|
| NR-AR | 0.7699 | 0.8169 | 0.5336 | 0.5266 |
| NR-AR-LBD | 0.8707 | 0.8395 | 0.6574 | 0.6489 |
| NR-AhR | 0.8878 | 0.8635 | 0.6096 | 0.4975 |
| NR-Aromatase | 0.8099 | 0.7824 | 0.3757 | 0.3858 |
| NR-ER | 0.6820 | 0.7249 | 0.4520 | 0.4411 |
| NR-ER-LBD | 0.7932 | 0.8837 | 0.5178 | 0.5668 |
| NR-PPAR-gamma | 0.8109 | 0.9083 | 0.1625 | 0.5119 |
| SR-ARE | 0.7539 | 0.8154 | 0.4556 | 0.5319 |
| SR-ATAD5 | 0.7611 | 0.8622 | 0.1780 | 0.4755 |
| SR-HSE | 0.7012 | 0.6895 | 0.2626 | 0.3192 |
| SR-MMP | 0.8594 | 0.8719 | 0.6296 | 0.6375 |
| SR-p53 | 0.8179 | 0.7948 | 0.4014 | 0.2969 |
| **Mean** | **0.7932** | **0.8211** | **0.4197** | **0.4950** |

**Success criteria met: 12/12 endpoints achieve AUROC > 0.65** (target was 8/12).

## Installation

```bash
pip install rdkit scikit-learn pandas numpy matplotlib seaborn joblib tqdm
```

If pip fails for RDKit, use conda:
```bash
conda install -c conda-forge rdkit
pip install scikit-learn pandas numpy matplotlib seaborn joblib tqdm
```

## Usage

### Train the model

```bash
python src/train.py
```

This will:
1. Download the Tox21 dataset (~120 KB)
2. Generate Morgan fingerprints for all molecules
3. Train 12 Random Forest classifiers
4. Save metrics to `results/`
5. Save the model to `models/tox21_rf_model.joblib`
6. Generate AUROC bar charts, ROC curves, and confusion matrices

### Predict toxicity for a new molecule

```python
from src.predict import predict_toxicity

result = predict_toxicity("CCO")  # ethanol
for endpoint, info in result.items():
    print(f"{endpoint}: {info['label']} (P={info['probability']:.3f})")
```

**Output format**: each endpoint returns a dict with:
- `prediction`: `1` (toxic), `0` (non-toxic), or `-1` (no model)
- `probability`: float probability of being toxic, or `None`
- `label`: `"TOXIC"`, `"non-toxic"`, or `"unknown"`

### Batch prediction

```python
from src.predict import batch_predict

smiles_list = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
results = batch_predict(smiles_list)
for r in results:
    print(r["smiles"], r["NR-AhR"]["label"])
```

### Load model manually

```python
from src.predict import load_model, predict_toxicity

model = load_model("models/tox21_rf_model.joblib")  # load once
result1 = predict_toxicity("CCO", model=model)
result2 = predict_toxicity("c1ccccc1", model=model)
```

### Example: known compounds

| Compound | NR-AhR | NR-ER | SR-MMP |
|---|---|---|---|
| Ethanol (CCO) | non-toxic (0.09) | non-toxic (0.26) | non-toxic (0.11) |
| Benzene (c1ccccc1) | non-toxic (0.20) | non-toxic (0.34) | non-toxic (0.25) |
| Benzocaine | **TOXIC (0.59)** | **TOXIC (0.51)** | non-toxic (0.29) |

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Featurization | Morgan FP (r=2, 2048 bits) | ECFP4 equivalent; well-validated for RF models |
| Multi-task approach | 12 separate RF models | Different missing-label patterns per endpoint |
| Missing labels | Mask (exclude from training) | Safer than imputation; no label leakage |
| Class imbalance | `class_weight='balanced'` | Automatic reweighting; simpler than oversampling |
| Evaluation metric | AUROC + AUPRC | AUROC for ranking; AUPRC for imbalanced positives |
| Model persistence | joblib (compress=3) | Standard sklearn serialization; ~3x compression |

## Limitations

1. **No molecular 3D structure**: Morgan fingerprints are 2D; 3D conformer-based
   features (e.g., shape, pharmacophore) could improve predictions.

2. **Missing label assumption**: Labels are assumed Missing At Random (MAR).
   If missingness correlates with toxicity, this biases the model.

3. **Applicability domain**: The model has no built-in domain check. Predictions
   on molecules very different from Tox21 training data are unreliable.

4. **Scaffold bias**: Random splitting (not scaffold splitting) inflates
   performance estimates for structurally similar train/test molecules.

5. **Binary predictions only**: Continuous dose-response relationships are
   not modeled.

## Next Steps

- **Scaffold split**: Use Murcko scaffolds to evaluate true generalization.
- **Graph Neural Networks**: GCN/MPNN models typically outperform RF on Tox21.
- **Hyperparameter tuning**: Grid/random search for `max_depth`, `n_estimators`,
  `min_samples_leaf`.
- **Ensemble**: Combine RF with gradient boosting or GNN for improved predictions.
- **Uncertainty quantification**: Report prediction confidence intervals.
- **SHAP values**: Interpret feature importances at the molecular substructure level.
- **Applicability domain**: Add a Tanimoto similarity filter to flag out-of-domain
  molecules.
