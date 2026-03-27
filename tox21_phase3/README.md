# Tox21 Phase 3 — 3D Molecular Toxicity Prediction with Uni-Mol

Fine-tune the **Uni-Mol** pretrained 3D molecular model on Tox21 for multi-endpoint toxicity prediction. This is Phase 3 of a three-phase progression:

| Phase | Representation | Model | Mean AUROC (Random) | Mean AUROC (Scaffold) |
|-------|---------------|-------|--------------------|-----------------------|
| 1 | 1D fingerprints | Random Forest | 0.8211 | — |
| 2 | 2D molecular graph | GATv2 | 0.8178 | 0.7914 |
| 2 | 2D molecular graph | D-MPNN | 0.8524 | 0.7961 |
| **3** | **3D conformer** | **Uni-Mol v1** | **TBD** | **TBD** |
| 3 | 3D conformer | Uni-Mol v2 (bonus) | TBD | TBD |

## Why 3D?

Nuclear receptor binding endpoints (NR-ER, NR-ER-LBD, NR-AR) depend on receptor-ligand spatial fit — exactly where 3D geometry should help. Phase 2 showed NR-ER drops to near-random (~0.59) under scaffold split with 2D models. Uni-Mol's SE(3)-equivariant Transformer, pretrained on 209M molecular conformations, captures 3D shape information that 2D graphs cannot encode.

## Project Structure

```
tox21_phase3/
├── data/
│   └── tox21_prepared.csv          # Cleaned: SMILES + 12 endpoints (NaN = missing)
├── src/
│   ├── prepare_data.py             # Download, validate, format Tox21
│   ├── train_unimol_v1.py          # Uni-Mol v1 (ICLR 2023, 209M conformers)
│   ├── train_unimol_v2.py          # Uni-Mol v2 (NeurIPS 2024, 800M) — optional
│   ├── evaluate.py                 # AUROC/AUPRC/BalAcc per endpoint
│   ├── compare_all_phases.py       # Master comparison across all phases
│   └── visualize.py                # All comparison plots
├── predict.py                      # Multi-model toxicity prediction CLI
├── run_all.py                      # End-to-end pipeline
├── exp/                            # Model checkpoints (created by training)
├── results/                        # CSVs + plots
├── phase2_results/                 # Hardcoded Phase 1/2 results
└── requirements.txt
```

## Installation

```bash
# Requires Python 3.10+
pip install -r requirements.txt

# Or manually:
pip install "numpy<2.0.0" torch rdkit unimol_tools pandas scikit-learn matplotlib seaborn
```

> **Note**: `numpy<2.0.0` is required for RDKit compatibility with unimol_tools.

## Usage

### Full Pipeline (Recommended)

```bash
python run_all.py                    # Train v1 + v2, evaluate, compare, plot
python run_all.py --skip-v2          # Train v1 only (faster)
python run_all.py --epochs 20        # Quick test run
python run_all.py --skip-train       # Rebuild plots from existing results
```

### Individual Steps

```bash
# 1. Prepare data
python src/prepare_data.py

# 2. Train Uni-Mol v1
python src/train_unimol_v1.py --split both
python src/train_unimol_v1.py --split scaffold  # scaffold only

# 3. Train Uni-Mol v2 (optional)
python src/train_unimol_v2.py --split both

# 4. Evaluate
python src/evaluate.py

# 5. Compare all phases
python src/compare_all_phases.py

# 6. Generate plots
python src/visualize.py

# 7. Predict on custom SMILES
python predict.py "CCO"
python predict.py --demo
```

### GPU Notes

- **GPU strongly recommended**: Uni-Mol fine-tuning is ~10x faster on GPU.
- If no GPU / OOM: reduce batch size (`--batch-size 8`) or epochs (`--epochs 20`).
- Uni-Mol v2 requires more VRAM. If it fails, use `--skip-v2`.

## Evaluation

### Metrics
- **AUROC** (primary) — area under ROC curve
- **AUPRC** — area under precision-recall curve (robust to class imbalance)
- **Balanced Accuracy** — at 0.5 threshold

### Split Strategy
- **Scaffold split** (primary evaluation): groups molecules by Murcko scaffold, ensuring test molecules have novel scaffolds not seen in training.
- **Random split**: standard random 80/10/10 for comparison with Phase 1.

> **Note on scaffold split consistency**: This phase uses `unimol_tools`' built-in scaffold split implementation (Murcko decomposition via RDKit). The exact train/val/test assignments may differ slightly from Phase 2's custom split, but the methodology is identical, making comparison fair.

### Output Files

| File | Description |
|------|-------------|
| `results/unimol_v1_scaffold_results.csv` | Per-endpoint metrics (scaffold split) |
| `results/unimol_v1_random_results.csv` | Per-endpoint metrics (random split) |
| `results/all_phases_comparison.csv` | **Master table** — all models, all splits |
| `results/all_phases_mean_auroc.png` | Hero chart: mean AUROC across all models |
| `results/all_phases_heatmap.png` | AUROC heatmap (endpoints × models) |
| `results/3d_vs_2d_improvement.png` | Per-endpoint Uni-Mol vs D-MPNN delta |
| `results/binding_vs_stress_comparison.png` | NR vs SR group comparison |
| `results/scaffold_drop_all_models.png` | Random→scaffold AUROC gap |
| `results/progression_chart.png` | 1D→2D→3D progression story |

## Interpretation Guide

### Expected Outcomes
- **Binding endpoints (NR-ER, NR-ER-LBD, NR-AR)**: Most likely to benefit from 3D geometry.
- **Stress response (SR-ARE, SR-MMP)**: Modest or no improvement — driven by reactive substructures captured by 2D.
- **Scaffold robustness**: Uni-Mol's 209M-conformer pretraining should show a smaller random→scaffold gap.

### What Each Plot Tells You
- **progression_chart.png**: The story in one image — does 3D add value beyond 2D?
- **3d_vs_2d_improvement.png**: Which specific endpoints benefit from 3D?
- **binding_vs_stress_comparison.png**: Does 3D help binding more than stress response? (Hypothesis: yes)
- **scaffold_drop_all_models.png**: Which model generalizes best to novel scaffolds?

## Technical Details

- **Uni-Mol v1**: SE(3)-equivariant Transformer, pretrained on 209M 3D conformations (ICLR 2023)
- **Uni-Mol v2**: Two-track Transformer, pretrained on 800M conformations, 84M–1.1B params (NeurIPS 2024)
- **Conformer generation**: Handled internally by `unimol_tools` (RDKit ETKDG)
- **Missing labels**: Preserved as NaN; `unimol_tools` handles them via its multilabel_classification task
- **Reproducibility**: Seeds set where possible, but conformer generation has inherent randomness

## References

- Uni-Mol (v1): Zhou et al., "Uni-Mol: A Universal 3D Molecular Representation Learning Framework", ICLR 2023
- Uni-Mol2 (v2): Lu et al., "Uni-Mol2: Exploring Molecular Pretraining Model at Scale", NeurIPS 2024
- Tox21 dataset: Mayr et al., via MoleculeNet (Wu et al., 2018)
