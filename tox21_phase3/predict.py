#!/usr/bin/env python3
"""
Tox21 Phase 3 — Multi-Model Prediction Interface

Takes a SMILES string and produces toxicity predictions from:
  - Uni-Mol v1 (3D pretrained, primary Phase 3 model)
  - Ensemble of available models

Highlights endpoints with high disagreement between models, flagging
molecules where 3D geometry significantly affects the prediction.

Usage:
    python predict.py "CCO"
    python predict.py "CC(=O)Oc1ccccc1C(=O)O"
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
EXP_DIR = PROJECT_ROOT / "exp"

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

DISAGREEMENT_THRESHOLD = 0.2  # Flag if model predictions differ by >0.2


def load_unimol_predictor(exp_name: str):
    """Load a trained Uni-Mol predictor."""
    exp_path = EXP_DIR / exp_name
    if not exp_path.exists():
        return None
    try:
        from unimol_tools import MolPredict
        return MolPredict(load_model=str(exp_path))
    except Exception as e:
        print(f"  [!] Could not load {exp_name}: {e}")
        return None


def predict_unimol(predictor, smiles: str) -> dict | None:
    """Run Uni-Mol prediction on a single SMILES string."""
    try:
        # MolPredict expects a CSV or DataFrame
        temp_df = pd.DataFrame({"smiles": [smiles]})
        temp_csv = PROJECT_ROOT / "tmp_predict.csv"
        temp_df.to_csv(temp_csv, index=False)
        result = predictor.predict(data=str(temp_csv))
        temp_csv.unlink(missing_ok=True)

        if isinstance(result, pd.DataFrame):
            probs = result.values[0]
        else:
            probs = np.array(result)[0]

        return {task: float(probs[i]) for i, task in enumerate(TOX21_TASKS)}
    except Exception as e:
        print(f"  [!] Prediction failed: {e}")
        return None


def predict_all_models(smiles: str) -> dict:
    """
    Run all available models on a SMILES string.

    Returns dict with:
      - model_name → {endpoint → probability}
      - 'ensemble' → mean of all model probabilities
      - 'disagreement' → endpoints with high cross-model variance
    """
    results = {}

    # Uni-Mol v1 (primary)
    v1_pred = load_unimol_predictor("unimol_v1_scaffold")
    if v1_pred is not None:
        preds = predict_unimol(v1_pred, smiles)
        if preds:
            results["Uni-Mol v1 (3D)"] = preds

    # Uni-Mol v2 (bonus)
    v2_pred = load_unimol_predictor("unimol_v2_scaffold")
    if v2_pred is not None:
        preds = predict_unimol(v2_pred, smiles)
        if preds:
            results["Uni-Mol v2 (3D)"] = preds

    if not results:
        print("  No trained models found. Run training first.")
        return {}

    # Ensemble: mean of all model probabilities
    all_probs = {task: [] for task in TOX21_TASKS}
    for model_preds in results.values():
        for task in TOX21_TASKS:
            if task in model_preds:
                all_probs[task].append(model_preds[task])

    ensemble = {}
    disagreement = {}
    for task in TOX21_TASKS:
        if all_probs[task]:
            ensemble[task] = float(np.mean(all_probs[task]))
            if len(all_probs[task]) > 1:
                spread = max(all_probs[task]) - min(all_probs[task])
                if spread > DISAGREEMENT_THRESHOLD:
                    disagreement[task] = round(spread, 4)

    results["Ensemble"] = ensemble
    results["_disagreement"] = disagreement

    return results


def print_predictions(smiles: str, results: dict):
    """Pretty-print multi-model predictions."""
    print(f"\n{'='*70}")
    print(f"  SMILES: {smiles}")
    print(f"{'='*70}")

    if not results:
        print("  No predictions available.")
        return

    # Get model names (exclude ensemble and metadata)
    model_names = [k for k in results if not k.startswith("_") and k != "Ensemble"]

    # Header
    header = f"  {'Endpoint':<18}"
    for name in model_names:
        header += f" {name:>16}"
    header += f" {'Ensemble':>10}  {'Flag':>4}"
    print(header)
    print(f"  {'-'*len(header)}")

    disagreement = results.get("_disagreement", {})

    for task in TOX21_TASKS:
        row = f"  {task:<18}"
        for name in model_names:
            prob = results.get(name, {}).get(task, float("nan"))
            row += f" {prob:>16.4f}"

        ens_prob = results.get("Ensemble", {}).get(task, float("nan"))
        row += f" {ens_prob:>10.4f}"

        if task in disagreement:
            row += f"  [!] D={disagreement[task]:.3f}"

        print(row)

    # Summary
    ens = results.get("Ensemble", {})
    if ens:
        toxic_eps = [t for t in TOX21_TASKS if ens.get(t, 0) >= 0.5]
        if toxic_eps:
            print(f"\n  [!] Predicted TOXIC for: {', '.join(toxic_eps)}")
        else:
            print(f"\n  [OK] No endpoints predicted toxic (all probs < 0.5)")

    if disagreement:
        print(f"\n  [!] High model disagreement on: {', '.join(disagreement.keys())}")
        print(f"    These are endpoints where 3D geometry may significantly affect prediction.")


# ── Demo compounds ───────────────────────────────────────────────────────────

DEMO_COMPOUNDS = [
    ("Ethanol",       "CCO"),
    ("Aspirin",       "CC(=O)Oc1ccccc1C(=O)O"),
    ("Bisphenol A",   "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1"),
    ("Dioxin (TCDD)", "Clc1cc2oc3cc(Cl)c(Cl)cc3oc2cc1Cl"),
    ("Tamoxifen",     "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1"),
]


def main():
    parser = argparse.ArgumentParser(
        description="Multi-model toxicity prediction for SMILES strings"
    )
    parser.add_argument(
        "smiles", nargs="?", default=None,
        help="SMILES string to predict. If not given, runs demo compounds."
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run predictions on demo compounds"
    )
    args = parser.parse_args()

    if args.smiles:
        results = predict_all_models(args.smiles)
        print_predictions(args.smiles, results)
    elif args.demo or args.smiles is None:
        print("\n" + "=" * 70)
        print("  Multi-Model Toxicity Predictions — Phase 3")
        print("=" * 70)
        for name, smi in DEMO_COMPOUNDS:
            print(f"\n  Compound: {name}")
            results = predict_all_models(smi)
            print_predictions(smi, results)


if __name__ == "__main__":
    main()
