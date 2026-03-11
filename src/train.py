"""
train.py
--------
Main training script for the Tox21 Random Forest toxicity model.

Pipeline (run twice — random split and scaffold split):
  1. Download / load the Tox21 dataset
  2. Featurize SMILES -> Morgan fingerprints (radius=2, 2048 bits)
  3. Split into train (80%) / val (10%) / test (10%)
       - run_pipeline(..., split_type="random")   -> stratified random split
       - run_pipeline(..., split_type="scaffold")  -> Murcko scaffold split
  4. Train one Random Forest per toxicity endpoint
  5. Evaluate on validation set (AUROC, AUPRC, balanced accuracy)
  6. Evaluate on test set
  7. Save model and results to disk
  8. Generate visualizations
  9. Print side-by-side comparison of both splits

Run:
    python src/train.py
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

# Make src/ importable from any working directory
sys.path.insert(0, os.path.dirname(__file__))

from data_acquisition import load_tox21, download_tox21, inspect_dataset
from preprocessing import prepare_data
from model import Tox21RandomForest
from evaluation import (
    evaluate_predictions,
    plot_auroc_bar,
    plot_roc_curves,
    plot_confusion_matrices,
    save_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def run_pipeline(df: pd.DataFrame, split_type: str) -> dict:
    """
    Run the full training & evaluation pipeline for a given split type.

    Args:
        df         : Raw Tox21 DataFrame.
        split_type : 'random' or 'scaffold'.

    Returns:
        dict with 'val_metrics' and 'test_metrics' DataFrames.
    """
    tag = split_type  # used as suffix in file names

    logger.info(f"\n{'='*60}")
    logger.info(f"SPLIT TYPE: {split_type.upper()}")
    logger.info(f"{'='*60}")

    # ------------------------------------------------------------------
    # Step 1: Preprocessing — fingerprints + split
    # ------------------------------------------------------------------
    logger.info(f"\n[{tag}] Preprocessing: SMILES -> Morgan fingerprints + split...")
    data = prepare_data(
        df,
        train_frac=0.8,
        val_frac=0.1,
        test_frac=0.1,
        random_state=42,
        split_type=split_type,
    )

    X_train    = data["X_train"]
    X_val      = data["X_val"]
    X_test     = data["X_test"]
    y_train    = data["y_train"]
    y_val      = data["y_val"]
    y_test     = data["y_test"]
    task_names = data["task_names"]

    logger.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"X_val  : {X_val.shape},   y_val  : {y_val.shape}")
    logger.info(f"X_test : {X_test.shape},  y_test : {y_test.shape}")

    # ------------------------------------------------------------------
    # Step 2: Model training
    # ------------------------------------------------------------------
    logger.info(f"\n[{tag}] Training Random Forest models (one per endpoint)...")
    model = Tox21RandomForest(task_names=task_names)
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Step 3: Validation set evaluation
    # ------------------------------------------------------------------
    logger.info(f"\n[{tag}] Evaluating on validation set...")
    y_proba_val = model.predict_proba(X_val)
    val_metrics = evaluate_predictions(
        y_val, y_proba_val, task_names,
        split_name=f"{tag}_validation"
    )
    save_metrics(val_metrics, split_name=f"{tag}_validation")
    plot_auroc_bar(val_metrics, split_name=f"{tag}_validation")
    plot_roc_curves(y_val, y_proba_val, task_names, split_name=f"{tag}_validation")
    plot_confusion_matrices(y_val, y_proba_val, task_names, split_name=f"{tag}_validation")

    # ------------------------------------------------------------------
    # Step 4: Test set evaluation
    # ------------------------------------------------------------------
    logger.info(f"\n[{tag}] Evaluating on test set...")
    y_proba_test = model.predict_proba(X_test)
    test_metrics = evaluate_predictions(
        y_test, y_proba_test, task_names,
        split_name=f"{tag}_test"
    )
    save_metrics(test_metrics, split_name=f"{tag}_test")
    plot_auroc_bar(test_metrics, split_name=f"{tag}_test")
    plot_roc_curves(y_test, y_proba_test, task_names, split_name=f"{tag}_test")
    plot_confusion_matrices(y_test, y_proba_test, task_names, split_name=f"{tag}_test")

    # ------------------------------------------------------------------
    # Step 5: Save model
    # ------------------------------------------------------------------
    logger.info(f"\n[{tag}] Saving model to disk...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"tox21_rf_{tag}.joblib")
    model.save(model_path)
    logger.info(f"Model saved: {model_path}")

    # ------------------------------------------------------------------
    # Step 6: Demo predictions
    # ------------------------------------------------------------------
    logger.info(f"\n[{tag}] Demo predictions on 3 test set molecules...")
    _demo_predictions(model, data["smiles_test"], y_test, task_names)

    return {"val_metrics": val_metrics, "test_metrics": test_metrics}


def main():
    logger.info("=" * 60)
    logger.info("TOX21 RANDOM FOREST TOXICITY PREDICTION — TRAINING PIPELINE")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Acquire dataset (shared across both runs)
    # ------------------------------------------------------------------
    logger.info("\n[Step 1] Acquiring Tox21 dataset...")
    file_path = download_tox21()
    df = load_tox21(file_path)
    inspect_dataset(df)

    # ------------------------------------------------------------------
    # Run 1: Random (stratified) split
    # ------------------------------------------------------------------
    random_results = run_pipeline(df, split_type="random")

    # ------------------------------------------------------------------
    # Run 2: Scaffold split
    # ------------------------------------------------------------------
    scaffold_results = run_pipeline(df, split_type="scaffold")

    # ------------------------------------------------------------------
    # Final comparison
    # ------------------------------------------------------------------
    _print_comparison(
        random_results["val_metrics"],   random_results["test_metrics"],
        scaffold_results["val_metrics"], scaffold_results["test_metrics"],
    )

    logger.info("\nTraining pipeline complete!")
    logger.info(f"Results saved in: {os.path.abspath(RESULTS_DIR)}")
    logger.info(f"Models saved in:  {os.path.abspath(MODEL_DIR)}")


def _demo_predictions(model, smiles_test, y_test, task_names):
    """Run and display predictions for 3 molecules from the test set."""
    from preprocessing import smiles_to_morgan

    n_demo = min(3, len(smiles_test))
    for idx in range(n_demo):
        smi = smiles_test[idx]
        fp = smiles_to_morgan(smi)
        if fp is None:
            continue

        X = fp.reshape(1, -1)
        proba = model.predict_proba(X)[0]

        print(f"\nDemo molecule [{idx+1}]: {smi}")
        print(f"  {'Endpoint':<18} {'True':>6} {'P(tox)':>8} {'Pred':>6}")
        print(f"  {'-'*44}")
        for i, task in enumerate(task_names):
            true_val = y_test[idx, i]
            true_str = str(int(true_val)) if not np.isnan(true_val) else " NaN"
            prob_str = f"{proba[i]:.4f}" if not np.isnan(proba[i]) else "  N/A"
            pred = 1 if (not np.isnan(proba[i]) and proba[i] >= 0.5) else 0
            print(f"  {task:<18} {true_str:>6} {prob_str:>8} {pred:>6}")


def _print_comparison(
    rand_val: pd.DataFrame,  rand_test: pd.DataFrame,
    scaf_val: pd.DataFrame,  scaf_test: pd.DataFrame,
) -> None:
    """Print a side-by-side AUROC comparison of random vs scaffold splits."""
    print("\n" + "=" * 80)
    print("SPLIT COMPARISON — AUROC (Random vs Scaffold)")
    print("=" * 80)

    header = (
        f"{'Endpoint':<18} "
        f"{'Rand_Val':>10} {'Rand_Test':>10} "
        f"{'Scaf_Val':>10} {'Scaf_Test':>10} "
        f"{'Δ Test':>8}"
    )
    print(header)
    print("-" * len(header))

    for i in range(len(rand_test)):
        endpoint = rand_test.iloc[i]["Endpoint"]
        rv = rand_val.iloc[i]["AUROC"]
        rt = rand_test.iloc[i]["AUROC"]
        sv = scaf_val.iloc[i]["AUROC"]
        st = scaf_test.iloc[i]["AUROC"]
        delta = st - rt if (not np.isnan(st) and not np.isnan(rt)) else float("nan")

        def fmt(v):
            return f"{v:.4f}" if not np.isnan(v) else "  N/A"

        print(
            f"  {endpoint:<16} "
            f"{fmt(rv):>10} {fmt(rt):>10} "
            f"{fmt(sv):>10} {fmt(st):>10} "
            f"{fmt(delta):>8}"
        )

    print("-" * len(header))

    for label, val_m, test_m in [
        ("Random  ", rand_val, rand_test),
        ("Scaffold", scaf_val, scaf_test),
    ]:
        v_auroc = val_m["AUROC"].dropna().mean()
        t_auroc = test_m["AUROC"].dropna().mean()
        n_pass  = (test_m["AUROC"].dropna() >= 0.65).sum()
        n_total = test_m["AUROC"].dropna().count()
        print(
            f"  {label}  Mean Val AUROC: {v_auroc:.4f}  "
            f"Mean Test AUROC: {t_auroc:.4f}  "
            f"Endpoints >= 0.65: {n_pass}/{n_total}"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()
