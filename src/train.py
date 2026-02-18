"""
train.py
--------
Main training script for the Tox21 Random Forest toxicity model.

Pipeline:
  1. Download / load the Tox21 dataset
  2. Featurize SMILES -> Morgan fingerprints (radius=2, 2048 bits)
  3. Split into train (80%) / val (10%) / test (10%)
  4. Train one Random Forest per toxicity endpoint
  5. Evaluate on validation set (AUROC, AUPRC, balanced accuracy)
  6. Evaluate on test set
  7. Save model and results to disk
  8. Generate visualizations

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


def main():
    logger.info("=" * 60)
    logger.info("TOX21 RANDOM FOREST TOXICITY PREDICTION — TRAINING PIPELINE")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Data acquisition
    # ------------------------------------------------------------------
    logger.info("\n[Step 1] Acquiring Tox21 dataset...")
    file_path = download_tox21()
    df = load_tox21(file_path)
    inspect_dataset(df)

    # ------------------------------------------------------------------
    # Step 2: Preprocessing — fingerprints + train/val/test split
    # ------------------------------------------------------------------
    logger.info("\n[Step 2] Preprocessing: SMILES -> Morgan fingerprints + split...")
    data = prepare_data(df, train_frac=0.8, val_frac=0.1, test_frac=0.1, random_state=42)

    X_train = data["X_train"]
    X_val   = data["X_val"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_val   = data["y_val"]
    y_test  = data["y_test"]
    task_names = data["task_names"]

    logger.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"X_val  : {X_val.shape},   y_val  : {y_val.shape}")
    logger.info(f"X_test : {X_test.shape},  y_test : {y_test.shape}")

    # ------------------------------------------------------------------
    # Step 3: Model training
    # ------------------------------------------------------------------
    logger.info("\n[Step 3] Training Random Forest models (one per endpoint)...")
    model = Tox21RandomForest(task_names=task_names)
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Step 4: Validation set evaluation
    # ------------------------------------------------------------------
    logger.info("\n[Step 4] Evaluating on validation set...")
    y_proba_val = model.predict_proba(X_val)
    val_metrics = evaluate_predictions(y_val, y_proba_val, task_names, split_name="validation")
    save_metrics(val_metrics, split_name="validation")

    # Visualizations for validation set
    plot_auroc_bar(val_metrics, split_name="validation")
    plot_roc_curves(y_val, y_proba_val, task_names, split_name="validation")
    plot_confusion_matrices(y_val, y_proba_val, task_names, split_name="validation")

    # ------------------------------------------------------------------
    # Step 5: Test set evaluation
    # ------------------------------------------------------------------
    logger.info("\n[Step 5] Evaluating on test set...")
    y_proba_test = model.predict_proba(X_test)
    test_metrics = evaluate_predictions(y_test, y_proba_test, task_names, split_name="test")
    save_metrics(test_metrics, split_name="test")

    plot_auroc_bar(test_metrics, split_name="test")
    plot_roc_curves(y_test, y_proba_test, task_names, split_name="test")
    plot_confusion_matrices(y_test, y_proba_test, task_names, split_name="test")

    # ------------------------------------------------------------------
    # Step 6: Save model
    # ------------------------------------------------------------------
    logger.info("\n[Step 6] Saving model to disk...")
    model_path = model.save()
    logger.info(f"Model saved: {model_path}")

    # ------------------------------------------------------------------
    # Step 7: Quick demo prediction on test set examples
    # ------------------------------------------------------------------
    logger.info("\n[Step 7] Demo predictions on 3 test set molecules...")
    _demo_predictions(model, data["smiles_test"], y_test, task_names)

    # ------------------------------------------------------------------
    # Step 8: Print final summary
    # ------------------------------------------------------------------
    _print_final_summary(val_metrics, test_metrics)

    logger.info("\nTraining pipeline complete!")
    logger.info(f"Results saved in: {os.path.abspath(RESULTS_DIR)}")
    logger.info(f"Model saved in:   {os.path.abspath(MODEL_DIR)}")


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


def _print_final_summary(val_metrics: pd.DataFrame, test_metrics: pd.DataFrame) -> None:
    """Print a final performance summary table."""
    print("\n" + "=" * 70)
    print("FINAL PERFORMANCE SUMMARY")
    print("=" * 70)

    summary = pd.DataFrame({
        "Endpoint":   val_metrics["Endpoint"],
        "Val_AUROC":  val_metrics["AUROC"],
        "Test_AUROC": test_metrics["AUROC"],
        "Val_AUPRC":  val_metrics["AUPRC"],
        "Test_AUPRC": test_metrics["AUPRC"],
    })
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    valid_val = val_metrics["AUROC"].dropna()
    valid_test = test_metrics["AUROC"].dropna()

    print(f"\n  Mean Validation AUROC: {valid_val.mean():.4f}")
    print(f"  Mean Test AUROC      : {valid_test.mean():.4f}")
    print(f"  Val endpoints >= 0.65: {(valid_val >= 0.65).sum()}/{len(valid_val)}")
    print(f"  Test endpoints >= 0.65: {(valid_test >= 0.65).sum()}/{len(valid_test)}")

    # Success criterion check
    n_above_threshold = (valid_test >= 0.65).sum()
    if n_above_threshold >= 8:
        print(f"\n  SUCCESS: {n_above_threshold}/12 endpoints achieve AUROC >= 0.65")
    else:
        print(f"\n  NOTE: {n_above_threshold}/12 endpoints achieve AUROC >= 0.65 "
              f"(target: 8/12)")


if __name__ == "__main__":
    main()
