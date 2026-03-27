#!/usr/bin/env python3
"""
Tox21 Phase 3 — Uni-Mol v2 Training

Fine-tunes the Uni-Mol v2 pretrained model (800M conformations, NeurIPS 2024)
on Tox21 for multi-endpoint toxicity prediction.

This is a BONUS experiment. If Uni-Mol v2 is unavailable or OOM, this script
prints a clear warning and exits gracefully.
"""

import argparse
import pathlib
import sys
import time
import traceback

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_CSV = PROJECT_ROOT / "data" / "tox21_prepared.csv"
EXP_DIR = PROJECT_ROOT / "exp"

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


def train_unimol_v2(
    split: str,
    model_size: str = "84m",
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    early_stopping: int = 10,
    remove_hs: bool = False,
) -> pathlib.Path | None:
    """
    Train Uni-Mol v2 on Tox21 with the specified split.

    Returns save_path on success, or None on failure.
    """
    try:
        from unimol_tools import MolTrain
    except ImportError:
        print("  ERROR: unimol_tools not installed. Run: pip install unimol_tools")
        return None

    save_path = str(EXP_DIR / f"unimol_v2_{split}")

    print(f"\n{'='*60}")
    print(f"  Training Uni-Mol v2 ({model_size})  |  Split: {split.upper()}")
    print(f"{'='*60}")
    print(f"  Data:           {DATA_CSV}")
    print(f"  Save path:      {save_path}")
    print(f"  Model size:     {model_size}")
    print(f"  Epochs:         {epochs}")
    print(f"  Batch size:     {batch_size}")
    print(f"  Learning rate:  {learning_rate}")
    print()

    if not DATA_CSV.exists():
        print(f"  ERROR: Data file not found: {DATA_CSV}")
        print(f"  Run src/prepare_data.py first.")
        return None

    t0 = time.time()

    try:
        clf = MolTrain(
            task='multilabel_classification',
            data_type='molecule',
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            early_stopping=early_stopping,
            metrics='auc',
            split=split,
            kfold=1,
            save_path=save_path,
            remove_hs=remove_hs,
            smiles_col='smiles',
            target_cols=TOX21_TASKS,
            model_name='unimolv2',
            model_size=model_size,
        )

        clf.fit(data=str(DATA_CSV))

        elapsed = time.time() - t0
        print(f"\n  [OK] Uni-Mol v2 [{split}] training complete in {elapsed/60:.1f} min")
        print(f"    Model saved to: {save_path}")
        return pathlib.Path(save_path)

    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            print(f"\n  [!] GPU MEMORY ERROR during Uni-Mol v2 [{split}] training:")
            print(f"    {e}")
            print(f"\n  Suggestions:")
            print(f"    1. Reduce batch_size (current: {batch_size})")
            print(f"    2. Try a smaller model_size")
            print(f"    3. Skip v2 with --skip-v2 flag")
            return None
        raise

    except Exception as e:
        print(f"\n  [!] FAILED: Uni-Mol v2 [{split}] training:")
        print(f"    {type(e).__name__}: {e}")
        traceback.print_exc()
        print(f"\n  Uni-Mol v2 is a bonus experiment. Continuing with v1 results only.")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Train Uni-Mol v2 on Tox21 (scaffold + random splits)"
    )
    parser.add_argument(
        "--split", choices=["scaffold", "random", "both"], default="both",
        help="Which split to run (default: both)"
    )
    parser.add_argument("--model-size", default="84m",
                        help="Model size: 84m, 164m (default: 84m)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--early-stopping", type=int, default=10)
    args = parser.parse_args()

    splits = ["scaffold", "random"] if args.split == "both" else [args.split]
    success_count = 0

    for split in splits:
        result = train_unimol_v2(
            split=split,
            model_size=args.model_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            early_stopping=args.early_stopping,
        )
        if result is not None:
            success_count += 1

    if success_count == 0:
        print("\n  [!] No Uni-Mol v2 runs completed successfully.")
        print("    Phase 3 comparison will proceed with Uni-Mol v1 only.")
    else:
        print(f"\n  [OK] {success_count}/{len(splits)} Uni-Mol v2 runs completed.")


if __name__ == "__main__":
    main()
