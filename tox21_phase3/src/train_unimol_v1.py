#!/usr/bin/env python3
"""
Tox21 Phase 3 — Uni-Mol v1 Training

Fine-tunes the Uni-Mol v1 pretrained model (209M conformations, ICLR 2023)
on Tox21 for multi-endpoint toxicity prediction.

Runs twice: scaffold split + random split.
"""

import argparse
import pathlib
import sys
import time

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_CSV = PROJECT_ROOT / "data" / "tox21_prepared.csv"
EXP_DIR = PROJECT_ROOT / "exp"

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


def train_unimol_v1(
    split: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    early_stopping: int = 10,
    remove_hs: bool = False,
) -> pathlib.Path:
    """
    Train Uni-Mol v1 on Tox21 with the specified split.

    Parameters
    ----------
    split : str
        'scaffold' or 'random'
    epochs : int
        Maximum training epochs
    batch_size : int
        Training batch size
    learning_rate : float
        Learning rate
    early_stopping : int
        Early stopping patience
    remove_hs : bool
        Whether to remove hydrogens (False = keep for richer 3D)

    Returns
    -------
    save_path : pathlib.Path
        Directory where model and predictions were saved
    """
    from unimol_tools import MolTrain

    save_path = str(EXP_DIR / f"unimol_v1_{split}")

    print(f"\n{'='*60}")
    print(f"  Training Uni-Mol v1  |  Split: {split.upper()}")
    print(f"{'='*60}")
    print(f"  Data:           {DATA_CSV}")
    print(f"  Save path:      {save_path}")
    print(f"  Epochs:         {epochs}")
    print(f"  Batch size:     {batch_size}")
    print(f"  Learning rate:  {learning_rate}")
    print(f"  Early stopping: {early_stopping}")
    print(f"  Remove H:       {remove_hs}")
    print()

    if not DATA_CSV.exists():
        print(f"  ERROR: Data file not found: {DATA_CSV}")
        print(f"  Run src/prepare_data.py first.")
        sys.exit(1)

    t0 = time.time()

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
        model_name='unimolv1',
    )

    clf.fit(data=str(DATA_CSV))

    elapsed = time.time() - t0
    print(f"\n  [OK] Uni-Mol v1 [{split}] training complete in {elapsed/60:.1f} min")
    print(f"    Model saved to: {save_path}")

    return pathlib.Path(save_path)


def main():
    parser = argparse.ArgumentParser(
        description="Train Uni-Mol v1 on Tox21 (scaffold + random splits)"
    )
    parser.add_argument(
        "--split", choices=["scaffold", "random", "both"], default="both",
        help="Which split to run (default: both)"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--early-stopping", type=int, default=10)
    args = parser.parse_args()

    splits = ["scaffold", "random"] if args.split == "both" else [args.split]

    for split in splits:
        train_unimol_v1(
            split=split,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            early_stopping=args.early_stopping,
        )

    print("\n  All Uni-Mol v1 training runs complete.")


if __name__ == "__main__":
    main()
