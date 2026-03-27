#!/usr/bin/env python3
"""
Tox21 Phase 3 — Master Pipeline

End-to-end orchestration: data prep → train Uni-Mol → evaluate → compare → visualize.

Usage:
    python run_all.py                  # Full pipeline (v1 + v2)
    python run_all.py --skip-v2        # Skip Uni-Mol v2 (v1 only)
    python run_all.py --skip-train     # Only rebuild tables/plots from existing results
    python run_all.py --epochs 20      # Faster training for testing
    python run_all.py --split scaffold # Run only scaffold split
"""

import argparse
import pathlib
import sys
import time

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


def section(title: str):
    print(f"\n{'═'*65}")
    print(f"  {title}")
    print(f"{'═'*65}")


def main(args):
    t_start = time.time()

    # ── 1. Data Preparation ──────────────────────────────────────────────────
    if not args.skip_train:
        section("STEP 1: Preparing Tox21 Data")
        from prepare_data import prepare_data
        df = prepare_data()
        print(f"  Prepared {len(df)} molecules.")

    # ── 2. Train Uni-Mol v1 ──────────────────────────────────────────────────
    if not args.skip_train:
        section("STEP 2: Training Uni-Mol v1")
        from train_unimol_v1 import train_unimol_v1

        splits = (["scaffold", "random"] if args.split == "both"
                  else [args.split])

        for split in splits:
            train_unimol_v1(
                split=split,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                early_stopping=args.early_stopping,
            )

    # ── 3. Train Uni-Mol v2 (optional) ───────────────────────────────────────
    if not args.skip_train and not args.skip_v2:
        section("STEP 3: Training Uni-Mol v2 (Bonus)")
        from train_unimol_v2 import train_unimol_v2

        splits = (["scaffold", "random"] if args.split == "both"
                  else [args.split])

        for split in splits:
            train_unimol_v2(
                split=split,
                model_size=args.model_size,
                epochs=args.epochs,
                batch_size=max(8, args.batch_size // 2),  # Smaller for v2
                learning_rate=args.lr,
                early_stopping=args.early_stopping,
            )
    elif args.skip_v2:
        print("\n  Skipping Uni-Mol v2 (--skip-v2 flag).")

    # ── 4. Evaluate ──────────────────────────────────────────────────────────
    section("STEP 4: Evaluating Trained Models")
    from evaluate import evaluate_all
    evaluated = evaluate_all()

    # ── 5. Build Comparison Tables ───────────────────────────────────────────
    section("STEP 5: Building Cross-Phase Comparison")
    from compare_all_phases import build_master_comparison, save_master_comparison
    master = build_master_comparison()
    save_master_comparison(master)

    # ── 6. Generate Visualizations ───────────────────────────────────────────
    section("STEP 6: Generating Visualizations")
    from visualize import generate_all_plots
    generate_all_plots()

    # ── 7. Summary ───────────────────────────────────────────────────────────
    section("PIPELINE COMPLETE")
    elapsed = time.time() - t_start
    print(f"  Total runtime: {elapsed/60:.1f} min")

    results_dir = PROJECT_ROOT / "results"
    if results_dir.exists():
        print(f"\n  Generated files in {results_dir}/:")
        for f in sorted(results_dir.glob("*")):
            size_kb = f.stat().st_size / 1024
            print(f"    {f.name:<45} ({size_kb:.1f} KB)")

    # Print mean AUROCs
    mean_row = master[master["endpoint"] == "Mean"]
    if not mean_row.empty:
        print(f"\n  Mean AUROC Summary:")
        for col in mean_row.columns:
            if col != "endpoint":
                val = mean_row[col].values[0]
                if pd.notna(val):
                    print(f"    {col:<30} {val:.4f}")

    print()


def parse_args():
    p = argparse.ArgumentParser(
        description="Tox21 Phase 3 — Uni-Mol 3D Toxicity Prediction Pipeline"
    )
    p.add_argument(
        "--split", choices=["scaffold", "random", "both"], default="both",
        help="Which split strategy (default: both)"
    )
    p.add_argument("--epochs", type=int, default=50,
                   help="Max training epochs (default: 50)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size for Uni-Mol v1 (default: 32)")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Learning rate (default: 1e-4)")
    p.add_argument("--early-stopping", type=int, default=10,
                   help="Early stopping patience (default: 10)")
    p.add_argument("--model-size", default="84m",
                   help="Uni-Mol v2 model size: 84m, 164m (default: 84m)")
    p.add_argument("--skip-v2", action="store_true",
                   help="Skip Uni-Mol v2 training (run v1 only)")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training; rebuild tables/plots from existing results")
    return p.parse_args()


if __name__ == "__main__":
    import pandas as pd  # needed for summary printing
    args = parse_args()
    main(args)
