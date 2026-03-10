#!/usr/bin/env python3
"""
Tox21 Phase 2 — Full GNN Pipeline

Runs both GATv2 and D-MPNN under random and scaffold splits, then generates
the master comparison table and all required visualizations.

Usage:
    python run_all.py                     # default settings
    python run_all.py --epochs 50         # faster run for testing
    python run_all.py --model gatv2       # train only GATv2
    python run_all.py --model dmpnn       # train only D-MPNN
    python run_all.py --split random      # run only random split
    python run_all.py --split scaffold    # run only scaffold split
    python run_all.py --skip-train        # only rebuild tables/plots from saved CSVs
"""

import argparse
import pathlib
import random
import sys
import time

import numpy as np
import torch

# Ensure src/ is on the Python path
SRC_DIR = pathlib.Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def section(title: str):
    print(f"\n{'═'*65}")
    print(f"  {title}")
    print(f"{'═'*65}")


def main(args):
    set_seeds(42)
    t_start = time.time()

    # ── 1. Load & validate dataset ────────────────────────────────────────────
    section("STEP 1: Loading Tox21 dataset")
    from data_loading import load_tox21_df, TOX21_TASKS
    df, smiles_col = load_tox21_df()

    # ── 2. Featurize all molecules into PyG graphs ────────────────────────────
    section("STEP 2: Featurizing molecules → PyG graphs")
    from featurization import build_graph_list
    all_graphs, invalid = build_graph_list(df, smiles_col, verbose=True)
    print(f"  Valid graphs: {len(all_graphs)} | Invalid: {len(invalid)}")

    # ── 3. Generate splits ────────────────────────────────────────────────────
    section("STEP 3: Generating train/val/test splits")
    from splitting import random_split, scaffold_split

    splits = {}

    if args.split in ("random", "both"):
        print("\n[Random Split 80/10/10]")
        train_r, val_r, test_r = random_split(df, smiles_col, seed=42)
        print(f"  Random — train: {len(train_r)} | val: {len(val_r)} | test: {len(test_r)}")

        # Map DataFrame indices to pre-built graph objects
        # (build_graph_list preserves order after dropping invalid rows)
        # We rebuild graph lists per split from the DataFrames
        from featurization import build_graph_list as bgl
        train_graphs_r, _ = bgl(train_r, smiles_col, verbose=False)
        val_graphs_r,   _ = bgl(val_r,   smiles_col, verbose=False)
        test_graphs_r,  _ = bgl(test_r,  smiles_col, verbose=False)
        splits["random"] = (train_graphs_r, val_graphs_r, test_graphs_r)

    if args.split in ("scaffold", "both"):
        print("\n[Scaffold Split 80/10/10]")
        train_s, val_s, test_s = scaffold_split(df, smiles_col, seed=42)
        from featurization import build_graph_list as bgl
        train_graphs_s, _ = bgl(train_s, smiles_col, verbose=False)
        val_graphs_s,   _ = bgl(val_s,   smiles_col, verbose=False)
        test_graphs_s,  _ = bgl(test_s,  smiles_col, verbose=False)
        splits["scaffold"] = (train_graphs_s, val_graphs_s, test_graphs_s)

    # ── 4. Train & evaluate GATv2 ─────────────────────────────────────────────
    from evaluate import evaluate_predictions, save_results, print_results_table
    from train_gatv2 import train_gatv2, collect_predictions_gatv2

    gatv2_histories = {}
    gatv2_results   = {}  # split_name → (y_true, y_prob)

    if args.model in ("gatv2", "both") and not args.skip_train:
        section("STEP 4: Training GATv2")

        for split_name, (tr_g, va_g, te_g) in splits.items():
            print(f"\n── GATv2 [{split_name.upper()} SPLIT] ──")
            model, history, device = train_gatv2(
                tr_g, va_g,
                split_name=split_name,
                epochs=args.epochs,
                batch_size=64,
                hidden_dim=128,
                num_heads=8,
                num_layers=3,
                dropout=0.3,
                lr=1e-3,
                patience=args.patience,
                seed=42,
                verbose=True,
            )
            gatv2_histories[split_name] = history

            # Evaluate on test set
            y_true, y_prob = collect_predictions_gatv2(model, te_g, device)
            gatv2_results[split_name] = (y_true, y_prob)

            metrics_df = evaluate_predictions(y_true, y_prob)
            print_results_table(metrics_df, f"GATv2 [{split_name}] Test Set")
            save_results(metrics_df, f"results_gatv2_{split_name}.csv")

    # ── 5. Train & evaluate D-MPNN ────────────────────────────────────────────
    from train_dmpnn import train_dmpnn, collect_predictions_dmpnn

    dmpnn_results = {}

    if args.model in ("dmpnn", "both") and not args.skip_train:
        section("STEP 5: Training D-MPNN")

        for split_name, (tr_g, va_g, te_g) in splits.items():
            print(f"\n── D-MPNN [{split_name.upper()} SPLIT] ──")
            model, history, device = train_dmpnn(
                tr_g, va_g,
                split_name=split_name,
                epochs=args.epochs,
                batch_size=50,
                hidden_dim=300,
                num_layers=3,
                dropout=0.15,
                lr=1e-3,
                patience=args.patience,
                seed=42,
                verbose=True,
            )

            y_true, y_prob = collect_predictions_dmpnn(model, te_g, device)
            dmpnn_results[split_name] = (y_true, y_prob)

            metrics_df = evaluate_predictions(y_true, y_prob)
            print_results_table(metrics_df, f"D-MPNN [{split_name}] Test Set")
            save_results(metrics_df, f"results_dmpnn_{split_name}.csv")

    # ── 6. Build comparison tables ────────────────────────────────────────────
    section("STEP 6: Building master comparison table")
    from evaluate import build_comparison_table, save_comparison_tables

    result_files = {
        "GATv2_Random":    "results_gatv2_random.csv",
        "GATv2_Scaffold":  "results_gatv2_scaffold.csv",
        "DMPNN_Random":    "results_dmpnn_random.csv",
        "DMPNN_Scaffold":  "results_dmpnn_scaffold.csv",
    }
    auroc_df, auprc_df = build_comparison_table(result_files)
    save_comparison_tables(auroc_df, auprc_df)

    # ── 7. Visualizations ────────────────────────────────────────────────────
    section("STEP 7: Generating visualizations")
    from visualize import (
        plot_auroc_comparison,
        plot_random_vs_scaffold_delta,
        plot_roc_curves,
        plot_confusion_matrices,
        plot_training_curves,
    )

    # Plot 1: Grouped AUROC bar
    plot_auroc_comparison(auroc_df)

    # Plot 2: Random vs scaffold delta
    plot_random_vs_scaffold_delta(auroc_df)

    # Plots 3 & 4: Best scaffold model (ROC + confusion matrices)
    # Determine which scaffold model has higher mean AUROC
    best_scaffold_model = None
    best_scaffold_auroc = -1.0
    best_scaffold_data  = None

    scaffold_candidates = []
    if "random" in gatv2_results and "scaffold" in gatv2_results:
        pass  # we want scaffold split specifically
    for model_name, results_dict in [("GATv2", gatv2_results), ("D-MPNN", dmpnn_results)]:
        if "scaffold" in results_dict:
            y_true, y_prob = results_dict["scaffold"]
            from evaluate import evaluate_predictions as ep
            mdf = ep(y_true, y_prob)
            mean_auroc = mdf["AUROC"].mean()
            print(f"  {model_name} scaffold mean AUROC: {mean_auroc:.4f}")
            if mean_auroc > best_scaffold_auroc:
                best_scaffold_auroc = mean_auroc
                best_scaffold_model = model_name
                best_scaffold_data  = (y_true, y_prob)

    if best_scaffold_data is not None:
        print(f"\n  Best scaffold model: {best_scaffold_model} (AUROC={best_scaffold_auroc:.4f})")
        y_true, y_prob = best_scaffold_data
        plot_roc_curves(y_true, y_prob, best_scaffold_model, TOX21_TASKS)
        plot_confusion_matrices(y_true, y_prob, best_scaffold_model, TOX21_TASKS)
    else:
        print("  No scaffold predictions available for ROC/confusion plots.")

    # Plot 5: Training curves (GATv2)
    history_rand = gatv2_histories.get("random", [])
    history_scaf = gatv2_histories.get("scaffold", [])
    if history_rand or history_scaf:
        plot_training_curves(history_rand, history_scaf)

    # ── 8. Final summary ──────────────────────────────────────────────────────
    section("SUMMARY")
    elapsed = time.time() - t_start
    print(f"  Total runtime: {elapsed/60:.1f} min")
    print()

    # Print mean AUROCs from comparison table
    mean_row = auroc_df[auroc_df["Endpoint"] == "Mean"]
    if not mean_row.empty:
        for col in mean_row.columns:
            if col != "Endpoint" and "_AUROC" in col:
                val = mean_row[col].values[0]
                label = col.replace("_AUROC", "").replace("_", " ")
                print(f"  Mean AUROC  {label:<22}: {val:.4f}")
    print()

    results_dir = pathlib.Path(__file__).parent / "results"
    print(f"  Results saved to: {results_dir}/")
    print(f"  Models saved to:  {pathlib.Path(__file__).parent / 'models'}/")
    print()
    print("  Generated files:")
    for f in sorted(results_dir.glob("*.csv")):
        print(f"    {f.name}")
    for f in sorted(results_dir.glob("*.png")):
        print(f"    {f.name}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Tox21 Phase 2 — GNN Toxicity Prediction Pipeline"
    )
    p.add_argument(
        "--model", choices=["gatv2", "dmpnn", "both"], default="both",
        help="Which model to train (default: both)"
    )
    p.add_argument(
        "--split", choices=["random", "scaffold", "both"], default="both",
        help="Which splitting strategy to use (default: both)"
    )
    p.add_argument(
        "--epochs", type=int, default=100,
        help="Maximum training epochs per model (default: 100)"
    )
    p.add_argument(
        "--patience", type=int, default=15,
        help="Early stopping patience in epochs (default: 15)"
    )
    p.add_argument(
        "--skip-train", action="store_true",
        help="Skip training; only rebuild tables and plots from existing CSVs"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed (default: 42)"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seeds(args.seed)
    main(args)
