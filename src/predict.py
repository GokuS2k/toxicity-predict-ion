"""
Inference interface: load a trained GNN and predict toxicity from SMILES.

Usage (CLI demo):
    python src/predict.py

Usage (API):
    from predict import load_model, predict_toxicity, batch_predict
    model, device = load_model()
    result = predict_toxicity("CCO", model, device)
"""

import pathlib
from typing import Optional

import torch
from torch_geometric.loader import DataLoader

from featurization import smiles_to_graph
from model import MolecularGNN
from dataset import TOX21_TASKS

MODELS_DIR = pathlib.Path(__file__).parent.parent / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "tox21_gnn_model.pt"


def load_model(
    model_path: pathlib.Path = DEFAULT_MODEL_PATH,
    device: Optional[torch.device] = None,
) -> tuple[MolecularGNN, torch.device]:
    """Load a trained GNN from a checkpoint file."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(model_path, map_location=device)
    args = ckpt.get("args", {})

    model = MolecularGNN(
        hidden_dim=args.get("hidden_dim", 128),
        num_heads=args.get("num_heads", 4),
        num_layers=args.get("num_layers", 3),
        dropout=args.get("dropout", 0.2),
        num_tasks=len(TOX21_TASKS),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from {model_path}  (epoch {ckpt.get('epoch','?')}, "
          f"val_auroc={ckpt.get('val_auroc', 0.0):.4f})")
    return model, device


def predict_toxicity(
    smiles: str,
    model: MolecularGNN,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """
    Predict toxicity for a single SMILES string.

    Returns a dict with:
      - 'smiles': input SMILES
      - 'valid': bool (False if SMILES could not be parsed)
      - per-task entries: {'prediction': int, 'probability': float, 'label': str}
    """
    graph = smiles_to_graph(smiles)
    if graph is None:
        return {"smiles": smiles, "valid": False}

    graph = graph.to(device)

    # Add batch dimension (single graph)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(graph)
        probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()

    result = {"smiles": smiles, "valid": True}
    for task, prob in zip(TOX21_TASKS, probs):
        pred = int(prob >= threshold)
        result[task] = {
            "prediction": pred,
            "probability": round(prob, 4),
            "label": "TOXIC" if pred == 1 else "non-toxic",
        }
    return result


def batch_predict(
    smiles_list: list[str],
    model: MolecularGNN,
    device: torch.device,
    batch_size: int = 64,
    threshold: float = 0.5,
) -> list[dict]:
    """Predict toxicity for a list of SMILES strings."""
    # Build graphs
    graphs, valid_idx, invalid_smiles = [], [], []
    for i, smi in enumerate(smiles_list):
        g = smiles_to_graph(smi)
        if g is not None:
            g.smiles = smi
            graphs.append(g)
            valid_idx.append(i)
        else:
            invalid_smiles.append((i, smi))

    results = [None] * len(smiles_list)
    for idx, smi in invalid_smiles:
        results[idx] = {"smiles": smi, "valid": False}

    if not graphs:
        return results

    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().tolist()
            all_probs.extend(probs)

    for i, (orig_idx, probs) in enumerate(zip(valid_idx, all_probs)):
        smi = smiles_list[orig_idx]
        entry = {"smiles": smi, "valid": True}
        for task, prob in zip(TOX21_TASKS, probs):
            pred = int(prob >= threshold)
            entry[task] = {
                "prediction": pred,
                "probability": round(prob, 4),
                "label": "TOXIC" if pred == 1 else "non-toxic",
            }
        results[orig_idx] = entry

    return results


def print_prediction(result: dict):
    """Pretty-print a single prediction result."""
    smi = result["smiles"]
    print(f"\nSMILES: {smi}")
    if not result.get("valid", True):
        print("  ✗  Invalid SMILES — could not parse molecule")
        return
    print(f"  {'Endpoint':<18} {'Prob':>6}  {'Label'}")
    print(f"  {'-'*40}")
    for task in TOX21_TASKS:
        t = result[task]
        marker = "⚠" if t["prediction"] == 1 else " "
        print(f"  {marker} {task:<16} {t['probability']:>6.4f}  {t['label']}")


# ── Demo ─────────────────────────────────────────────────────────────────────

DEMO_SMILES = [
    ("Ethanol",       "CCO"),
    ("Aspirin",       "CC(=O)Oc1ccccc1C(=O)O"),
    ("Bisphenol A",   "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1"),
    ("Dioxin (TCDD)", "Clc1cc2oc3cc(Cl)c(Cl)cc3oc2cc1Cl"),
    ("Tamoxifen",     "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1"),
]


if __name__ == "__main__":
    if not DEFAULT_MODEL_PATH.exists():
        print(f"No trained model found at {DEFAULT_MODEL_PATH}")
        print("Run:  python src/train.py")
        raise SystemExit(1)

    model, device = load_model()

    print("\n" + "="*55)
    print("  Molecular Toxicity Predictions (GNN)")
    print("="*55)

    for name, smi in DEMO_SMILES:
        print(f"\n{'─'*55}")
        print(f"  Compound: {name}")
        result = predict_toxicity(smi, model, device)
        print_prediction(result)

    print(f"\n{'='*55}")
