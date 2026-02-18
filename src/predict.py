"""
predict.py
----------
Prediction interface for the Tox21 Random Forest toxicity model.

Usage:
    from src.predict import predict_toxicity, load_model

    result = predict_toxicity("CCO")  # ethanol
    for endpoint, info in result.items():
        print(f"{endpoint}: {'TOXIC' if info['prediction'] == 1 else 'non-toxic'} "
              f"(p={info['probability']:.3f})")
"""

import os
import logging
import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "tox21_rf_model.joblib")


def _smiles_to_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray | None:
    """Convert a SMILES string to a Morgan fingerprint array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    from rdkit.Chem import AllChem
    from rdkit.DataStructs import ConvertToNumpyArray

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.uint8)
    ConvertToNumpyArray(fp, arr)
    return arr


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """
    Load the trained Tox21RandomForest model from disk.

    Args:
        model_path : Path to a .joblib model file.

    Returns:
        Loaded Tox21RandomForest instance.
    """
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from model import Tox21RandomForest

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            f"Run 'python src/train.py' first to train and save the model."
        )
    return Tox21RandomForest.load(model_path)


def predict_toxicity(
    smiles_string: str,
    model=None,
    model_path: str = DEFAULT_MODEL_PATH,
    threshold: float = 0.5,
) -> dict:
    """
    Predict toxicity for a single molecule given as a SMILES string.

    Args:
        smiles_string : SMILES representation of the molecule.
        model         : Pre-loaded Tox21RandomForest (optional, loaded from disk if None).
        model_path    : Path to the model file (used if model is None).
        threshold     : Probability threshold for binary classification.

    Returns:
        dict mapping endpoint name -> {
            "prediction"  : int  (1 = toxic, 0 = non-toxic, -1 = no model),
            "probability" : float (P(toxic), NaN if no model),
            "label"       : str  ("TOXIC", "non-toxic", or "unknown"),
        }

    Raises:
        ValueError : If the SMILES string is invalid.
    """
    # Validate SMILES
    fp = _smiles_to_fingerprint(smiles_string)
    if fp is None:
        raise ValueError(f"Invalid SMILES string: {smiles_string!r}")

    # Load model if not provided
    if model is None:
        model = load_model(model_path)

    # Predict — reshape to (1, n_bits) for single sample
    X = fp.reshape(1, -1)
    proba = model.predict_proba(X)  # shape (1, n_tasks)
    preds = model.predict(X, threshold=threshold)  # shape (1, n_tasks)

    results = {}
    for i, task in enumerate(model.task_names):
        prob = proba[0, i]
        pred = preds[0, i]

        if np.isnan(prob):
            label = "unknown"
        elif pred == 1:
            label = "TOXIC"
        else:
            label = "non-toxic"

        results[task] = {
            "prediction": int(pred),
            "probability": float(prob) if not np.isnan(prob) else None,
            "label": label,
        }

    return results


def batch_predict(
    smiles_list: list[str],
    model=None,
    model_path: str = DEFAULT_MODEL_PATH,
    threshold: float = 0.5,
) -> list[dict]:
    """
    Predict toxicity for multiple SMILES strings in a batch.

    Skips invalid SMILES and logs warnings.

    Args:
        smiles_list : List of SMILES strings.
        model       : Pre-loaded Tox21RandomForest (loaded from disk if None).
        model_path  : Path to the model file.
        threshold   : Probability threshold.

    Returns:
        List of dicts, one per valid input SMILES. Each dict has keys:
        "smiles", "valid", and (if valid) all endpoint predictions.
    """
    if model is None:
        model = load_model(model_path)

    results = []
    valid_fps = []
    valid_smiles = []
    invalid_indices = []

    for i, smi in enumerate(smiles_list):
        fp = _smiles_to_fingerprint(smi)
        if fp is not None:
            valid_fps.append(fp)
            valid_smiles.append(smi)
        else:
            logger.warning(f"Invalid SMILES at index {i}: {smi!r}")
            invalid_indices.append(i)

    if not valid_fps:
        return [{"smiles": s, "valid": False} for s in smiles_list]

    X = np.vstack(valid_fps)
    proba = model.predict_proba(X)
    preds = model.predict(X, threshold=threshold)

    valid_results = []
    for j, smi in enumerate(valid_smiles):
        entry = {"smiles": smi, "valid": True}
        for k, task in enumerate(model.task_names):
            prob = proba[j, k]
            pred = preds[j, k]
            entry[task] = {
                "prediction": int(pred),
                "probability": float(prob) if not np.isnan(prob) else None,
                "label": "TOXIC" if pred == 1 else ("non-toxic" if pred != -1 else "unknown"),
            }
        valid_results.append(entry)

    # Reconstruct ordered list (valid + invalid markers)
    valid_iter = iter(valid_results)
    output = []
    invalid_set = set(invalid_indices)
    vi = 0
    for i in range(len(smiles_list)):
        if i in invalid_set:
            output.append({"smiles": smiles_list[i], "valid": False})
        else:
            output.append(next(valid_iter))

    return output


def print_prediction(smiles: str, result: dict) -> None:
    """Pretty-print a prediction result for a single molecule."""
    print(f"\nMolecule: {smiles}")
    print("-" * 50)
    print(f"{'Endpoint':<18} {'Label':<12} {'P(toxic)':>10}")
    print("-" * 50)
    for endpoint, info in result.items():
        prob_str = f"{info['probability']:.4f}" if info["probability"] is not None else "  N/A  "
        print(f"  {endpoint:<16} {info['label']:<12} {prob_str:>10}")
    print("-" * 50)


if __name__ == "__main__":
    # Demo: test on a few known molecules
    test_molecules = [
        ("CCO", "Ethanol"),
        ("c1ccccc1", "Benzene"),
        ("CC(=O)Oc1ccccc1C(=O)O", "Aspirin"),
        ("ClC(Cl)(Cl)Cl", "Carbon tetrachloride"),
        ("CCOC(=O)c1ccc(N)cc1", "Benzocaine"),
    ]

    print("Loading model...")
    model = load_model()

    for smiles, name in test_molecules:
        print(f"\n{'='*50}")
        print(f"Compound: {name}")
        result = predict_toxicity(smiles, model=model)
        print_prediction(smiles, result)
