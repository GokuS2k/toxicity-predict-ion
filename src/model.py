"""
model.py
--------
Random Forest multi-task toxicity classifier for the Tox21 dataset.

Design choices:
  - 12 separate Random Forest classifiers (one per endpoint): this outperforms
    MultiOutputClassifier because each endpoint has different missing-label
    patterns, class imbalances, and feature importances.
  - class_weight='balanced': automatically compensates for the 5–20% positive
    rate seen across Tox21 endpoints without requiring manual upsampling.
  - Hyperparameters: n_estimators=100, max_depth=20, random_state=42.
    These are reasonable defaults; see README for tuning guidance.
  - Training mask: samples with NaN labels for a given endpoint are excluded
    from that endpoint's training — not imputed to 0 or 1.
"""

import os
import logging
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

DEFAULT_RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 20,
    "class_weight": "balanced",   # handles class imbalance
    "random_state": 42,
    "n_jobs": -1,                  # use all available CPU cores
    "min_samples_leaf": 2,         # slight regularisation to reduce overfitting
}


class Tox21RandomForest:
    """
    Multi-task Random Forest for Tox21 toxicity prediction.

    One RandomForestClassifier is trained per endpoint. During training,
    samples missing the label for that endpoint are masked out.

    Attributes:
        task_names (list[str]) : Names of the 12 Tox21 endpoints.
        models     (list)      : Fitted RandomForestClassifier objects.
        rf_params  (dict)      : Hyperparameters passed to each RF.
    """

    def __init__(self, task_names: list[str], rf_params: dict | None = None):
        self.task_names = task_names
        self.rf_params = rf_params if rf_params is not None else DEFAULT_RF_PARAMS
        self.models: list[RandomForestClassifier | None] = [None] * len(task_names)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "Tox21RandomForest":
        """
        Train one RF per task, using only samples with known labels.

        Args:
            X_train : Feature matrix, shape (n_samples, n_features).
            y_train : Label matrix, shape (n_samples, n_tasks), NaN = missing.

        Returns:
            self
        """
        n_tasks = len(self.task_names)
        assert y_train.shape[1] == n_tasks, \
            f"Expected {n_tasks} label columns, got {y_train.shape[1]}"

        logger.info(f"Training {n_tasks} Random Forest classifiers...")

        for i, task in enumerate(self.task_names):
            # Mask: only keep samples with known label for this task
            known_mask = ~np.isnan(y_train[:, i])
            X_task = X_train[known_mask]
            y_task = y_train[known_mask, i].astype(int)

            n_pos = (y_task == 1).sum()
            n_neg = (y_task == 0).sum()
            logger.info(
                f"  [{i+1:2d}/{n_tasks}] {task:<18} | "
                f"n={len(y_task):5d} | pos={n_pos:4d} ({n_pos/len(y_task)*100:5.1f}%) | "
                f"neg={n_neg:4d}"
            )

            # Skip if only one class present (can't train a classifier)
            if len(np.unique(y_task)) < 2:
                logger.warning(f"    -> Skipping {task}: only one class in training data.")
                self.models[i] = None
                continue

            rf = RandomForestClassifier(**self.rf_params)
            rf.fit(X_task, y_task)
            self.models[i] = rf

        trained = sum(m is not None for m in self.models)
        logger.info(f"Successfully trained {trained}/{n_tasks} models.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict positive-class probabilities for all tasks.

        Returns:
            proba : np.ndarray, shape (n_samples, n_tasks)
                    NaN for tasks where no model was trained.
        """
        n_samples = X.shape[0]
        proba = np.full((n_samples, len(self.task_names)), np.nan)

        for i, (task, model) in enumerate(zip(self.task_names, self.models)):
            if model is None:
                logger.warning(f"No model for task '{task}', returning NaN.")
                continue
            # RF predict_proba returns (n_samples, 2); column 1 = P(positive)
            proba[:, i] = model.predict_proba(X)[:, 1]

        return proba

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary labels for all tasks using a probability threshold.

        Returns:
            predictions : np.ndarray of int, shape (n_samples, n_tasks)
                          -1 for tasks without a trained model.
        """
        proba = self.predict_proba(X)
        preds = np.where(np.isnan(proba), -1, (proba >= threshold).astype(int))
        return preds

    def save(self, path: str | None = None) -> str:
        """
        Persist the model ensemble to disk using joblib.

        Args:
            path : File path (default: models/tox21_rf_model.joblib)

        Returns:
            Absolute path where the model was saved.
        """
        if path is None:
            os.makedirs(MODEL_DIR, exist_ok=True)
            path = os.path.join(MODEL_DIR, "tox21_rf_model.joblib")

        payload = {
            "task_names": self.task_names,
            "models": self.models,
            "rf_params": self.rf_params,
            "fingerprint_radius": 2,
            "fingerprint_nbits": 2048,
        }
        joblib.dump(payload, path, compress=3)
        logger.info(f"Model saved to: {path}")
        return os.path.abspath(path)

    @classmethod
    def load(cls, path: str) -> "Tox21RandomForest":
        """
        Load a saved Tox21RandomForest from disk.

        Args:
            path : Path to a .joblib file previously saved with `save()`.

        Returns:
            A ready-to-use Tox21RandomForest instance.
        """
        payload = joblib.load(path)
        instance = cls(
            task_names=payload["task_names"],
            rf_params=payload["rf_params"]
        )
        instance.models = payload["models"]
        logger.info(f"Model loaded from: {path}")
        return instance

    def feature_importances(self) -> dict[str, np.ndarray]:
        """
        Return feature importances from each trained RF.

        Returns:
            dict mapping task name -> importance array (length n_features),
            or empty array if model was not trained.
        """
        importances = {}
        for task, model in zip(self.task_names, self.models):
            if model is not None:
                importances[task] = model.feature_importances_
            else:
                importances[task] = np.array([])
        return importances
