from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """Return a dictionary of simple scalar metrics."""

    average = "binary" if len(np.unique(y_true)) == 2 else "weighted"
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }


def textual_report(y_true, y_pred) -> str:
    """Generate a human friendly classification report."""

    return classification_report(y_true, y_pred)
