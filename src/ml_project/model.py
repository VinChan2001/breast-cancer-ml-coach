from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_estimator(model_type: str, hyperparameters: Dict[str, Any]) -> Pipeline:
    """Create the estimator pipeline for the requested model type."""

    if model_type == "logistic_regression":
        classifier = LogisticRegression(**hyperparameters)
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", classifier),
        ])

    if model_type == "random_forest":
        classifier = RandomForestClassifier(**hyperparameters)
        return Pipeline([
            ("clf", classifier),
        ])

    raise ValueError(
        f"Unsupported model type '{model_type}'. Expected 'logistic_regression' or 'random_forest'."
    )
