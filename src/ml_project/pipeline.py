from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from .config import TrainingConfig
from .data import load_dataset
from .metrics import compute_metrics, textual_report
from .model import build_estimator
from .persistence import save_metrics, save_model


def _select_hyperparameters(config: TrainingConfig) -> Dict[str, Any]:
    if config.model_type == "logistic_regression":
        return config.logistic_regression
    if config.model_type == "random_forest":
        return config.random_forest
    raise ValueError(f"Unsupported model type '{config.model_type}'.")


def run_pipeline(config: TrainingConfig) -> Dict[str, Any]:
    dataset = load_dataset(config.dataset, config.test_size, config.random_state)
    estimator = build_estimator(config.model_type, _select_hyperparameters(config))

    estimator.fit(dataset.X_train, dataset.y_train)
    predictions = estimator.predict(dataset.X_test)

    metrics = compute_metrics(dataset.y_test, predictions)
    report = textual_report(dataset.y_test, predictions)

    output_dir = config.resolve_output_dir()
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    prefix = f"{config.model_type}_{config.dataset}_{timestamp}"

    model_path = save_model(estimator, output_dir / f"{prefix}.joblib")
    metrics_path = save_metrics(metrics, output_dir / f"{prefix}_metrics.json")
    report_path = output_dir / f"{prefix}_report.txt"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write(report)

    return {
        "metrics": metrics,
        "report_path": report_path,
        "model_path": model_path,
        "metrics_path": metrics_path,
    }
