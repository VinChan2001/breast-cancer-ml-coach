from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from ml_project.config import load_config
from ml_project.data import load_dataset
from ml_project.metrics import compute_metrics
from ml_project.model import build_estimator


IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def _select_hyperparameters(config):
    if config.model_type == "logistic_regression":
        return config.logistic_regression
    if config.model_type == "random_forest":
        return config.random_forest
    raise ValueError(f"Unsupported model type '{config.model_type}'.")


def main() -> None:
    config = load_config("configs/default.yaml")
    dataset = load_dataset(config.dataset, config.test_size, config.random_state)
    estimator = build_estimator(config.model_type, _select_hyperparameters(config))

    estimator.fit(dataset.X_train, dataset.y_train)
    predictions = estimator.predict(dataset.X_test)

    metrics = compute_metrics(dataset.y_test, predictions)

    # Confusion matrix plot
    cm = confusion_matrix(dataset.y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Breast Cancer")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    # Metrics bar chart
    names = list(metrics.keys())
    values = [metrics[name] for name in names]
    plt.figure(figsize=(6, 4))
    plt.bar(names, values, color="#4c72b0")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Evaluation Metrics - Breast Cancer Model")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "metric_scores.png", dpi=200)
    plt.close()

    with (IMAGES_DIR / "metrics.json").open("w", encoding="utf-8") as fh:
        import json

        json.dump(metrics, fh, indent=2)


if __name__ == "__main__":
    main()
