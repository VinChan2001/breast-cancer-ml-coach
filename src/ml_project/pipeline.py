from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from .config import TrainingConfig
from .data import load_dataset
from .metrics import compute_metrics, textual_report
from .model import build_estimator
from .persistence import save_dataframe, save_metrics, save_model, save_text


def _select_hyperparameters(config: TrainingConfig) -> Dict[str, Any]:
    if config.model_type == "logistic_regression":
        return config.logistic_regression
    if config.model_type == "random_forest":
        return config.random_forest
    raise ValueError(f"Unsupported model type '{config.model_type}'.")


def _run_cross_validation(config: TrainingConfig) -> Tuple[Dict[str, float], Any]:
    dataset = load_dataset(config.dataset, config.test_size, config.random_state)
    estimator = build_estimator(config.model_type, _select_hyperparameters(config))

    cv = StratifiedKFold(
        n_splits=config.evaluation.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )

    scoring = config.evaluation.scoring_list()
    cv_results = cross_validate(
        estimator,
        dataset.X_train,
        dataset.y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=config.evaluation.n_jobs,
        return_train_score=True,
    )

    summary: Dict[str, float] = {}
    for key, values in cv_results.items():
        array = np.asarray(values)
        summary[f"{key}_mean"] = float(np.mean(array))
        summary[f"{key}_std"] = float(np.std(array, ddof=1))

    return summary, estimator, dataset


def _compile_predictions(estimator, dataset) -> pd.DataFrame:
    X_test = dataset.X_test.copy()
    predictions = estimator.predict(dataset.X_test)

    results = pd.DataFrame({
        "actual": dataset.y_test,
        "predicted": predictions,
    })

    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(dataset.X_test)
        for idx, class_label in enumerate(estimator.classes_):
            results[f"prob_{class_label}"] = probabilities[:, idx]

    return pd.concat([X_test.reset_index(drop=True), results.reset_index(drop=True)], axis=1)


def _feature_importances(estimator, feature_names: list[str]) -> pd.DataFrame | None:
    if hasattr(estimator, "named_steps") and "clf" in estimator.named_steps:
        model = estimator.named_steps["clf"]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            return (
                pd.DataFrame({
                    "feature": feature_names,
                    "importance": importances,
                })
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
    return None


def run_pipeline(config: TrainingConfig) -> Dict[str, Any]:
    cv_summary, estimator, dataset = _run_cross_validation(config)

    estimator.fit(dataset.X_train, dataset.y_train)
    predictions = estimator.predict(dataset.X_test)

    metrics = compute_metrics(dataset.y_test, predictions)
    report = textual_report(dataset.y_test, predictions)

    prediction_frame = _compile_predictions(estimator, dataset)
    feature_ranking = _feature_importances(estimator, list(dataset.X_train.columns))

    output_dir = config.resolve_output_dir()
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    prefix = f"{config.model_type}_{config.dataset}_{timestamp}"

    model_path = save_model(estimator, output_dir / f"{prefix}.joblib")
    metrics_path = save_metrics(metrics, output_dir / f"{prefix}_metrics.json")
    cv_metrics_path = save_metrics(cv_summary, output_dir / f"{prefix}_cv_metrics.json")
    report_path = save_text(report, output_dir / f"{prefix}_report.txt")
    predictions_path = save_dataframe(prediction_frame, output_dir / f"{prefix}_predictions.csv")

    feature_path = None
    if feature_ranking is not None:
        feature_path = save_dataframe(feature_ranking, output_dir / f"{prefix}_feature_importances.csv")

    return {
        "metrics": metrics,
        "cross_validation": cv_summary,
        "report_path": report_path,
        "model_path": model_path,
        "metrics_path": metrics_path,
        "cv_metrics_path": cv_metrics_path,
        "predictions_path": predictions_path,
        "feature_importances_path": feature_path,
    }
