from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .config import TrainingConfig, load_config
from .pipeline import run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a tabular classification model using scikit-learn.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file overriding the defaults.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["breast_cancer", "wine", "iris"],
        help="Dataset override without editing config.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["logistic_regression", "random_forest"],
        help="Model type override without editing config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to store trained artefacts.",
    )
    return parser


def _merge_overrides(config: TrainingConfig, dataset: Optional[str], model_type: Optional[str], output_dir: Optional[str]) -> TrainingConfig:
    if dataset is not None:
        config.dataset = dataset
    if model_type is not None:
        config.model_type = model_type
    if output_dir is not None:
        config.output_dir = Path(output_dir)
    return config


def _format_cv_summary(cv_metrics: dict[str, float]) -> dict[str, float]:
    return {
        key: value
        for key, value in cv_metrics.items()
        if key.startswith("test_") and key.endswith("_mean")
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config = _merge_overrides(config, args.dataset, args.model_type, args.output_dir)

    results = run_pipeline(config)

    summary_payload = {
        "holdout_metrics": results["metrics"],
        "cross_validation": _format_cv_summary(results["cross_validation"]),
    }
    print(json.dumps(summary_payload, indent=2))

    print(f"Model saved to: {results['model_path']}")
    print(f"Holdout metrics saved to: {results['metrics_path']}")
    print(f"Cross-validation metrics saved to: {results['cv_metrics_path']}")
    print(f"Report saved to: {results['report_path']}")
    print(f"Predictions saved to: {results['predictions_path']}")
    if results["feature_importances_path"] is not None:
        print(f"Feature importances saved to: {results['feature_importances_path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
