from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Tuple

import yaml


@dataclass
class EvaluationConfig:
    """Configuration for evaluation and validation routines."""

    cv_folds: int = 5
    scoring: Tuple[str, ...] = ("accuracy", "precision", "recall", "f1")
    n_jobs: int = -1

    def __post_init__(self) -> None:
        if isinstance(self.scoring, Iterable) and not isinstance(self.scoring, tuple):
            self.scoring = tuple(self.scoring)

    def scoring_list(self) -> list[str]:
        return list(self.scoring)


@dataclass
class TrainingConfig:
    """Dataclass capturing the knobs for the training pipeline."""

    dataset: Literal["breast_cancer", "wine", "iris"] = "breast_cancer"
    test_size: float = 0.2
    random_state: int = 42
    model_type: Literal["logistic_regression", "random_forest"] = "logistic_regression"
    logistic_regression: Dict[str, Any] = field(
        default_factory=lambda: {
            "C": 2.5,
            "max_iter": 500,
            "penalty": "l2",
        }
    )
    random_forest: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_split": 5,
        }
    )
    output_dir: Path = field(default_factory=lambda: Path("artifacts"))
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)
        if not isinstance(self.evaluation, EvaluationConfig):
            if isinstance(self.evaluation, dict):
                self.evaluation = EvaluationConfig(**self.evaluation)
            else:
                self.evaluation = EvaluationConfig()

    def resolve_output_dir(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir

    @classmethod
    def from_yaml(cls, path: Path | str) -> "TrainingConfig":
        with Path(path).open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if "evaluation" in data and isinstance(data["evaluation"], dict):
            data["evaluation"] = EvaluationConfig(**data["evaluation"])
        return cls(**data)


def load_config(path: str | None) -> TrainingConfig:
    if path is None:
        return TrainingConfig()
    return TrainingConfig.from_yaml(path)
