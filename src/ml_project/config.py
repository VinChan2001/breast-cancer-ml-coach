from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal

import yaml


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

    def __post_init__(self) -> None:
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)

    def resolve_output_dir(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir

    @classmethod
    def from_yaml(cls, path: Path | str) -> "TrainingConfig":
        with Path(path).open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls(**data)


def load_config(path: str | None) -> TrainingConfig:
    if path is None:
        return TrainingConfig()
    return TrainingConfig.from_yaml(path)
