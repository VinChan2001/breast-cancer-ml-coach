"""Utilities for training and evaluating a classification model on tabular data."""

from .config import EvaluationConfig, TrainingConfig
from .pipeline import run_pipeline

__all__ = ["TrainingConfig", "EvaluationConfig", "run_pipeline"]
