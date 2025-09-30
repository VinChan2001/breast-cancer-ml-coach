"""Utilities for training and evaluating a classification model on tabular data."""

from .config import TrainingConfig
from .pipeline import run_pipeline

__all__ = ["TrainingConfig", "run_pipeline"]
