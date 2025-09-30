from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def _breast_cancer() -> Tuple[pd.DataFrame, pd.Series]:
    dataset = load_breast_cancer(as_frame=True)
    return dataset.data, dataset.target


def _wine() -> Tuple[pd.DataFrame, pd.Series]:
    dataset = load_wine(as_frame=True)
    return dataset.data, dataset.target


def _iris() -> Tuple[pd.DataFrame, pd.Series]:
    dataset = load_iris(as_frame=True)
    return dataset.data, dataset.target


_LOADERS: Dict[str, Callable[[], Tuple[pd.DataFrame, pd.Series]]] = {
    "breast_cancer": _breast_cancer,
    "wine": _wine,
    "iris": _iris,
}


def load_dataset(name: str, test_size: float, random_state: int) -> DatasetSplit:
    if name not in _LOADERS:
        raise ValueError(f"Unsupported dataset '{name}'. Options: {', '.join(_LOADERS)}")
    features, targets = _LOADERS[name]()
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=test_size,
        random_state=random_state,
        stratify=targets,
    )
    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
