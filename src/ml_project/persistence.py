from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd


def save_model(model: Any, path: Path) -> Path:
    path = path.with_suffix(".joblib") if path.suffix == "" else path
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def save_metrics(metrics: Dict[str, float], path: Path) -> Path:
    path = path.with_suffix(".json") if path.suffix == "" else path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    return path


def save_text(content: str, path: Path) -> Path:
    path = path.with_suffix(".txt") if path.suffix == "" else path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(content)
    return path


def save_dataframe(frame: pd.DataFrame, path: Path) -> Path:
    path = path.with_suffix(".csv") if path.suffix == "" else path
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path
