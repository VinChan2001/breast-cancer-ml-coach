from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib


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
