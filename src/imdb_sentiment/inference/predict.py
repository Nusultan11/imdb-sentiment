from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
from sklearn.pipeline import Pipeline


def load_model(model_path: str | Path) -> Pipeline:
    resolved_path = Path(model_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Model file not found: {resolved_path}")
    return joblib.load(resolved_path)


def predict_texts(model: Pipeline, texts: Iterable[str]) -> list[int]:
    batch = list(texts)
    if not batch:
        raise ValueError("texts must not be empty")
    predictions = model.predict(batch)
    return [int(pred) for pred in predictions]
