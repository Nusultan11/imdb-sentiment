from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
from sklearn.pipeline import Pipeline


def load_model(model_path: str | Path) -> Pipeline:
    return joblib.load(Path(model_path))


def predict_texts(model: Pipeline, texts: Iterable[str]) -> list[int]:
    predictions = model.predict(list(texts))
    return [int(pred) for pred in predictions]