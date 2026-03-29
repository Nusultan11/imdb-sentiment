from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import joblib
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.pipeline import Pipeline


def load_model(model_path: str | Path) -> Pipeline:
    resolved_path = Path(model_path)
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", InconsistentVersionWarning)
        model = joblib.load(resolved_path)

    for warning in caught_warnings:
        if isinstance(warning.message, InconsistentVersionWarning):
            mismatch = warning.message
            raise RuntimeError(
                "Failed to load model because scikit-learn versions do not match. "
                f"Artifact was created with scikit-learn {mismatch.original_sklearn_version}, "
                f"but current environment uses {mismatch.current_sklearn_version}. "
                "Install the matching scikit-learn version before loading this model."
            )

    return model


def predict_texts(model: Pipeline, texts: Iterable[str]) -> list[int]:
    predictions = model.predict(list(texts))
    return [int(pred) for pred in predictions]


def predict_from_model_path(model_path: str | Path, texts: Iterable[str]) -> list[int]:
    model = load_model(model_path)
    return predict_texts(model, texts)
