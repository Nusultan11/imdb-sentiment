from __future__ import annotations

from dataclasses import dataclass
import warnings
from pathlib import Path
from typing import Iterable

import joblib
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.pipeline import Pipeline

from imdb_sentiment.artifacts.lstm import load_lstm_artifact_sidecars
from imdb_sentiment.data.lstm import LSTMVocabulary, encode_lstm_text
from imdb_sentiment.models.lstm.model import SentimentLSTM, build_lstm_model
from imdb_sentiment.settings import AppConfig, LSTMModelConfig

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised only when torch is absent
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


@dataclass(slots=True)
class LSTMInferenceArtifacts:
    model: SentimentLSTM
    vocabulary: LSTMVocabulary
    max_length: int


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for LSTM inference. Install torch before calling "
            "load_lstm_checkpoint() or predict_lstm_texts()."
        ) from TORCH_IMPORT_ERROR


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


def load_lstm_checkpoint(
    model_path: str | Path,
    config: LSTMModelConfig,
) -> LSTMInferenceArtifacts:
    _require_torch()
    checkpoint = torch.load(Path(model_path))
    _artifact_contract, vocabulary_payload, training_config_payload = load_lstm_artifact_sidecars(
        model_path
    )
    vocabulary = LSTMVocabulary(token_to_id=vocabulary_payload)
    serialized_model = training_config_payload.get("model")
    if not isinstance(serialized_model, dict):
        raise ValueError("training_config.json must contain a model mapping.")
    model = build_lstm_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return LSTMInferenceArtifacts(
        model=model,
        vocabulary=vocabulary,
        max_length=int(serialized_model["max_length"]),
    )


def predict_lstm_texts(
    artifacts: LSTMInferenceArtifacts,
    texts: Iterable[str],
) -> list[int]:
    _require_torch()
    texts_list = list(texts)
    if not texts_list:
        return []

    token_id_rows = [
        encode_lstm_text(
            text=text,
            vocabulary=artifacts.vocabulary,
            max_length=artifacts.max_length,
        )
        for text in texts_list
    ]
    token_ids = torch.tensor(token_id_rows, dtype=torch.long)
    with torch.no_grad():
        logits = artifacts.model(token_ids)
        predictions = torch.sigmoid(logits).ge(0.5).to(dtype=torch.int64)
    return [int(prediction) for prediction in predictions.tolist()]


def predict_from_model_path(
    model_path: str | Path,
    texts: Iterable[str],
    config: AppConfig | None = None,
) -> list[int]:
    if config is not None and config.experiment.family == "lstm":
        if not isinstance(config.model, LSTMModelConfig):
            raise TypeError("LSTM inference expects LSTMModelConfig.")
        artifacts = load_lstm_checkpoint(model_path, config.model)
        return predict_lstm_texts(artifacts, texts)

    model = load_model(model_path)
    return predict_texts(model, texts)
