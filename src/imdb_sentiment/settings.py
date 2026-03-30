from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class PathsConfig:
    model_output: Path
    val_metrics_output: Path
    test_metrics_output: Path


@dataclass(slots=True)
class ExperimentConfig:
    family: str
    name: str


@dataclass(slots=True)
class TfidfModelConfig:
    type: str
    max_features: int
    ngram_range: tuple[int, int]
    max_iter: int


@dataclass(slots=True)
class LSTMModelConfig:
    type: str
    vocab_size: int
    max_length: int
    embedding_dim: int
    hidden_dim: int
    batch_size: int
    epochs: int
    dropout: float
    lr: float


@dataclass(slots=True)
class TransformerModelConfig:
    type: str
    pretrained_model_name: str
    max_length: int
    batch_size: int
    epochs: int
    dropout: float
    lr: float
    weight_decay: float


ModelConfig = TfidfModelConfig | LSTMModelConfig | TransformerModelConfig


@dataclass(slots=True)
class AppConfig:
    experiment: ExperimentConfig
    seed: int
    paths: PathsConfig
    model: ModelConfig


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_int(value: Any, name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    integer_value = _require_int(value, name)
    if integer_value < 1:
        raise ValueError(f"{name} must be at least 1")
    return integer_value


def _require_float(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a float")
    return float(value)


def _require_positive_float(value: Any, name: str) -> float:
    float_value = _require_float(value, name)
    if float_value <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return float_value


def _require_probability(value: Any, name: str) -> float:
    probability = _require_float(value, name)
    if probability < 0 or probability >= 1:
        raise ValueError(f"{name} must be in the range [0, 1)")
    return probability


def _require_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_ngram_range(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2 or not all(isinstance(v, int) for v in value):
        raise ValueError(f"{name} must be a list of two integers")
    min_n, max_n = value
    if min_n < 1:
        raise ValueError(f"{name} minimum must be at least 1")
    if min_n > max_n:
        raise ValueError(f"{name} minimum must be less than or equal to maximum")
    return min_n, max_n


def _resolve_input_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()

    project_candidate = PROJECT_ROOT / candidate
    if project_candidate.exists():
        return project_candidate

    return candidate


def _resolve_project_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate

    return (PROJECT_ROOT / candidate).resolve()


def _load_tfidf_model_config(model_payload: dict[str, Any]) -> TfidfModelConfig:
    return TfidfModelConfig(
        type=_require_str(model_payload.get("type"), "model.type"),
        max_features=_require_positive_int(model_payload.get("max_features"), "model.max_features"),
        ngram_range=_require_ngram_range(model_payload.get("ngram_range"), "model.ngram_range"),
        max_iter=_require_positive_int(model_payload.get("max_iter"), "model.max_iter"),
    )


def _load_lstm_model_config(model_payload: dict[str, Any]) -> LSTMModelConfig:
    return LSTMModelConfig(
        type=_require_str(model_payload.get("type"), "model.type"),
        vocab_size=_require_positive_int(model_payload.get("vocab_size"), "model.vocab_size"),
        max_length=_require_positive_int(model_payload.get("max_length"), "model.max_length"),
        embedding_dim=_require_positive_int(model_payload.get("embedding_dim"), "model.embedding_dim"),
        hidden_dim=_require_positive_int(model_payload.get("hidden_dim"), "model.hidden_dim"),
        batch_size=_require_positive_int(model_payload.get("batch_size"), "model.batch_size"),
        epochs=_require_positive_int(model_payload.get("epochs"), "model.epochs"),
        dropout=_require_probability(model_payload.get("dropout"), "model.dropout"),
        lr=_require_positive_float(model_payload.get("lr"), "model.lr"),
    )


def _load_transformer_model_config(model_payload: dict[str, Any]) -> TransformerModelConfig:
    return TransformerModelConfig(
        type=_require_str(model_payload.get("type"), "model.type"),
        pretrained_model_name=_require_str(
            model_payload.get("pretrained_model_name"),
            "model.pretrained_model_name",
        ),
        max_length=_require_positive_int(model_payload.get("max_length"), "model.max_length"),
        batch_size=_require_positive_int(model_payload.get("batch_size"), "model.batch_size"),
        epochs=_require_positive_int(model_payload.get("epochs"), "model.epochs"),
        dropout=_require_probability(model_payload.get("dropout"), "model.dropout"),
        lr=_require_positive_float(model_payload.get("lr"), "model.lr"),
        weight_decay=_require_positive_float(model_payload.get("weight_decay"), "model.weight_decay"),
    )


def _load_model_config(family: str, model_payload: dict[str, Any]) -> ModelConfig:
    model_type = _require_str(model_payload.get("type"), "model.type")

    if family == "tfidf":
        if model_type != "logistic_regression":
            raise ValueError("TF-IDF experiments must use model.type=logistic_regression")
        return _load_tfidf_model_config(model_payload)

    if family == "lstm":
        if model_type != "lstm":
            raise ValueError("LSTM experiments must use model.type=lstm")
        return _load_lstm_model_config(model_payload)

    if family == "transformer":
        if model_type != "distilbert":
            raise ValueError("Transformer experiments must use model.type=distilbert")
        return _load_transformer_model_config(model_payload)

    raise ValueError(f"Unsupported experiment.family: {family}")


def load_config(path: str | Path = "configs/baseline.yaml") -> AppConfig:
    config_path = _resolve_input_path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload = _require_dict(payload, "config")

    experiment_payload = _require_dict(payload.get("experiment"), "experiment")
    paths_payload = _require_dict(payload.get("paths"), "paths")
    model_payload = _require_dict(payload.get("model"), "model")

    experiment = ExperimentConfig(
        family=_require_str(experiment_payload.get("family"), "experiment.family"),
        name=_require_str(experiment_payload.get("name"), "experiment.name"),
    )

    return AppConfig(
        experiment=experiment,
        seed=_require_int(payload.get("seed"), "seed"),
        paths=PathsConfig(
            model_output=_resolve_project_path(
                _require_str(paths_payload.get("model_output"), "paths.model_output")
            ),
            val_metrics_output=_resolve_project_path(
                _require_str(paths_payload.get("val_metrics_output"), "paths.val_metrics_output")
            ),
            test_metrics_output=_resolve_project_path(
                _require_str(paths_payload.get("test_metrics_output"), "paths.test_metrics_output")
            ),
        ),
        model=_load_model_config(experiment.family, model_payload),
    )
