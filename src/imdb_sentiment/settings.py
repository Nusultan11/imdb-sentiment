from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(slots=True)
class PathsConfig:
    model_output: Path
    metrics_output: Path


@dataclass(slots=True)
class ModelConfig:
    type: str
    max_features: int
    ngram_range: tuple[int, int]
    max_iter: int


@dataclass(slots=True)
class AppConfig:
    seed: int
    paths: PathsConfig
    model: ModelConfig


def _require_dict(payload: object, name: str) -> dict:
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a mapping")
    return payload


def _require_int(payload: dict, key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _require_str(payload: dict, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _require_ngram(payload: dict, key: str) -> tuple[int, int]:
    value = payload.get(key)
    if not isinstance(value, list) or len(value) != 2 or not all(isinstance(v, int) for v in value):
        raise ValueError(f"{key} must be a list of two integers")
    return value[0], value[1]


def load_config(path: str | Path = "configs/baseline.yaml") -> AppConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload = _require_dict(payload, "config")
    paths_payload = _require_dict(payload.get("paths"), "paths")
    model_payload = _require_dict(payload.get("model"), "model")

    return AppConfig(
        seed=_require_int(payload, "seed"),
        paths=PathsConfig(
            model_output=Path(_require_str(paths_payload, "model_output")),
            metrics_output=Path(_require_str(paths_payload, "metrics_output")),
        ),
        model=ModelConfig(
            type=_require_str(model_payload, "type"),
            max_features=_require_int(model_payload, "max_features"),
            ngram_range=_require_ngram(model_payload, "ngram_range"),
            max_iter=_require_int(model_payload, "max_iter"),
        ),
    )
