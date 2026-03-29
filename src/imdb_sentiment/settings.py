from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_int(value: Any, name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_ngram_range(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2 or not all(isinstance(v, int) for v in value):
        raise ValueError(f"{name} must be a list of two integers")
    return value[0], value[1]


def load_config(path: str | Path = "configs/baseline.yaml") -> AppConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload = _require_dict(payload, "config")

    paths_payload = _require_dict(payload.get("paths"), "paths")
    model_payload = _require_dict(payload.get("model"), "model")

    return AppConfig(
        seed=_require_int(payload.get("seed"), "seed"),
        paths=PathsConfig(
            model_output=Path(_require_str(paths_payload.get("model_output"), "paths.model_output")),
            metrics_output=Path(_require_str(paths_payload.get("metrics_output"), "paths.metrics_output")),
        ),
        model=ModelConfig(
            type=_require_str(model_payload.get("type"), "model.type"),
            max_features=_require_int(model_payload.get("max_features"), "model.max_features"),
            ngram_range=_require_ngram_range(model_payload.get("ngram_range"), "model.ngram_range"),
            max_iter=_require_int(model_payload.get("max_iter"), "model.max_iter"),
        ),
    )
