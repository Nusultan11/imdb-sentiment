from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(slots=True)
class PathsConfig:
    raw_data: Path
    model_output: Path
    metrics_output: Path


@dataclass(slots=True)
class DataConfig:
    text_column: str
    target_column: str
    test_size: float


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
    data: DataConfig
    model: ModelConfig


def load_config(path: str | Path = "configs/baseline.yaml") -> AppConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return AppConfig(
        seed=payload["seed"],
        paths=PathsConfig(
            raw_data=Path(payload["paths"]["raw_data"]),
            model_output=Path(payload["paths"]["model_output"]),
            metrics_output=Path(payload["paths"]["metrics_output"]),
        ),
        data=DataConfig(
            text_column=payload["data"]["text_column"],
            target_column=payload["data"]["target_column"],
            test_size=payload["data"]["test_size"],
        ),
        model=ModelConfig(
            type=payload["model"]["type"],
            max_features=payload["model"]["max_features"],
            ngram_range=tuple(payload["model"]["ngram_range"]),
            max_iter=payload["model"]["max_iter"],
        ),
    )
