from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset
import joblib
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from imdb_sentiment.data import dataset as dataset_module
from imdb_sentiment.models.baseline import build_baseline_model
from imdb_sentiment.settings import AppConfig


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _prepare_split(split: Dataset) -> tuple[list[str], list[int]]:
    texts = list(split["text"])
    labels = list(split["label"])
    return texts, labels


def _save_artifacts(
    config: AppConfig,
    model: Pipeline,
    metrics: dict[str, float],
) -> None:
    _ensure_parent_dir(config.paths.model_output)
    _ensure_parent_dir(config.paths.metrics_output)

    joblib.dump(model, config.paths.model_output)
    config.paths.metrics_output.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )


def run_training(config: AppConfig) -> dict[str, float]:
    if config.model.type != "logistic_regression":
        raise ValueError(f"Unsupported model.type: {config.model.type}")

    dataset = dataset_module.load_imdb_dataset()
    train_split = dataset["train"]
    test_split = dataset["test"]

    x_train, y_train = _prepare_split(train_split)
    x_test, y_test = _prepare_split(test_split)

    model = build_baseline_model(
        max_features=config.model.max_features,
        ngram_range=config.model.ngram_range,
        max_iter=config.model.max_iter,
        random_state=config.seed,
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    metrics = {"accuracy": float(accuracy_score(y_test, y_pred))}

    _save_artifacts(config, model, metrics)

    return metrics
