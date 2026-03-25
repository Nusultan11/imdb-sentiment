from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score

from imdb_sentiment.data import dataset as dataset_module
from imdb_sentiment.features.preprocess import normalize_review_text
from imdb_sentiment.models.baseline import build_baseline_model
from imdb_sentiment.settings import AppConfig


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def run_training(config: AppConfig) -> dict[str, float]:
    dataset = dataset_module.load_imdb_dataset()
    train_split = dataset["train"]
    test_split = dataset["test"]

    x_train = [normalize_review_text(text) for text in train_split["text"]]
    y_train = train_split["label"]

    x_test = [normalize_review_text(text) for text in test_split["text"]]
    y_test = test_split["label"]

    model = build_baseline_model(
        max_features=config.model.max_features,
        ngram_range=config.model.ngram_range,
        max_iter=config.model.max_iter,
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    metrics = {"accuracy": float(accuracy_score(y_test, y_pred))}
    _ensure_parent_dir(config.paths.model_output)
    _ensure_parent_dir(config.paths.metrics_output)

    joblib.dump(model, config.paths.model_output)
    config.paths.metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics
