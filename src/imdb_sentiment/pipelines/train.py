from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from imdb_sentiment.data.dataset import load_imdb_dataset
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
    metrics: dict[str, float | list[list[int]]],
) -> None:
    _ensure_parent_dir(config.paths.model_output)
    _ensure_parent_dir(config.paths.metrics_output)

    joblib.dump(model, config.paths.model_output)
    config.paths.metrics_output.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )


def run_training(config: AppConfig) -> dict[str, float | list[list[int]]]:
    if config.model.type != "logistic_regression":
        raise ValueError(f"Unsupported model.type: {config.model.type}")

    dataset = load_imdb_dataset()
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
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()

    metrics: dict[str, float | list[list[int]]] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
    }

    _save_artifacts(config, model, metrics)

    return metrics
