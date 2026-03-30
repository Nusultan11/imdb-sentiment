from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset
import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from imdb_sentiment.data.dataset import load_imdb_dataset
from imdb_sentiment.models.tfidf.baseline import build_baseline_model
from imdb_sentiment.settings import AppConfig, TfidfModelConfig


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
    _ensure_parent_dir(config.paths.val_metrics_output)

    joblib.dump(model, config.paths.model_output)
    config.paths.val_metrics_output.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )


def run_tfidf_training(config: AppConfig) -> dict[str, float]:
    if not isinstance(config.model, TfidfModelConfig):
        raise TypeError("TF-IDF training expects TfidfModelConfig")

    dataset = load_imdb_dataset()
    full_train = dataset["train"]

    train_val_split = full_train.train_test_split(
        test_size=0.2,
        seed=config.seed,
    )

    train_split = train_val_split["train"]
    val_split = train_val_split["test"]

    x_train, y_train = _prepare_split(train_split)
    x_val, y_val = _prepare_split(val_split)

    model = build_baseline_model(
        max_features=config.model.max_features,
        ngram_range=config.model.ngram_range,
        max_iter=config.model.max_iter,
        random_state=config.seed,
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    _save_artifacts(config, model, metrics)

    return metrics
