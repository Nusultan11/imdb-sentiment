from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from imdb_sentiment.data.dataset import load_imdb_dataset
from imdb_sentiment.inference.predict import load_model
from imdb_sentiment.settings import AppConfig


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _prepare_split(split: Dataset) -> tuple[list[str], list[int]]:
    texts = list(split["text"])
    labels = list(split["label"])
    return texts, labels


def _save_metrics(output_path: Path, metrics: dict[str, float]) -> None:
    _ensure_parent_dir(output_path)
    output_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )


def _evaluate_predictions(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def run_evaluation(
    config: AppConfig,
    output_path: str | Path | None = None,
) -> dict[str, float]:
    model: Pipeline = load_model(config.paths.model_output)

    dataset = load_imdb_dataset()
    test_split = dataset["test"]

    x_test, y_test = _prepare_split(test_split)
    y_pred = [int(pred) for pred in model.predict(x_test)]

    metrics = _evaluate_predictions(y_test, y_pred)

    resolved_output_path = config.paths.test_metrics_output if output_path is None else Path(output_path)
    _save_metrics(resolved_output_path, metrics)

    return metrics
