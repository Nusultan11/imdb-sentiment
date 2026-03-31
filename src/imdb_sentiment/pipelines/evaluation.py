from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from imdb_sentiment.artifacts.lstm_runtime import load_restored_lstm_artifacts
from imdb_sentiment.data.dataset import load_imdb_dataset
from imdb_sentiment.data.lstm import build_lstm_dataloader
from imdb_sentiment.inference.predict import load_model
from imdb_sentiment.models.lstm.model import build_lstm_model
from imdb_sentiment.settings import AppConfig, LSTMModelConfig

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised only when torch is absent
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _prepare_split(split: Dataset) -> tuple[list[str], list[int]]:
    texts = list(split["text"])
    labels = [int(label) for label in split["label"]]
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


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for LSTM evaluation. Install torch before running run_evaluation()."
        ) from TORCH_IMPORT_ERROR


def _resolve_torch_device() -> torch.device:
    _require_torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_batch_to_device(token_ids: torch.Tensor, labels: torch.Tensor, device: torch.device):
    return token_ids.to(device), labels.to(device)


def _evaluate_tfidf_model(config: AppConfig) -> dict[str, float]:
    model: Pipeline = load_model(config.paths.model_output)

    dataset = load_imdb_dataset()
    test_split = dataset["test"]

    x_test, y_test = _prepare_split(test_split)
    y_pred = [int(pred) for pred in model.predict(x_test)]
    return _evaluate_predictions(y_test, y_pred)


def _evaluate_lstm_model(config: AppConfig) -> dict[str, float]:
    _require_torch()
    if not isinstance(config.model, LSTMModelConfig):
        raise TypeError("LSTM evaluation expects LSTMModelConfig")

    device = _resolve_torch_device()
    checkpoint = torch.load(config.paths.model_output, map_location=device)
    restored_artifacts = load_restored_lstm_artifacts(config.paths.model_output)

    dataset = load_imdb_dataset()
    test_split = dataset["test"]
    x_test, y_test = _prepare_split(test_split)

    dataloader = build_lstm_dataloader(
        texts=x_test,
        labels=y_test,
        vocabulary=restored_artifacts.vocabulary,
        max_length=restored_artifacts.model_config.max_length,
        batch_size=config.model.batch_size,
        shuffle=False,
        preprocessing=restored_artifacts.model_config.preprocessing,
        seed=config.seed,
    )
    model = build_lstm_model(restored_artifacts.model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_predictions: list[int] = []
    with torch.no_grad():
        for token_ids, _labels in dataloader:
            token_ids, _labels = _move_batch_to_device(token_ids, _labels, device)
            logits = model(token_ids)
            predictions = torch.sigmoid(logits).ge(restored_artifacts.decision_threshold).to(dtype=torch.int64)
            all_predictions.extend(predictions.cpu().tolist())

    return _evaluate_predictions(y_test, all_predictions)


def run_evaluation(
    config: AppConfig,
    output_path: str | Path | None = None,
) -> dict[str, float]:
    family = config.experiment.family

    if family == "tfidf":
        metrics = _evaluate_tfidf_model(config)
    elif family == "lstm":
        metrics = _evaluate_lstm_model(config)
    else:
        raise NotImplementedError(
            "Evaluation is not implemented for this experiment family yet."
        )

    resolved_output_path = config.paths.test_metrics_output if output_path is None else Path(output_path)
    _save_metrics(resolved_output_path, metrics)

    return metrics
