from __future__ import annotations

from pathlib import Path

from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from imdb_sentiment.artifacts.lstm import (
    build_lstm_training_config_payload,
    build_lstm_threshold_tuning_payload,
    resolve_lstm_artifact_contract,
    write_json_artifact,
)
from imdb_sentiment.data.dataset import load_imdb_dataset
from imdb_sentiment.data.lstm import build_lstm_dataloader, build_lstm_vocabulary
from imdb_sentiment.models.lstm.model import build_lstm_model
from imdb_sentiment.settings import AppConfig, LSTMModelConfig

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - exercised only when torch is absent
    torch = None
    Tensor = object
    nn = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for LSTM training. Install torch before running run_lstm_training()."
        ) from TORCH_IMPORT_ERROR


def _resolve_torch_device() -> torch.device:
    _require_torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_batch_to_device(token_ids: Tensor, labels: Tensor, device: torch.device) -> tuple[Tensor, Tensor]:
    return token_ids.to(device), labels.to(device)


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def _split_train_dataset(dataset: DatasetDict, seed: int) -> tuple[Dataset, Dataset]:
    train_val_split = dataset["train"].train_test_split(
        test_size=0.2,
        seed=seed,
    )
    return train_val_split["train"], train_val_split["test"]


def _prepare_texts_and_labels(split: Dataset) -> tuple[list[str], list[int]]:
    return list(split["text"]), [int(label) for label in split["label"]]


def _set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)

def _train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    loss_fn,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    batch_count = 0

    for token_ids, labels in dataloader:
        token_ids, labels = _move_batch_to_device(token_ids, labels, device)
        optimizer.zero_grad()
        logits = model(token_ids)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        batch_count += 1

    if batch_count == 0:
        raise ValueError("LSTM training dataloader produced no batches.")

    return total_loss / batch_count


def _evaluate_lstm_model(
    model: nn.Module,
    dataloader,
    loss_fn,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    batch_count = 0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for token_ids, labels in dataloader:
            token_ids, labels = _move_batch_to_device(token_ids, labels, device)
            logits = model(token_ids)
            loss = loss_fn(logits, labels)
            total_loss += float(loss.item())
            batch_count += 1

            predictions = torch.sigmoid(logits).ge(0.5).to(dtype=torch.int64)
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.to(dtype=torch.int64).cpu().tolist())

    if batch_count == 0:
        raise ValueError("LSTM evaluation dataloader produced no batches.")

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    return {
        "loss": total_loss / batch_count,
        "accuracy": float(accuracy_score(all_labels, all_predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _build_training_history_payload(
    best_epoch: int,
    history: list[dict[str, float | int]],
) -> dict[str, object]:
    return {
        "best_epoch": best_epoch,
        "history": history,
    }


def _save_training_history(
    config: AppConfig,
    best_epoch: int,
    history: list[dict[str, float | int]],
) -> None:
    artifact_contract = resolve_lstm_artifact_contract(config)
    write_json_artifact(
        artifact_contract.training_history_output,
        _build_training_history_payload(best_epoch=best_epoch, history=history),
    )


def _save_best_lstm_artifacts(
    config: AppConfig,
    model: nn.Module,
    vocabulary,
    metrics: dict[str, float],
) -> None:
    artifact_contract = resolve_lstm_artifact_contract(config)
    _ensure_parent_dir(artifact_contract.model_output)
    _ensure_parent_dir(artifact_contract.val_metrics_output)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocabulary": vocabulary.token_to_id,
        "max_length": config.model.max_length,
        "family": config.experiment.family,
        "name": config.experiment.name,
    }
    torch.save(checkpoint, artifact_contract.model_output)
    write_json_artifact(artifact_contract.vocab_output, vocabulary.token_to_id)
    write_json_artifact(
        artifact_contract.training_config_output,
        build_lstm_training_config_payload(
            config=config,
            artifact_contract=artifact_contract,
        ),
    )
    write_json_artifact(
        artifact_contract.threshold_tuning_output,
        build_lstm_threshold_tuning_payload(),
    )
    write_json_artifact(artifact_contract.val_metrics_output, metrics)


def run_lstm_training(config: AppConfig) -> dict[str, float]:
    _require_torch()
    if not isinstance(config.model, LSTMModelConfig):
        raise TypeError("LSTM training expects LSTMModelConfig")

    device = _resolve_torch_device()
    _set_torch_seed(config.seed)
    dataset = load_imdb_dataset()
    train_split, val_split = _split_train_dataset(dataset, seed=config.seed)
    x_train, y_train = _prepare_texts_and_labels(train_split)
    x_val, y_val = _prepare_texts_and_labels(val_split)

    vocabulary = build_lstm_vocabulary(
        texts=x_train,
        max_size=config.model.vocab_size,
    )
    train_dataloader = build_lstm_dataloader(
        texts=x_train,
        labels=y_train,
        vocabulary=vocabulary,
        max_length=config.model.max_length,
        batch_size=config.model.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    val_dataloader = build_lstm_dataloader(
        texts=x_val,
        labels=y_val,
        vocabulary=vocabulary,
        max_length=config.model.max_length,
        batch_size=config.model.batch_size,
        shuffle=False,
        seed=config.seed,
    )

    model = build_lstm_model(config.model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.model.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_metrics: dict[str, float] | None = None
    best_f1 = float("-inf")
    best_epoch = 0
    training_history: list[dict[str, float | int]] = []

    for epoch_index in range(config.model.epochs):
        train_loss = _train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        validation_metrics = _evaluate_lstm_model(
            model=model,
            dataloader=val_dataloader,
            loss_fn=loss_fn,
            device=device,
        )
        epoch_number = epoch_index + 1
        history_entry: dict[str, float | int] = {
            "epoch": epoch_number,
            "train_loss": train_loss,
            "val_loss": validation_metrics["loss"],
            "val_f1": validation_metrics["f1"],
        }
        training_history.append(history_entry)
        if validation_metrics["f1"] > best_f1:
            best_f1 = validation_metrics["f1"]
            best_metrics = validation_metrics
            best_epoch = epoch_number
            _save_best_lstm_artifacts(
                config=config,
                model=model,
                vocabulary=vocabulary,
                metrics=validation_metrics,
            )

    if best_metrics is None:
        raise RuntimeError("LSTM training finished without producing validation metrics.")

    _save_training_history(
        config=config,
        best_epoch=best_epoch,
        history=training_history,
    )

    return best_metrics
