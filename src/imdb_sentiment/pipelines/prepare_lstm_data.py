from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from datasets import Dataset, DatasetDict

from imdb_sentiment.data.dataset import load_imdb_dataset
from imdb_sentiment.features.lstm_preprocessing import tokenize_lstm_text
from imdb_sentiment.features.preprocess import normalize_review_text
from imdb_sentiment.settings import AppConfig, LSTMModelConfig

LSTM_PREPARED_DATA_PIPELINE_ROLE = "export_only"


@dataclass(slots=True)
class LSTMPreparedDataPaths:
    train_path: Path
    val_path: Path
    test_path: Path
    metadata_path: Path


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _split_lstm_dataset(seed: int) -> DatasetDict:
    dataset = load_imdb_dataset()
    train_val_split = dataset["train"].train_test_split(
        test_size=0.2,
        seed=seed,
    )
    return DatasetDict(
        {
            "train": train_val_split["train"],
            "val": train_val_split["test"],
            "test": dataset["test"],
        }
    )


def _serialize_lstm_text(text: str, preprocessing: str) -> str:
    if preprocessing == "whitespace_v1":
        return normalize_review_text(text)
    return " ".join(tokenize_lstm_text(text, preprocessing=preprocessing))


def _write_jsonl(path: Path, split: Dataset, preprocessing: str) -> int:
    _ensure_parent_dir(path)

    row_count = 0
    with path.open("w", encoding="utf-8") as sink:
        for text, label in zip(split["text"], split["label"], strict=True):
            sink.write(
                json.dumps(
                    {
                        "text": _serialize_lstm_text(text, preprocessing=preprocessing),
                        "label": int(label),
                    },
                    ensure_ascii=False,
                )
            )
            sink.write("\n")
            row_count += 1

    return row_count


def _resolve_output_dir(config: AppConfig, output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    return config.paths.model_output.parent / "prepared_data"


def prepare_lstm_data(
    config: AppConfig,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Export LSTM-ready JSONL snapshots for inspection or external reuse.

    The main LSTM training pipeline still reads the IMDb dataset directly and
    creates its own train/validation split. These exported files are an
    auxiliary artifact bundle, not the source of truth for `run_lstm_training`.
    """
    if not isinstance(config.model, LSTMModelConfig):
        raise TypeError("LSTM data preparation expects LSTMModelConfig")

    dataset = _split_lstm_dataset(config.seed)
    resolved_output_dir = _resolve_output_dir(config, output_dir)

    train_path = resolved_output_dir / "train.jsonl"
    val_path = resolved_output_dir / "val.jsonl"
    test_path = resolved_output_dir / "test.jsonl"
    metadata_path = resolved_output_dir / "metadata.json"

    train_rows = _write_jsonl(train_path, dataset["train"], preprocessing=config.model.preprocessing)
    val_rows = _write_jsonl(val_path, dataset["val"], preprocessing=config.model.preprocessing)
    test_rows = _write_jsonl(test_path, dataset["test"], preprocessing=config.model.preprocessing)

    metadata = {
        "family": config.experiment.family,
        "name": config.experiment.name,
        "seed": config.seed,
        "format": "jsonl",
        "columns": ["text", "label"],
        "train_rows": train_rows,
        "val_rows": val_rows,
        "test_rows": test_rows,
        "max_length": config.model.max_length,
        "vocab_size": config.model.vocab_size,
        "batch_size": config.model.batch_size,
        "epochs": config.model.epochs,
        "preprocessing": config.model.preprocessing,
        "pipeline_role": LSTM_PREPARED_DATA_PIPELINE_ROLE,
        "used_by_training": False,
    }
    _ensure_parent_dir(metadata_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
        "metadata_path": metadata_path,
    }
