from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, DatasetDict

import imdb_sentiment.pipelines.prepare_lstm_data as prepare_lstm_data_module
from imdb_sentiment.settings import load_config


def test_prepare_lstm_data_writes_train_val_test_and_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "lstm.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  family: lstm",
                "  name: baseline_test",
                "seed: 42",
                "paths:",
                f"  model_output: {(tmp_path / 'artifacts' / 'models' / 'model.pt').as_posix()}",
                f"  val_metrics_output: {(tmp_path / 'artifacts' / 'reports' / 'val_metrics.json').as_posix()}",
                f"  test_metrics_output: {(tmp_path / 'artifacts' / 'reports' / 'test_metrics.json').as_posix()}",
                "model:",
                "  type: lstm",
                "  vocab_size: 20000",
                "  max_length: 64",
                "  embedding_dim: 128",
                "  hidden_dim: 128",
                "  batch_size: 32",
                "  epochs: 5",
                "  dropout: 0.3",
                "  lr: 0.001",
            ]
        ),
        encoding="utf-8",
    )

    fake_dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "text": [
                        "Great movie",
                        "Excellent plot",
                        "Amazing acting",
                        "Loved every scene",
                        "Bad acting",
                        "Terrible ending",
                        "Boring script",
                        "Hated every minute",
                        "Wonderful soundtrack",
                        "Awful pacing",
                    ],
                    "label": [1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
                }
            ),
            "test": Dataset.from_dict(
                {
                    "text": ["Wonderful film", "Awful film"],
                    "label": [1, 0],
                }
            ),
        }
    )
    monkeypatch.setattr(prepare_lstm_data_module, "load_imdb_dataset", lambda: fake_dataset)

    config = load_config(config_path)
    prepared_paths = prepare_lstm_data_module.prepare_lstm_data(config)

    train_rows = prepared_paths["train_path"].read_text(encoding="utf-8").strip().splitlines()
    val_rows = prepared_paths["val_path"].read_text(encoding="utf-8").strip().splitlines()
    test_rows = prepared_paths["test_path"].read_text(encoding="utf-8").strip().splitlines()
    metadata = json.loads(prepared_paths["metadata_path"].read_text(encoding="utf-8"))

    assert len(train_rows) == 8
    assert len(val_rows) == 2
    assert len(test_rows) == 2
    assert metadata["train_rows"] == 8
    assert metadata["val_rows"] == 2
    assert metadata["test_rows"] == 2
    assert metadata["format"] == "jsonl"
    assert metadata["max_length"] == 64

    first_train_record = json.loads(train_rows[0])
    assert set(first_train_record) == {"text", "label"}
    assert first_train_record["text"] == first_train_record["text"].lower()
