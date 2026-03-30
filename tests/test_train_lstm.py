from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, DatasetDict
import torch

import imdb_sentiment.pipelines.train_lstm as train_lstm_module
from imdb_sentiment.settings import load_config


def test_run_lstm_training_saves_best_checkpoint_and_metrics(
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
                "  vocab_size: 50",
                "  max_length: 6",
                "  embedding_dim: 8",
                "  hidden_dim: 6",
                "  batch_size: 2",
                "  epochs: 2",
                "  dropout: 0.2",
                "  lr: 0.01",
            ]
        ),
        encoding="utf-8",
    )

    fake_dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "text": [
                        "great movie",
                        "great acting",
                        "loved the ending",
                        "excellent film",
                        "bad script",
                        "terrible acting",
                        "boring story",
                        "hated the ending",
                        "wonderful scenes",
                        "awful pacing",
                    ],
                    "label": [1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
                }
            ),
            "test": Dataset.from_dict(
                {
                    "text": ["great film", "awful film"],
                    "label": [1, 0],
                }
            ),
        }
    )
    monkeypatch.setattr(train_lstm_module, "load_imdb_dataset", lambda: fake_dataset)

    config = load_config(config_path)
    metrics = train_lstm_module.run_lstm_training(config)

    assert {"loss", "accuracy", "precision", "recall", "f1"} <= metrics.keys()
    assert config.paths.model_output.exists()
    assert config.paths.val_metrics_output.exists()
    assert not config.paths.test_metrics_output.exists()
    vocab_output = config.paths.model_output.parent / "vocab.json"
    training_config_output = config.paths.model_output.parent / "training_config.json"
    assert vocab_output.exists()
    assert training_config_output.exists()

    saved_metrics = json.loads(config.paths.val_metrics_output.read_text(encoding="utf-8"))
    assert saved_metrics == metrics
    saved_vocabulary = json.loads(vocab_output.read_text(encoding="utf-8"))
    assert saved_vocabulary["<pad>"] == 0
    assert saved_vocabulary["<unk>"] == 1
    saved_training_config = json.loads(training_config_output.read_text(encoding="utf-8"))
    assert saved_training_config["experiment"] == {"family": "lstm", "name": "baseline_test"}
    assert saved_training_config["seed"] == 42
    assert saved_training_config["model"] == {
        "type": "lstm",
        "vocab_size": 50,
        "max_length": 6,
        "embedding_dim": 8,
        "hidden_dim": 6,
        "batch_size": 2,
        "epochs": 2,
        "dropout": 0.2,
        "lr": 0.01,
    }
    assert saved_training_config["artifacts"] == {
        "model_output": "model.pt",
        "vocab_output": "vocab.json",
        "training_config_output": "training_config.json",
        "val_metrics_output": "val_metrics.json",
    }

    checkpoint = torch.load(config.paths.model_output)
    assert set(checkpoint) == {"model_state_dict", "vocabulary", "max_length", "family", "name"}
    assert checkpoint["max_length"] == 6
    assert checkpoint["family"] == "lstm"
    assert checkpoint["name"] == "baseline_test"
    assert "<pad>" in checkpoint["vocabulary"]
