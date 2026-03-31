from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, DatasetDict
import torch

import imdb_sentiment.pipelines.train_lstm as train_lstm_module
from imdb_sentiment.settings import load_config


def test_resolve_torch_device_prefers_cuda_when_available(monkeypatch) -> None:
    monkeypatch.setattr(train_lstm_module.torch.cuda, "is_available", lambda: True)

    device = train_lstm_module._resolve_torch_device()

    assert device.type == "cuda"


def test_move_batch_to_device_moves_tensors_to_cpu() -> None:
    token_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    labels = torch.tensor([1.0, 0.0], dtype=torch.float32)

    moved_token_ids, moved_labels = train_lstm_module._move_batch_to_device(
        token_ids,
        labels,
        torch.device("cpu"),
    )

    assert moved_token_ids.device.type == "cpu"
    assert moved_labels.device.type == "cpu"
    assert moved_token_ids.tolist() == [[1, 2], [3, 4]]
    assert moved_labels.tolist() == [1.0, 0.0]


def test_select_best_threshold_returns_best_f1_metrics() -> None:
    best_threshold, best_metrics = train_lstm_module._select_best_threshold(
        labels=[1, 1, 0, 0],
        probabilities=[0.91, 0.55, 0.45, 0.05],
    )

    assert best_threshold == 0.46
    assert best_metrics == {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }


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
                "  bidirectional: false",
                "  pooling: last_hidden",
                "  preprocessing: regex_v2",
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
    training_history_output = config.paths.model_output.parent / "training_history.json"
    threshold_tuning_output = config.paths.model_output.parent / "threshold_tuning.json"
    assert vocab_output.exists()
    assert training_config_output.exists()
    assert training_history_output.exists()
    assert threshold_tuning_output.exists()

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
        "bidirectional": False,
        "pooling": "last_hidden",
        "preprocessing": "regex_v2",
    }
    assert saved_training_config["artifacts"] == {
        "model_output": "model.pt",
        "vocab_output": "vocab.json",
        "training_config_output": "training_config.json",
        "training_history_output": "training_history.json",
        "threshold_tuning_output": "threshold_tuning.json",
        "val_metrics_output": "val_metrics.json",
        "test_metrics_output": "test_metrics.json",
    }
    assert saved_training_config["required_for_inference"] == [
        "model.pt",
        "vocab.json",
        "training_config.json",
    ]
    assert saved_training_config["required_for_evaluation"] == [
        "model.pt",
        "vocab.json",
        "training_config.json",
    ]
    saved_training_history = json.loads(training_history_output.read_text(encoding="utf-8"))
    saved_threshold_tuning = json.loads(threshold_tuning_output.read_text(encoding="utf-8"))
    assert saved_training_history["best_epoch"] == 1
    assert len(saved_training_history["history"]) == 2
    assert saved_training_history["history"][0]["epoch"] == 1
    assert "train_loss" in saved_training_history["history"][0]
    assert "val_loss" in saved_training_history["history"][0]
    assert "val_f1" in saved_training_history["history"][0]
    assert 0.1 <= saved_threshold_tuning["decision_threshold"] <= 0.9
    assert saved_threshold_tuning["selection_strategy"] == "validation_best_f1"

    checkpoint = torch.load(config.paths.model_output)
    assert set(checkpoint) == {"model_state_dict", "vocabulary", "max_length", "family", "name"}
    assert checkpoint["max_length"] == 6
    assert checkpoint["family"] == "lstm"
    assert checkpoint["name"] == "baseline_test"
    assert "<pad>" in checkpoint["vocabulary"]


def test_run_lstm_training_builds_model_from_lstm_architecture_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "lstm_bidirectional.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  family: lstm",
                "  name: bidirectional_test",
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
                "  bidirectional: true",
                "  pooling: masked_mean",
                "  preprocessing: regex_v2",
                "  batch_size: 2",
                "  epochs: 1",
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
                    "text": ["great movie", "bad film", "loved it", "hated it"],
                    "label": [1, 0, 1, 0],
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

    class _FakeVocabulary:
        token_to_id = {"<pad>": 0, "<unk>": 1}

    monkeypatch.setattr(
        train_lstm_module,
        "build_lstm_vocabulary",
        lambda texts, max_size, preprocessing="whitespace_v1": _FakeVocabulary(),
    )
    monkeypatch.setattr(train_lstm_module, "build_lstm_dataloader", lambda **kwargs: [("unused", "unused")])

    captured: dict[str, object] = {}

    def _fake_build_lstm_model(model_config):
        captured["bidirectional"] = model_config.bidirectional
        captured["pooling"] = model_config.pooling
        captured["preprocessing"] = model_config.preprocessing
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(train_lstm_module, "build_lstm_model", _fake_build_lstm_model)
    monkeypatch.setattr(train_lstm_module, "_train_one_epoch", lambda **kwargs: 0.1)
    monkeypatch.setattr(
        train_lstm_module,
        "_evaluate_lstm_model",
        lambda **kwargs: (
            {
                "loss": 0.2,
                "accuracy": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
            },
            0.4,
        ),
    )
    monkeypatch.setattr(train_lstm_module, "_save_best_lstm_artifacts", lambda **kwargs: None)
    monkeypatch.setattr(train_lstm_module, "_save_training_history", lambda **kwargs: None)

    config = load_config(config_path)
    metrics = train_lstm_module.run_lstm_training(config)

    assert captured["bidirectional"] is True
    assert captured["pooling"] == "masked_mean"
    assert captured["preprocessing"] == "regex_v2"
    assert metrics["f1"] == 1.0
