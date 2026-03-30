from pathlib import Path
import json

from datasets import Dataset, DatasetDict

import imdb_sentiment.pipelines.train as train_module
import imdb_sentiment.pipelines.train_tfidf as train_tfidf_module
from imdb_sentiment.settings import load_config


def test_run_training_returns_accuracy(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  family: tfidf",
                "  name: baseline_test",
                "seed: 42",
                "paths:",
                f"  model_output: {tmp_path.as_posix()}/artifacts/models/baseline.joblib",
                f"  val_metrics_output: {tmp_path.as_posix()}/artifacts/reports/val_metrics.json",
                f"  test_metrics_output: {tmp_path.as_posix()}/artifacts/reports/test_metrics.json",
                "model:",
                "  type: logistic_regression",
                "  max_features: 100",
                "  ngram_range: [1, 2]",
                "  max_iter: 100",
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
                        "excellent plot",
                        "amazing acting",
                        "loved every scene",
                        "bad acting",
                        "terrible ending",
                        "boring script",
                        "hated every minute",
                    ],
                    "label": [1, 1, 1, 1, 0, 0, 0, 0],
                }
            ),
            "test": Dataset.from_dict(
                {
                    "text": ["loved it", "hated it"],
                    "label": [1, 0],
                }
            ),
        }
    )
    monkeypatch.setattr(train_tfidf_module, "load_imdb_dataset", lambda: fake_dataset)

    config = load_config(config_path)
    metrics = train_module.run_training(config)

    assert "accuracy" in metrics
    assert {"accuracy", "precision", "recall", "f1"} <= metrics.keys()
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert config.paths.model_output.exists()
    assert config.paths.val_metrics_output.exists()
    assert not config.paths.test_metrics_output.exists()
    saved_metrics = json.loads(config.paths.val_metrics_output.read_text(encoding="utf-8"))
    assert saved_metrics == metrics


def test_run_training_routes_lstm_family_to_lstm_trainer(tmp_path: Path, monkeypatch) -> None:
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
                "  max_length: 32",
                "  embedding_dim: 16",
                "  hidden_dim: 16",
                "  bidirectional: false",
                "  batch_size: 2",
                "  epochs: 1",
                "  dropout: 0.3",
                "  lr: 0.001",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    expected_metrics = {
        "loss": 0.4,
        "accuracy": 0.75,
        "precision": 0.8,
        "recall": 0.67,
        "f1": 0.73,
    }
    monkeypatch.setattr(train_module, "run_lstm_training", lambda config_arg: expected_metrics)

    metrics = train_module.run_training(config)

    assert metrics == expected_metrics
