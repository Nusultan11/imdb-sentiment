import json
from pathlib import Path

from datasets import Dataset, DatasetDict
import joblib
import torch

import imdb_sentiment.pipelines.evaluation as evaluation_module
from imdb_sentiment.data.lstm import build_lstm_vocabulary
from imdb_sentiment.models.baseline import build_baseline_model
from imdb_sentiment.models.lstm.model import build_lstm_model
from imdb_sentiment.settings import load_config


def test_run_evaluation_writes_test_metrics(tmp_path: Path, monkeypatch) -> None:
    model = build_baseline_model(
        max_features=100,
        ngram_range=(1, 2),
        max_iter=100,
        random_state=42,
    )
    model.fit(
        ["great movie", "excellent acting", "bad plot", "awful ending"],
        [1, 1, 0, 0],
    )

    model_path = tmp_path / "artifacts" / "models" / "baseline.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  family: tfidf",
                "  name: baseline_test",
                "seed: 42",
                "paths:",
                f"  model_output: {model_path.as_posix()}",
                f"  val_metrics_output: {(tmp_path / 'artifacts' / 'reports' / 'val_metrics.json').as_posix()}",
                f"  test_metrics_output: {(tmp_path / 'artifacts' / 'reports' / 'test_metrics.json').as_posix()}",
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
            "train": Dataset.from_dict({"text": ["placeholder"], "label": [1]}),
            "test": Dataset.from_dict(
                {
                    "text": ["great film", "terrible movie"],
                    "label": [1, 0],
                }
            ),
        }
    )
    monkeypatch.setattr(evaluation_module, "load_imdb_dataset", lambda: fake_dataset)

    config = load_config(config_path)
    output_path = tmp_path / "artifacts" / "reports" / "test_metrics.json"
    metrics = evaluation_module.run_evaluation(config, output_path=output_path)

    assert {"accuracy", "precision", "recall", "f1"} <= metrics.keys()
    assert output_path.exists()
    saved_metrics = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved_metrics == metrics


def test_run_evaluation_uses_config_test_metrics_output_by_default(tmp_path: Path, monkeypatch) -> None:
    model = build_baseline_model(
        max_features=100,
        ngram_range=(1, 2),
        max_iter=100,
        random_state=42,
    )
    model.fit(
        ["great movie", "excellent acting", "bad plot", "awful ending"],
        [1, 1, 0, 0],
    )

    model_path = tmp_path / "artifacts" / "models" / "baseline.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    test_metrics_output = tmp_path / "artifacts" / "reports" / "from_config_test_metrics.json"
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  family: tfidf",
                "  name: baseline_test",
                "seed: 42",
                "paths:",
                f"  model_output: {model_path.as_posix()}",
                f"  val_metrics_output: {(tmp_path / 'artifacts' / 'reports' / 'val_metrics.json').as_posix()}",
                f"  test_metrics_output: {test_metrics_output.as_posix()}",
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
            "train": Dataset.from_dict({"text": ["placeholder"], "label": [1]}),
            "test": Dataset.from_dict(
                {
                    "text": ["great film", "terrible movie"],
                    "label": [1, 0],
                }
            ),
        }
    )
    monkeypatch.setattr(evaluation_module, "load_imdb_dataset", lambda: fake_dataset)

    config = load_config(config_path)
    metrics = evaluation_module.run_evaluation(config)

    assert {"accuracy", "precision", "recall", "f1"} <= metrics.keys()
    assert test_metrics_output.exists()


def test_run_evaluation_supports_lstm_checkpoints(tmp_path: Path, monkeypatch) -> None:
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
                    "text": ["great movie", "bad ending"],
                    "label": [1, 0],
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
    monkeypatch.setattr(evaluation_module, "load_imdb_dataset", lambda: fake_dataset)

    config = load_config(config_path)
    vocabulary = build_lstm_vocabulary(["great movie", "bad ending"], max_size=50)
    model = build_lstm_model(config.model)
    config.paths.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocabulary": vocabulary.token_to_id,
            "max_length": config.model.max_length,
            "family": config.experiment.family,
            "name": config.experiment.name,
        },
        config.paths.model_output,
    )

    metrics = evaluation_module.run_evaluation(config)

    assert {"accuracy", "precision", "recall", "f1"} <= metrics.keys()
    assert config.paths.test_metrics_output.exists()
    saved_metrics = json.loads(config.paths.test_metrics_output.read_text(encoding="utf-8"))
    assert saved_metrics == metrics
