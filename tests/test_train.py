from pathlib import Path
import json

from datasets import Dataset, DatasetDict

import imdb_sentiment.pipelines.train as train_module
from imdb_sentiment.settings import load_config


def test_run_training_returns_accuracy(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 42",
                "paths:",
                f"  model_output: {tmp_path.as_posix()}/artifacts/models/baseline.joblib",
                f"  metrics_output: {tmp_path.as_posix()}/artifacts/reports/metrics.json",
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
    monkeypatch.setattr(train_module, "load_imdb_dataset", lambda: fake_dataset)

    config = load_config(config_path)
    metrics = train_module.run_training(config)

    assert "accuracy" in metrics
    assert {"accuracy", "precision", "recall", "f1"} <= metrics.keys()
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert config.paths.model_output.exists()
    assert config.paths.metrics_output.exists()
    saved_metrics = json.loads(config.paths.metrics_output.read_text(encoding="utf-8"))
    assert saved_metrics == metrics
