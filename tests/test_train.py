from pathlib import Path

import imdb_sentiment.pipelines.train as train_module
from imdb_sentiment.settings import load_config


def test_run_training_returns_accuracy(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 42",
                "paths:",
                "  model_output: artifacts/models/baseline.joblib",
                "  metrics_output: artifacts/reports/metrics.json",
                "model:",
                "  type: logistic_regression",
                "  max_features: 100",
                "  ngram_range: [1, 2]",
                "  max_iter: 100",
            ]
        ),
        encoding="utf-8",
    )

    fake_dataset = {
        "train": {
            "text": ["great movie", "excellent plot", "bad acting", "terrible ending"],
            "label": [1, 1, 0, 0],
        },
        "test": {
            "text": ["loved it", "hated it"],
            "label": [1, 0],
        },
    }
    monkeypatch.setattr(train_module, "load_dataset", lambda _: fake_dataset)

    config = load_config(config_path)
    metrics = train_module.run_training(config)

    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
