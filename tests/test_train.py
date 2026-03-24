from pathlib import Path

import pandas as pd

from imdb_sentiment.pipelines.train import run_training
from imdb_sentiment.settings import load_config


def test_run_training_returns_accuracy(tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        {
            "review": [
                "great movie",
                "excellent plot",
                "bad acting",
                "terrible ending",
                "loved it",
                "hated it",
            ],
            "sentiment": ["positive", "positive", "negative", "negative", "positive", "negative"],
        }
    )
    csv_path = tmp_path / "imdb.csv"
    dataset.to_csv(csv_path, index=False)

    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 42",
                "paths:",
                f"  raw_data: {csv_path.as_posix()}",
                "  model_output: artifacts/models/baseline.joblib",
                "  metrics_output: artifacts/reports/metrics.json",
                "data:",
                "  text_column: review",
                "  target_column: sentiment",
                "  test_size: 0.5",
                "model:",
                "  type: logistic_regression",
                "  max_features: 100",
                "  ngram_range: [1, 2]",
                "  max_iter: 100",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    metrics = run_training(config)

    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
