from pathlib import Path

import pytest

from imdb_sentiment.settings import load_config


def test_load_config_raises_for_missing_required_key(tmp_path: Path) -> None:
    bad_config = tmp_path / "missing_seed.yaml"
    bad_config.write_text(
        "\n".join(
            [
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

    with pytest.raises(ValueError, match="seed must be an integer"):
        load_config(bad_config)


def test_load_config_raises_for_bad_ngram_type(tmp_path: Path) -> None:
    bad_config = tmp_path / "bad_ngram.yaml"
    bad_config.write_text(
        "\n".join(
            [
                "seed: 42",
                "paths:",
                "  model_output: artifacts/models/baseline.joblib",
                "  metrics_output: artifacts/reports/metrics.json",
                "model:",
                "  type: logistic_regression",
                "  max_features: 100",
                "  ngram_range: one_two",
                "  max_iter: 100",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ngram_range must be a list of two integers"):
        load_config(bad_config)
