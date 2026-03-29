from pathlib import Path

import pytest

from imdb_sentiment.settings import load_config


def _write_config(
    tmp_path: Path,
    ngram_range: str = "[1, 2]",
    max_features: int = 100,
    max_iter: int = 100,
) -> Path:
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  family: tfidf",
                "  name: baseline_test",
                "seed: 42",
                "paths:",
                f"  model_output: {(tmp_path / 'artifacts' / 'models' / 'baseline.joblib').as_posix()}",
                f"  val_metrics_output: {(tmp_path / 'artifacts' / 'reports' / 'val_metrics.json').as_posix()}",
                f"  test_metrics_output: {(tmp_path / 'artifacts' / 'reports' / 'test_metrics.json').as_posix()}",
                "model:",
                "  type: logistic_regression",
                f"  max_features: {max_features}",
                f"  ngram_range: {ngram_range}",
                f"  max_iter: {max_iter}",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def test_load_config_rejects_ngram_range_with_minimum_less_than_one(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, "[0, 2]")

    with pytest.raises(ValueError, match="model.ngram_range minimum must be at least 1"):
        load_config(config_path)


def test_load_config_rejects_ngram_range_with_reversed_bounds(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, "[2, 1]")

    with pytest.raises(
        ValueError,
        match="model.ngram_range minimum must be less than or equal to maximum",
    ):
        load_config(config_path)


def test_load_config_rejects_non_positive_max_features(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, max_features=0)

    with pytest.raises(ValueError, match="model.max_features must be at least 1"):
        load_config(config_path)


def test_load_config_rejects_non_positive_max_iter(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, max_iter=0)

    with pytest.raises(ValueError, match="model.max_iter must be at least 1"):
        load_config(config_path)
