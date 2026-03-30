from pathlib import Path

import pytest

import imdb_sentiment.settings as settings_module
from imdb_sentiment.settings import LSTMModelConfig, TfidfModelConfig, TransformerModelConfig, load_config


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


def test_load_config_returns_tfidf_model_config_for_tfidf_family(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))

    assert isinstance(config.model, TfidfModelConfig)


def test_load_config_supports_lstm_family_specific_fields(tmp_path: Path) -> None:
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

    config = load_config(config_path)

    assert isinstance(config.model, LSTMModelConfig)
    assert config.experiment.family == "lstm"
    assert config.experiment.name == "baseline_test"
    assert config.seed == 42
    assert config.model.type == "lstm"
    assert config.model.vocab_size == 20000
    assert config.model.max_length == 64
    assert config.model.embedding_dim == 128
    assert config.model.hidden_dim == 128
    assert config.model.batch_size == 32
    assert config.model.epochs == 5
    assert config.model.dropout == 0.3
    assert config.model.lr == 0.001
    assert config.paths.model_output == tmp_path / "artifacts" / "models" / "model.pt"
    assert config.paths.val_metrics_output == tmp_path / "artifacts" / "reports" / "val_metrics.json"
    assert config.paths.test_metrics_output == tmp_path / "artifacts" / "reports" / "test_metrics.json"


def test_load_config_supports_transformer_family_specific_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "transformer.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  family: transformer",
                "  name: baseline_test",
                "seed: 42",
                "paths:",
                f"  model_output: {(tmp_path / 'artifacts' / 'models' / 'checkpoint').as_posix()}",
                f"  val_metrics_output: {(tmp_path / 'artifacts' / 'reports' / 'val_metrics.json').as_posix()}",
                f"  test_metrics_output: {(tmp_path / 'artifacts' / 'reports' / 'test_metrics.json').as_posix()}",
                "model:",
                "  type: distilbert",
                "  pretrained_model_name: distilbert-base-uncased",
                "  max_length: 256",
                "  batch_size: 16",
                "  epochs: 3",
                "  dropout: 0.1",
                "  lr: 0.00002",
                "  weight_decay: 0.01",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert isinstance(config.model, TransformerModelConfig)
    assert config.model.pretrained_model_name == "distilbert-base-uncased"


def test_load_config_rejects_invalid_lstm_dropout(tmp_path: Path) -> None:
    config_path = tmp_path / "lstm_invalid.yaml"
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
                "  dropout: 1.0",
                "  lr: 0.001",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model.dropout must be in the range \\[0, 1\\)"):
        load_config(config_path)


def test_load_config_rejects_non_lstm_model_type_for_lstm_family(tmp_path: Path) -> None:
    config_path = tmp_path / "lstm_invalid_type.yaml"
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
                "  type: logistic_regression",
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

    with pytest.raises(ValueError, match="LSTM experiments must use model.type=lstm"):
        load_config(config_path)


def test_load_config_resolves_project_relative_paths_outside_repo_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / "baseline.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  family: tfidf",
                "  name: baseline_test",
                "seed: 42",
                "paths:",
                "  model_output: artifacts/models/baseline.joblib",
                "  val_metrics_output: artifacts/reports/val_metrics.json",
                "  test_metrics_output: artifacts/reports/test_metrics.json",
                "model:",
                "  type: logistic_regression",
                "  max_features: 100",
                "  ngram_range: [1, 2]",
                "  max_iter: 100",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(settings_module, "PROJECT_ROOT", project_root)
    monkeypatch.chdir(tmp_path)

    config = load_config("configs/baseline.yaml")

    assert config.paths.model_output == project_root / "artifacts" / "models" / "baseline.joblib"
    assert config.paths.val_metrics_output == project_root / "artifacts" / "reports" / "val_metrics.json"
    assert config.paths.test_metrics_output == project_root / "artifacts" / "reports" / "test_metrics.json"
