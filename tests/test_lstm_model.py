from __future__ import annotations

from pathlib import Path

import pytest
import torch

from imdb_sentiment.models.lstm.model import SentimentLSTM, build_lstm_model, predict_logits
from imdb_sentiment.settings import load_config


def _write_lstm_config(tmp_path: Path) -> Path:
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
                "  vocab_size: 100",
                "  max_length: 32",
                "  embedding_dim: 16",
                "  hidden_dim: 12",
                "  batch_size: 4",
                "  epochs: 2",
                "  dropout: 0.3",
                "  lr: 0.001",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def test_build_lstm_model_returns_sentiment_lstm(tmp_path: Path) -> None:
    config = load_config(_write_lstm_config(tmp_path))

    model = build_lstm_model(config.model)

    assert isinstance(model, SentimentLSTM)
    assert model.embedding.num_embeddings == 100
    assert model.embedding.embedding_dim == 16
    assert model.encoder.hidden_size == 12


def test_predict_logits_returns_one_logit_per_sequence(tmp_path: Path) -> None:
    config = load_config(_write_lstm_config(tmp_path))
    model = build_lstm_model(config.model)
    token_ids = torch.randint(low=0, high=100, size=(4, 10))

    logits = predict_logits(model, token_ids)

    assert logits.shape == (4,)
    assert logits.dtype == torch.float32


def test_predict_logits_rejects_non_batched_inputs(tmp_path: Path) -> None:
    config = load_config(_write_lstm_config(tmp_path))
    model = build_lstm_model(config.model)
    token_ids = torch.randint(low=0, high=100, size=(10,))

    with pytest.raises(
        ValueError,
        match="SentimentLSTM expects token_ids with shape \\[batch_size, sequence_length\\].",
    ):
        predict_logits(model, token_ids)


def test_build_lstm_model_rejects_non_lstm_config(tmp_path: Path) -> None:
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
                "  max_features: 100",
                "  ngram_range: [1, 2]",
                "  max_iter: 100",
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)

    with pytest.raises(TypeError, match="build_lstm_model expects LSTMModelConfig."):
        build_lstm_model(config.model)
