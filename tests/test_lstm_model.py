from __future__ import annotations

from pathlib import Path

import pytest
import torch

from imdb_sentiment.models.lstm.model import SentimentLSTM, build_lstm_model, predict_logits
from imdb_sentiment.settings import load_config


def _write_lstm_config(
    tmp_path: Path,
    *,
    bidirectional: bool = False,
    pooling: str = "last_hidden",
) -> Path:
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
                f"  bidirectional: {'true' if bidirectional else 'false'}",
                f"  pooling: {pooling}",
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
    assert model.embedding.padding_idx == 0
    assert model.encoder.hidden_size == 12


@pytest.mark.parametrize(
    ("bidirectional", "pooling"),
    [
        (False, "last_hidden"),
        (True, "last_hidden"),
        (False, "masked_mean"),
        (True, "masked_mean"),
    ],
)
def test_build_lstm_model_smoke_supports_all_directionality_and_pooling_combinations(
    tmp_path: Path,
    bidirectional: bool,
    pooling: str,
) -> None:
    config = load_config(
        _write_lstm_config(
            tmp_path,
            bidirectional=bidirectional,
            pooling=pooling,
        )
    )
    model = build_lstm_model(config.model)
    token_ids = torch.tensor(
        [
            [7, 8, 9, 0, 0],
            [4, 5, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    expected_feature_dim = 24 if bidirectional else 12

    encoder_output, hidden_state, token_mask = model._encode(token_ids)
    pooled = model._pool(encoder_output, hidden_state, token_mask)

    logits = model(token_ids)

    assert isinstance(model, SentimentLSTM)
    assert model.bidirectional is bidirectional
    assert model.pooling == pooling
    assert model.classifier.in_features == expected_feature_dim
    assert encoder_output.shape == (2, 5, expected_feature_dim)
    assert pooled.shape == (2, expected_feature_dim)
    assert logits.shape == (2,)
    assert logits.dtype == torch.float32


def test_sentiment_lstm_encoder_supports_bidirectional_mode() -> None:
    model = SentimentLSTM(
        vocab_size=100,
        embedding_dim=16,
        hidden_dim=12,
        dropout=0.3,
        bidirectional=True,
        pooling="last_hidden",
        padding_idx=0,
    )

    assert model.bidirectional is True
    assert model.pooling == "last_hidden"
    assert model.encoder.bidirectional is True
    assert model.classifier.in_features == 24


def test_sentiment_lstm_encode_returns_output_hidden_state_and_token_mask() -> None:
    model = SentimentLSTM(
        vocab_size=100,
        embedding_dim=16,
        hidden_dim=12,
        dropout=0.3,
        bidirectional=True,
        pooling="last_hidden",
        padding_idx=0,
    )
    token_ids = torch.tensor(
        [
            [5, 6, 7, 0],
            [8, 9, 0, 0],
        ],
        dtype=torch.long,
    )

    encoder_output, hidden_state, token_mask = model._encode(token_ids)

    assert encoder_output.shape == (2, 4, 24)
    assert hidden_state.shape == (2, 2, 12)
    assert token_mask.tolist() == [
        [True, True, True, False],
        [True, True, False, False],
    ]


def test_sentiment_lstm_forward_supports_bidirectional_hidden_concatenation() -> None:
    model = SentimentLSTM(
        vocab_size=100,
        embedding_dim=16,
        hidden_dim=12,
        dropout=0.3,
        bidirectional=True,
        pooling="last_hidden",
        padding_idx=0,
    )
    token_ids = torch.randint(low=1, high=100, size=(3, 7))

    logits = model(token_ids)

    assert logits.shape == (3,)
    assert logits.dtype == torch.float32


def test_sentiment_lstm_pool_supports_masked_mean() -> None:
    model = SentimentLSTM(
        vocab_size=100,
        embedding_dim=16,
        hidden_dim=2,
        dropout=0.3,
        bidirectional=False,
        pooling="masked_mean",
        padding_idx=0,
    )
    encoder_output = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],
            [[2.0, 0.0], [4.0, 0.0], [6.0, 0.0]],
        ]
    )
    hidden_state = torch.zeros((1, 2, 2), dtype=torch.float32)
    token_mask = torch.tensor(
        [
            [True, True, False],
            [True, True, True],
        ]
    )

    pooled = model._pool(encoder_output, hidden_state, token_mask)

    assert torch.allclose(
        pooled,
        torch.tensor(
            [
                [2.0, 3.0],
                [4.0, 0.0],
            ]
        ),
    )


def test_sentiment_lstm_pool_rejects_unsupported_strategy() -> None:
    model = SentimentLSTM(
        vocab_size=100,
        embedding_dim=16,
        hidden_dim=2,
        dropout=0.3,
        bidirectional=False,
        pooling="attention",
        padding_idx=0,
    )
    encoder_output = torch.zeros((1, 2, 2), dtype=torch.float32)
    hidden_state = torch.zeros((1, 1, 2), dtype=torch.float32)
    token_mask = torch.tensor([[True, False]])

    with pytest.raises(ValueError, match="Unsupported pooling strategy: attention"):
        model._pool(encoder_output, hidden_state, token_mask)


def test_sentiment_lstm_classifier_size_depends_only_on_directionality() -> None:
    unidirectional_last_hidden = SentimentLSTM(
        vocab_size=100,
        embedding_dim=16,
        hidden_dim=12,
        dropout=0.3,
        bidirectional=False,
        pooling="last_hidden",
        padding_idx=0,
    )
    unidirectional_masked_mean = SentimentLSTM(
        vocab_size=100,
        embedding_dim=16,
        hidden_dim=12,
        dropout=0.3,
        bidirectional=False,
        pooling="masked_mean",
        padding_idx=0,
    )
    bidirectional_last_hidden = SentimentLSTM(
        vocab_size=100,
        embedding_dim=16,
        hidden_dim=12,
        dropout=0.3,
        bidirectional=True,
        pooling="last_hidden",
        padding_idx=0,
    )
    bidirectional_masked_mean = SentimentLSTM(
        vocab_size=100,
        embedding_dim=16,
        hidden_dim=12,
        dropout=0.3,
        bidirectional=True,
        pooling="masked_mean",
        padding_idx=0,
    )

    assert unidirectional_last_hidden.classifier.in_features == 12
    assert unidirectional_masked_mean.classifier.in_features == 12
    assert bidirectional_last_hidden.classifier.in_features == 24
    assert bidirectional_masked_mean.classifier.in_features == 24


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


def test_predict_logits_ignores_trailing_padding_tokens(tmp_path: Path) -> None:
    config = load_config(_write_lstm_config(tmp_path))
    model = build_lstm_model(config.model)
    model.eval()

    short_sequence = torch.tensor([[7, 8]], dtype=torch.long)
    padded_sequence = torch.tensor([[7, 8, 0, 0]], dtype=torch.long)

    short_logits = predict_logits(model, short_sequence)
    padded_logits = predict_logits(model, padded_sequence)

    assert torch.allclose(short_logits, padded_logits)
