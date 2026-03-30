from __future__ import annotations

from collections.abc import Sequence

from imdb_sentiment.settings import LSTMModelConfig

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - exercised only when torch is absent
    torch = None
    Tensor = object
    nn = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


if nn is not None:

    class SentimentLSTM(nn.Module):
        def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            hidden_dim: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
            self.encoder = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )
            self.dropout = nn.Dropout(p=dropout)
            self.classifier = nn.Linear(hidden_dim, 1)

        def forward(self, token_ids: Tensor) -> Tensor:
            embedded_tokens = self.embedding(token_ids)
            _, (hidden_state, _) = self.encoder(embedded_tokens)
            last_hidden_state = hidden_state[-1]
            dropped_hidden_state = self.dropout(last_hidden_state)
            logits = self.classifier(dropped_hidden_state)
            return logits.squeeze(dim=-1)

else:

    class SentimentLSTM:
        pass


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required to build the LSTM model. Install torch in the Colab or local "
            "runtime before calling build_lstm_model()."
        ) from TORCH_IMPORT_ERROR


def _validate_token_shape(token_ids: Tensor) -> None:
    if token_ids.ndim != 2:
        raise ValueError("SentimentLSTM expects token_ids with shape [batch_size, sequence_length].")


def _validate_config_type(config: LSTMModelConfig) -> None:
    if not isinstance(config, LSTMModelConfig):
        raise TypeError("build_lstm_model expects LSTMModelConfig.")


def build_lstm_model(config: LSTMModelConfig) -> SentimentLSTM:
    _require_torch()
    _validate_config_type(config)
    return SentimentLSTM(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    )


def predict_logits(model: SentimentLSTM, token_ids: Tensor) -> Tensor:
    _require_torch()
    _validate_token_shape(token_ids)
    return model(token_ids)


__all__: Sequence[str] = ["SentimentLSTM", "build_lstm_model", "predict_logits"]
