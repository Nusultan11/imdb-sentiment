from __future__ import annotations

from collections.abc import Sequence

from imdb_sentiment.settings import LSTMModelConfig

try:
    import torch
    from torch import Tensor, nn
    from torch.nn.utils.rnn import pack_padded_sequence
except ImportError as exc:  # pragma: no cover - exercised only when torch is absent
    torch = None
    Tensor = object
    nn = None
    pack_padded_sequence = None
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
            bidirectional: bool,
            padding_idx: int = 0,
        ) -> None:
            super().__init__()
            self.padding_idx = padding_idx
            self.bidirectional = bidirectional
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                padding_idx=padding_idx,
            )
            self.encoder = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=bidirectional,
            )
            classifier_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
            self.dropout = nn.Dropout(p=dropout)
            self.classifier = nn.Linear(classifier_input_dim, 1)

        def forward(self, token_ids: Tensor) -> Tensor:
            embedded_tokens = self.embedding(token_ids)
            token_lengths = token_ids.ne(self.padding_idx).sum(dim=1).clamp(min=1).cpu()
            packed_tokens = pack_padded_sequence(
                embedded_tokens,
                lengths=token_lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            _, (hidden_state, _) = self.encoder(packed_tokens)
            if self.bidirectional:
                forward_hidden = hidden_state[-2]
                backward_hidden = hidden_state[-1]
                last_hidden_state = torch.cat([forward_hidden, backward_hidden], dim=1)
            else:
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
        bidirectional=config.bidirectional,
        padding_idx=0,
    )


def predict_logits(model: SentimentLSTM, token_ids: Tensor) -> Tensor:
    _require_torch()
    _validate_token_shape(token_ids)
    return model(token_ids)


__all__: Sequence[str] = ["SentimentLSTM", "build_lstm_model", "predict_logits"]
