from __future__ import annotations

from collections.abc import Sequence

from imdb_sentiment.settings import LSTMModelConfig

try:
    import torch
    from torch import Tensor, nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
except ImportError as exc:  # pragma: no cover - exercised only when torch is absent
    torch = None
    Tensor = object
    nn = None
    pack_padded_sequence = None
    pad_packed_sequence = None
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
            pooling: str,
            padding_idx: int = 0,
        ) -> None:
            super().__init__()
            self.padding_idx = padding_idx
            self.bidirectional = bidirectional
            self.pooling = pooling
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

        def _encode(self, token_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            token_mask = token_ids.ne(self.padding_idx)
            token_lengths = token_mask.sum(dim=1).clamp(min=1).cpu()
            embedded_tokens = self.embedding(token_ids)
            packed_tokens = pack_padded_sequence(
                embedded_tokens,
                lengths=token_lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, (hidden_state, _) = self.encoder(packed_tokens)
            encoder_output, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=token_ids.size(1),
            )
            return encoder_output, hidden_state, token_mask

        def _pool(self, encoder_output: Tensor, hidden_state: Tensor, token_mask: Tensor) -> Tensor:
            if self.pooling == "last_hidden":
                if self.bidirectional:
                    forward_hidden = hidden_state[-2]
                    backward_hidden = hidden_state[-1]
                    return torch.cat([forward_hidden, backward_hidden], dim=1)
                return hidden_state[-1]

            if self.pooling == "masked_mean":
                mask = token_mask.unsqueeze(-1).to(dtype=encoder_output.dtype)
                summed = (encoder_output * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1.0)
                return summed / counts

            raise ValueError(f"Unsupported pooling strategy: {self.pooling}")

        def forward(self, token_ids: Tensor) -> Tensor:
            encoder_output, hidden_state, token_mask = self._encode(token_ids)
            pooled = self._pool(encoder_output, hidden_state, token_mask)
            pooled = self.dropout(pooled)
            logits = self.classifier(pooled)
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
        pooling=config.pooling,
        padding_idx=0,
    )


def predict_logits(model: SentimentLSTM, token_ids: Tensor) -> Tensor:
    _require_torch()
    _validate_token_shape(token_ids)
    return model(token_ids)


__all__: Sequence[str] = ["SentimentLSTM", "build_lstm_model", "predict_logits"]
