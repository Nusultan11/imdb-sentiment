from __future__ import annotations

import torch
import pytest

from imdb_sentiment.data.lstm import (
    PAD_TOKEN,
    UNK_TOKEN,
    LSTMTextDataset,
    build_lstm_dataloader,
    build_lstm_vocabulary,
    encode_lstm_text,
    tokenize_lstm_text,
)


def test_tokenize_lstm_text_normalizes_and_splits_text() -> None:
    raw_text = "Great<br /> movie &quot;night&quot;"

    tokens = tokenize_lstm_text(raw_text)

    assert tokens == ["great", "movie", "\"night\""]


def test_build_lstm_vocabulary_reserves_special_tokens_and_orders_by_frequency() -> None:
    vocabulary = build_lstm_vocabulary(
        texts=[
            "great movie",
            "great acting",
            "bad movie",
        ],
        max_size=5,
    )

    assert vocabulary.token_to_id[PAD_TOKEN] == 0
    assert vocabulary.token_to_id[UNK_TOKEN] == 1
    assert vocabulary.token_to_id["great"] == 2
    assert vocabulary.token_to_id["movie"] == 3
    assert vocabulary.token_to_id["acting"] == 4


def test_encode_lstm_text_truncates_and_pads_sequences() -> None:
    vocabulary = build_lstm_vocabulary(
        texts=["great movie tonight"],
        max_size=10,
    )

    token_ids = encode_lstm_text(
        text="great movie mystery tonight",
        vocabulary=vocabulary,
        max_length=5,
    )

    assert token_ids == [
        vocabulary.token_to_id["great"],
        vocabulary.token_to_id["movie"],
        vocabulary.unk_id,
        vocabulary.token_to_id["tonight"],
        vocabulary.pad_id,
    ]


def test_lstm_text_dataset_returns_token_ids_and_float_labels() -> None:
    vocabulary = build_lstm_vocabulary(
        texts=["great movie", "bad ending"],
        max_size=10,
    )
    dataset = LSTMTextDataset(
        texts=["great movie", "bad ending"],
        labels=[1, 0],
        vocabulary=vocabulary,
        max_length=4,
    )

    token_ids, label = dataset[0]

    assert token_ids.dtype == torch.long
    assert label.dtype == torch.float32
    assert token_ids.shape == (4,)
    assert label.item() == 1.0


def test_lstm_text_dataset_rejects_misaligned_texts_and_labels() -> None:
    vocabulary = build_lstm_vocabulary(texts=["great movie"], max_size=10)

    with pytest.raises(ValueError, match="texts and labels must have the same length."):
        LSTMTextDataset(
            texts=["great movie"],
            labels=[1, 0],
            vocabulary=vocabulary,
            max_length=4,
        )


def test_build_lstm_dataloader_returns_batched_tensors() -> None:
    vocabulary = build_lstm_vocabulary(
        texts=["great movie", "bad ending", "amazing acting"],
        max_size=10,
    )

    dataloader = build_lstm_dataloader(
        texts=["great movie", "bad ending", "amazing acting"],
        labels=[1, 0, 1],
        vocabulary=vocabulary,
        max_length=4,
        batch_size=2,
        shuffle=False,
    )
    token_ids, labels = next(iter(dataloader))

    assert token_ids.dtype == torch.long
    assert labels.dtype == torch.float32
    assert token_ids.shape == (2, 4)
    assert labels.shape == (2,)
