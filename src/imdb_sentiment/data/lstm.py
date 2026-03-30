from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from imdb_sentiment.features.preprocess import normalize_review_text

try:
    import torch
    from torch import Tensor
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - exercised only when torch is absent
    torch = None
    Tensor = object
    DataLoader = object
    Dataset = object
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def tokenize_lstm_text(text: str) -> list[str]:
    normalized_text = normalize_review_text(text)
    if not normalized_text:
        return []
    return normalized_text.split(" ")


@dataclass(slots=True)
class LSTMVocabulary:
    token_to_id: dict[str, int]

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    def lookup_token_id(self, token: str) -> int:
        return self.token_to_id.get(token, self.unk_id)

    def __len__(self) -> int:
        return len(self.token_to_id)


def build_lstm_vocabulary(
    texts: Iterable[str],
    max_size: int,
    min_frequency: int = 1,
) -> LSTMVocabulary:
    if max_size < 2:
        raise ValueError("max_size must be at least 2 to fit pad and unk tokens.")
    if min_frequency < 1:
        raise ValueError("min_frequency must be at least 1.")

    token_counter: Counter[str] = Counter()
    for text in texts:
        token_counter.update(tokenize_lstm_text(text))

    token_to_id = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }
    remaining_capacity = max_size - len(token_to_id)
    sorted_tokens = sorted(
        (
            (token, count)
            for token, count in token_counter.items()
            if count >= min_frequency and token not in token_to_id
        ),
        key=lambda item: (-item[1], item[0]),
    )

    for token, _count in sorted_tokens[:remaining_capacity]:
        token_to_id[token] = len(token_to_id)

    return LSTMVocabulary(token_to_id=token_to_id)


def encode_lstm_text(
    text: str,
    vocabulary: LSTMVocabulary,
    max_length: int,
) -> list[int]:
    if max_length < 1:
        raise ValueError("max_length must be at least 1.")

    token_ids = [vocabulary.lookup_token_id(token) for token in tokenize_lstm_text(text)]
    token_ids = token_ids[:max_length]
    padding_length = max_length - len(token_ids)
    if padding_length > 0:
        token_ids.extend([vocabulary.pad_id] * padding_length)
    return token_ids


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for the LSTM text pipeline. Install torch before building "
            "datasets or dataloaders."
        ) from TORCH_IMPORT_ERROR


if torch is not None:

    class LSTMTextDataset(Dataset):
        def __init__(
            self,
            texts: Sequence[str],
            labels: Sequence[int],
            vocabulary: LSTMVocabulary,
            max_length: int,
        ) -> None:
            if len(texts) != len(labels):
                raise ValueError("texts and labels must have the same length.")
            self._texts = list(texts)
            self._labels = [int(label) for label in labels]
            self._vocabulary = vocabulary
            self._max_length = max_length

        def __len__(self) -> int:
            return len(self._texts)

        def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
            token_ids = encode_lstm_text(
                text=self._texts[index],
                vocabulary=self._vocabulary,
                max_length=self._max_length,
            )
            return (
                torch.tensor(token_ids, dtype=torch.long),
                torch.tensor(self._labels[index], dtype=torch.float32),
            )

else:

    class LSTMTextDataset:
        def __init__(self, *args, **kwargs) -> None:
            _require_torch()


def build_lstm_dataloader(
    texts: Sequence[str],
    labels: Sequence[int],
    vocabulary: LSTMVocabulary,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
) -> DataLoader:
    _require_torch()
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    dataset = LSTMTextDataset(
        texts=texts,
        labels=labels,
        vocabulary=vocabulary,
        max_length=max_length,
    )
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


__all__ = [
    "LSTMTextDataset",
    "LSTMVocabulary",
    "PAD_TOKEN",
    "UNK_TOKEN",
    "build_lstm_dataloader",
    "build_lstm_vocabulary",
    "encode_lstm_text",
    "tokenize_lstm_text",
]
