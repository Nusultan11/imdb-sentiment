from __future__ import annotations

from datasets import DatasetDict, load_dataset as hf_load_dataset


def load_imdb_dataset() -> DatasetDict:
    return hf_load_dataset("imdb")
