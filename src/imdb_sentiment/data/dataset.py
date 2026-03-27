from __future__ import annotations

from datasets import DatasetDict, load_dataset as hf_load_dataset


IMDB_DATASET_NAME = "imdb"


def load_imdb_dataset() -> DatasetDict:
    """Load the IMDb dataset from Hugging Face with predefined splits."""
    try:
        return hf_load_dataset(IMDB_DATASET_NAME)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load IMDb dataset from Hugging Face. "
            "Check network access or provide a local dataset fallback."
        ) from exc
