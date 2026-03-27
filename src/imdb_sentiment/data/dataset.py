from __future__ import annotations

from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset as hf_load_dataset


IMDB_DATASET_NAME = "imdb"
LOCAL_TRAIN_CSV = Path("data/raw/imdb_train.csv")
LOCAL_TEST_CSV = Path("data/raw/imdb_test.csv")
REQUIRED_COLUMNS = {"text", "label"}


def _validate_split_columns(split: Dataset, split_name: str) -> None:
    missing = REQUIRED_COLUMNS.difference(split.column_names)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"{split_name} split is missing required columns: {missing_list}")


def _validate_dataset(dataset: DatasetDict) -> DatasetDict:
    if "train" not in dataset or "test" not in dataset:
        raise ValueError("Dataset must contain 'train' and 'test' splits")

    _validate_split_columns(dataset["train"], "train")
    _validate_split_columns(dataset["test"], "test")
    return dataset


def _load_remote_dataset() -> DatasetDict:
    dataset = hf_load_dataset(IMDB_DATASET_NAME)
    return _validate_dataset(dataset)


def _load_local_csv_fallback() -> DatasetDict:
    if not LOCAL_TRAIN_CSV.exists() or not LOCAL_TEST_CSV.exists():
        raise FileNotFoundError(
            f"Local fallback files not found: {LOCAL_TRAIN_CSV} and {LOCAL_TEST_CSV}"
        )

    dataset = hf_load_dataset(
        "csv",
        data_files={
            "train": str(LOCAL_TRAIN_CSV),
            "test": str(LOCAL_TEST_CSV),
        },
    )
    return _validate_dataset(dataset)


def load_imdb_dataset() -> DatasetDict:
    """Load IMDb from Hugging Face, fallback to local train/test CSV files."""
    try:
        return _load_remote_dataset()
    except Exception as remote_exc:
        try:
            return _load_local_csv_fallback()
        except Exception as local_exc:
            raise RuntimeError(
                "Failed to load IMDb dataset from both sources. "
                f"Hugging Face error: {remote_exc}. "
                f"Local CSV error: {local_exc}. "
                f"Expected local files: {LOCAL_TRAIN_CSV} and {LOCAL_TEST_CSV}"
            ) from local_exc