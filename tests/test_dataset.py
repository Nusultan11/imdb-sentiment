import pytest
from datasets import Dataset, DatasetDict

import imdb_sentiment.data.dataset as dataset_module


def test_load_imdb_dataset_uses_local_csv_fallback(monkeypatch, tmp_path) -> None:
    train_csv = tmp_path / "imdb_train.csv"
    test_csv = tmp_path / "imdb_test.csv"
    train_csv.write_text("text,label\ngreat movie,1\nbad movie,0\n", encoding="utf-8")
    test_csv.write_text("text,label\nnice film,1\nawful film,0\n", encoding="utf-8")

    monkeypatch.setattr(dataset_module, "LOCAL_TRAIN_CSV", train_csv)
    monkeypatch.setattr(dataset_module, "LOCAL_TEST_CSV", test_csv)

    def _fail_for_imdb_then_use_csv(name, *args, **kwargs):
        if name == dataset_module.IMDB_DATASET_NAME:
            raise ConnectionError("network is unavailable")
        if name == "csv":
            assert "data_files" in kwargs
            return DatasetDict(
                {
                    "train": Dataset.from_dict({"text": ["great movie", "bad movie"], "label": [1, 0]}),
                    "test": Dataset.from_dict({"text": ["nice film", "awful film"], "label": [1, 0]}),
                }
            )
        raise ValueError(f"Unexpected dataset name: {name}")

    monkeypatch.setattr(dataset_module, "hf_load_dataset", _fail_for_imdb_then_use_csv)
    dataset = dataset_module.load_imdb_dataset()

    assert dataset["train"].num_rows == 2
    assert dataset["test"].num_rows == 2
    assert set(dataset["train"].column_names) >= {"text", "label"}


def test_load_imdb_dataset_raises_readable_error_when_all_sources_fail(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(dataset_module, "LOCAL_TRAIN_CSV", tmp_path / "missing_train.csv")
    monkeypatch.setattr(dataset_module, "LOCAL_TEST_CSV", tmp_path / "missing_test.csv")

    def _raise_error(*_args, **_kwargs):
        raise ConnectionError("network is unavailable")

    monkeypatch.setattr(dataset_module, "hf_load_dataset", _raise_error)

    with pytest.raises(RuntimeError, match="Failed to load IMDb dataset from both sources"):
        dataset_module.load_imdb_dataset()
