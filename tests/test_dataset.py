import pytest

import imdb_sentiment.data.dataset as dataset_module


def test_load_imdb_dataset_raises_readable_error_on_failure(monkeypatch) -> None:
    def _raise_error(*_args, **_kwargs):
        raise ConnectionError("network is unavailable")

    monkeypatch.setattr(dataset_module, "hf_load_dataset", _raise_error)

    with pytest.raises(RuntimeError, match="Failed to load IMDb dataset from Hugging Face"):
        dataset_module.load_imdb_dataset()
