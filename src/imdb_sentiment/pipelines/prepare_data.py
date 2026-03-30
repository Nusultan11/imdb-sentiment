from __future__ import annotations

from pathlib import Path

from imdb_sentiment.pipelines.prepare_lstm_data import prepare_lstm_data
from imdb_sentiment.settings import AppConfig


def prepare_training_data(
    config: AppConfig,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    family = config.experiment.family

    if family == "lstm":
        return prepare_lstm_data(config, output_dir=output_dir)

    raise NotImplementedError(
        "Data preparation CLI is only implemented for Colab-first LSTM experiments right now."
    )
