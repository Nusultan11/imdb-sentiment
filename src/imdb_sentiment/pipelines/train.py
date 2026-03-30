from __future__ import annotations

from imdb_sentiment.pipelines.train_tfidf import run_tfidf_training
from imdb_sentiment.settings import AppConfig


def run_training(config: AppConfig) -> dict[str, float]:
    family = config.experiment.family

    if family == "tfidf":
        return run_tfidf_training(config)

    if family == "lstm":
        raise NotImplementedError(
            "LSTM training is expected to run in Colab. Export train/val/test data locally first."
        )

    if family == "transformer":
        raise NotImplementedError(
            "Transformer training runner is not implemented yet. Add a family-specific trainer "
            "before running transformer experiments."
        )

    raise ValueError(f"Unsupported experiment.family: {family}")
