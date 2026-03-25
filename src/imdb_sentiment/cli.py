from __future__ import annotations

import argparse
import logging
from typing import Sequence

from imdb_sentiment.pipelines.train import run_training
from imdb_sentiment.settings import load_config

LOGGER = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train IMDb sentiment baseline model.")
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Path to YAML config file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        metrics = run_training(config)
    except Exception:
        LOGGER.exception("Training failed")
        return 1

    LOGGER.info("Training finished")
    LOGGER.info("Accuracy: %.4f", metrics["accuracy"])
    LOGGER.info(
        "Precision: %.4f | Recall: %.4f | F1: %.4f",
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )
    LOGGER.info("Model saved to: %s", config.paths.model_output)
    LOGGER.info("Metrics saved to: %s", config.paths.metrics_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
