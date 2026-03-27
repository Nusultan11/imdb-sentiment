from __future__ import annotations

import argparse

from imdb_sentiment.pipelines.train import run_training
from imdb_sentiment.settings import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IMDb sentiment baseline model")
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    metrics = run_training(config)

    print("Training finished.")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Model saved to: {config.paths.model_output}")
    print(f"Metrics saved to: {config.paths.metrics_output}")


if __name__ == "__main__":
    main()