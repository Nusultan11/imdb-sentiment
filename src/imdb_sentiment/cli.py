from __future__ import annotations

import argparse
import json

from imdb_sentiment.inference.predict import predict_from_model_path
from imdb_sentiment.pipelines.evaluation import run_evaluation
from imdb_sentiment.pipelines.train import run_training
from imdb_sentiment.settings import load_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train IMDb sentiment baseline model")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train the baseline model")
    train_parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Path to YAML config file",
    )

    predict_parser = subparsers.add_parser("predict", help="Run inference with a saved model")
    predict_parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Path to YAML config file",
    )
    predict_parser.add_argument(
        "--text",
        action="append",
        required=True,
        help="Input text for sentiment prediction. Repeat --text to score multiple reviews.",
    )

    evaluate_parser = subparsers.add_parser("evaluate", help="Score a saved model on the IMDb test split")
    evaluate_parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Path to YAML config file",
    )
    evaluate_parser.add_argument(
        "--output",
        default=None,
        help="Optional metrics output path override",
    )

    return parser


def _run_train_command(config_path: str) -> None:
    config = load_config(config_path)
    metrics = run_training(config)

    print("Training finished.")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Model saved to: {config.paths.model_output}")
    print(f"Validation metrics saved to: {config.paths.val_metrics_output}")


def _run_predict_command(config_path: str, texts: list[str]) -> None:
    config = load_config(config_path)
    predictions = predict_from_model_path(config.paths.model_output, texts)
    print(json.dumps({"predictions": predictions}, ensure_ascii=False))


def _run_evaluate_command(config_path: str, output_path: str | None) -> None:
    config = load_config(config_path)
    metrics = run_evaluation(config, output_path=output_path)
    resolved_output_path = config.paths.test_metrics_output if output_path is None else output_path

    print("Evaluation finished.")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Test metrics saved to: {resolved_output_path}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command in (None, "train"):
        config_path = getattr(args, "config", "configs/baseline.yaml")
        _run_train_command(config_path)
        return

    if args.command == "predict":
        _run_predict_command(args.config, args.text)
        return

    if args.command == "evaluate":
        _run_evaluate_command(args.config, args.output)
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
