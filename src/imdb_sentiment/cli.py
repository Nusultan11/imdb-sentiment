from __future__ import annotations

import argparse
import json

from imdb_sentiment.inference.predict import (
    load_lstm_checkpoint,
    load_model,
    predict_lstm_texts,
    predict_texts,
)
from imdb_sentiment.pipelines.evaluation import run_evaluation
from imdb_sentiment.pipelines.model_comparison import compare_models, import_lstm_bundle
from imdb_sentiment.pipelines.prepare_data import prepare_training_data
from imdb_sentiment.pipelines.train import run_training
from imdb_sentiment.settings import AppConfig, LSTMModelConfig, load_config
from imdb_sentiment.webapp import serve_review_classifier


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

    prepare_data_parser = subparsers.add_parser(
        "prepare-data",
        help="Prepare family-specific training data artifacts",
    )
    prepare_data_parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Path to YAML config file",
    )
    prepare_data_parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional prepared data output directory override",
    )

    import_lstm_bundle_parser = subparsers.add_parser(
        "import-lstm-bundle",
        help="Import a downloaded LSTM artifact bundle into the configured experiment directory",
    )
    import_lstm_bundle_parser.add_argument(
        "--config",
        required=True,
        help="Path to the target LSTM YAML config file",
    )
    import_lstm_bundle_parser.add_argument(
        "--bundle",
        required=True,
        help="Path to the downloaded LSTM bundle zip file",
    )

    compare_models_parser = subparsers.add_parser(
        "compare-models",
        help="Evaluate multiple saved experiments on the IMDb test split and write a comparison report",
    )
    compare_models_parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="Path to a YAML config file. Repeat --config to compare multiple models.",
    )
    compare_models_parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional comparison report directory override",
    )

    web_parser = subparsers.add_parser(
        "serve-web",
        help="Start a local website for review classification",
    )
    web_parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Path to YAML config file",
    )
    web_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for the local web server",
    )
    web_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the local web server",
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
    family = config.experiment.family

    if family == "tfidf":
        model = load_model(config.paths.model_output)
        predictions = predict_texts(model, texts)
    elif family == "lstm":
        if not isinstance(config.model, LSTMModelConfig):
            raise TypeError("CLI LSTM predict expects LSTMModelConfig.")
        artifacts = load_lstm_checkpoint(config.paths.model_output)
        predictions = predict_lstm_texts(artifacts, texts)
    else:
        raise NotImplementedError(
            "Predict CLI is not implemented for this experiment family yet."
        )

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


def _run_prepare_data_command(config_path: str, output_dir: str | None) -> None:
    config = load_config(config_path)
    prepared_paths = prepare_training_data(config, output_dir=output_dir)
    serialized_paths = {
        name: str(path)
        for name, path in prepared_paths.items()
    }
    print(json.dumps(serialized_paths, ensure_ascii=False))


def _require_lstm_cli_config(config: AppConfig) -> None:
    if not isinstance(config.model, LSTMModelConfig):
        raise TypeError("CLI LSTM bundle import expects LSTMModelConfig.")


def _run_import_lstm_bundle_command(config_path: str, bundle_path: str) -> None:
    config = load_config(config_path)
    _require_lstm_cli_config(config)
    imported_paths = import_lstm_bundle(config, bundle_path)
    print(json.dumps(imported_paths, ensure_ascii=False))


def _run_compare_models_command(config_paths: list[str], output_dir: str | None) -> None:
    report = compare_models(config_paths, output_dir=output_dir)
    print(json.dumps(report, ensure_ascii=False))


def _run_serve_web_command(config_path: str, host: str, port: int) -> None:
    config = load_config(config_path)
    serve_review_classifier(config.paths.model_output, host=host, port=port)


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

    if args.command == "prepare-data":
        _run_prepare_data_command(args.config, args.output_dir)
        return

    if args.command == "import-lstm-bundle":
        _run_import_lstm_bundle_command(args.config, args.bundle)
        return

    if args.command == "compare-models":
        _run_compare_models_command(args.config, args.output_dir)
        return

    if args.command == "serve-web":
        _run_serve_web_command(args.config, args.host, args.port)
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
