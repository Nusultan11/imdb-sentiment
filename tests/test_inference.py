import json
import joblib
import pytest
import torch
from sklearn.exceptions import InconsistentVersionWarning

from imdb_sentiment.data.lstm import build_lstm_vocabulary
from imdb_sentiment.inference.predict import (
    load_lstm_checkpoint,
    load_model,
    predict_from_model_path,
    predict_lstm_texts,
    predict_texts,
)
from imdb_sentiment.models.baseline import build_baseline_model
from imdb_sentiment.models.lstm.model import build_lstm_model
from imdb_sentiment.settings import load_config


def test_inference_returns_binary_predictions(tmp_path) -> None:
    model = build_baseline_model(
        max_features=100,
        ngram_range=(1, 2),
        max_iter=100,
        random_state=42,
    )
    train_texts = ["amazing movie", "great acting", "bad movie", "awful plot"]
    train_labels = [1, 1, 0, 0]
    model.fit(train_texts, train_labels)

    model_path = tmp_path / "baseline.joblib"
    joblib.dump(model, model_path)
    loaded_model = load_model(model_path)

    texts = [
        "This movie was amazing, I loved it.",
        "Terrible film. Waste of time.",
    ]

    preds = predict_texts(loaded_model, texts)

    assert isinstance(preds, list)
    assert len(preds) == 2
    assert all(pred in [0, 1] for pred in preds)


def test_load_model_raises_runtime_error_on_sklearn_version_mismatch(monkeypatch, tmp_path) -> None:
    model_path = tmp_path / "baseline.joblib"
    model_path.write_text("placeholder", encoding="utf-8")

    def _load_with_warning(_path):
        import warnings

        warnings.warn(
            InconsistentVersionWarning(
                estimator_name="Pipeline",
                current_sklearn_version="1.7.1",
                original_sklearn_version="1.6.1",
            )
        )
        return object()

    monkeypatch.setattr("imdb_sentiment.inference.predict.joblib.load", _load_with_warning)

    with pytest.raises(RuntimeError, match="scikit-learn versions do not match"):
        load_model(model_path)


def test_predict_from_model_path_returns_binary_predictions(tmp_path) -> None:
    model = build_baseline_model(
        max_features=100,
        ngram_range=(1, 2),
        max_iter=100,
        random_state=42,
    )
    train_texts = ["excellent film", "loved the acting", "boring movie", "hated the ending"]
    train_labels = [1, 1, 0, 0]
    model.fit(train_texts, train_labels)

    model_path = tmp_path / "baseline.joblib"
    joblib.dump(model, model_path)

    preds = predict_from_model_path(
        model_path,
        ["What a wonderful movie.", "This was a terrible waste of time."],
    )

    assert isinstance(preds, list)
    assert len(preds) == 2
    assert all(pred in [0, 1] for pred in preds)


def test_predict_texts_accepts_generator_input(tmp_path) -> None:
    model = build_baseline_model(
        max_features=100,
        ngram_range=(1, 2),
        max_iter=100,
        random_state=42,
    )
    train_texts = ["excellent movie", "great acting", "boring movie", "terrible acting"]
    train_labels = [1, 1, 0, 0]
    model.fit(train_texts, train_labels)

    text_stream = (
        text
        for text in [
            "excellent story",
            "terrible story",
        ]
    )

    preds = predict_texts(model, text_stream)

    assert preds == [1, 0]
    assert all(isinstance(pred, int) for pred in preds)


def test_load_lstm_checkpoint_restores_vocabulary_and_max_length(tmp_path) -> None:
    config_path = tmp_path / "lstm.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  family: lstm",
                "  name: baseline_test",
                "seed: 42",
                "paths:",
                f"  model_output: {(tmp_path / 'artifacts' / 'models' / 'model.pt').as_posix()}",
                f"  val_metrics_output: {(tmp_path / 'artifacts' / 'reports' / 'val_metrics.json').as_posix()}",
                f"  test_metrics_output: {(tmp_path / 'artifacts' / 'reports' / 'test_metrics.json').as_posix()}",
                "model:",
                "  type: lstm",
                "  vocab_size: 50",
                "  max_length: 6",
                "  embedding_dim: 8",
                "  hidden_dim: 6",
                "  bidirectional: false",
                "  batch_size: 2",
                "  epochs: 2",
                "  dropout: 0.2",
                "  lr: 0.01",
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    vocabulary = build_lstm_vocabulary(["great movie", "awful ending"], max_size=50)
    model = build_lstm_model(config.model)
    config.paths.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocabulary": vocabulary.token_to_id,
            "max_length": config.model.max_length,
            "family": config.experiment.family,
            "name": config.experiment.name,
        },
        config.paths.model_output,
    )
    (config.paths.model_output.parent / "vocab.json").write_text(
        json.dumps(vocabulary.token_to_id, indent=2),
        encoding="utf-8",
    )
    (config.paths.model_output.parent / "training_config.json").write_text(
        json.dumps(
            {
                "experiment": {
                    "family": config.experiment.family,
                    "name": config.experiment.name,
                },
                "seed": config.seed,
                "model": {
                    "type": "lstm",
                    "vocab_size": config.model.vocab_size,
                    "max_length": config.model.max_length,
                    "embedding_dim": config.model.embedding_dim,
                    "hidden_dim": config.model.hidden_dim,
                    "batch_size": config.model.batch_size,
                    "epochs": config.model.epochs,
                    "dropout": config.model.dropout,
                    "lr": config.model.lr,
                },
                "artifacts": {
                    "model_output": "model.pt",
                    "vocab_output": "vocab.json",
                    "training_config_output": "training_config.json",
                    "val_metrics_output": "val_metrics.json",
                    "test_metrics_output": "test_metrics.json",
                },
                "required_for_inference": [
                    "model.pt",
                    "vocab.json",
                    "training_config.json",
                ],
                "required_for_evaluation": [
                    "model.pt",
                    "vocab.json",
                    "training_config.json",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    artifacts = load_lstm_checkpoint(config.paths.model_output, config.model)

    assert artifacts.max_length == 6
    assert artifacts.vocabulary.pad_id == 0
    assert artifacts.vocabulary.unk_id == 1
    assert artifacts.vocabulary.token_to_id["great"] == vocabulary.token_to_id["great"]
    assert artifacts.model.embedding.num_embeddings == config.model.vocab_size
    assert artifacts.model.encoder.hidden_size == config.model.hidden_dim
    assert artifacts.model.training is False


def test_predict_from_model_path_supports_lstm_checkpoints(tmp_path) -> None:
    config_path = tmp_path / "lstm.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  family: lstm",
                "  name: baseline_test",
                "seed: 42",
                "paths:",
                f"  model_output: {(tmp_path / 'artifacts' / 'models' / 'model.pt').as_posix()}",
                f"  val_metrics_output: {(tmp_path / 'artifacts' / 'reports' / 'val_metrics.json').as_posix()}",
                f"  test_metrics_output: {(tmp_path / 'artifacts' / 'reports' / 'test_metrics.json').as_posix()}",
                "model:",
                "  type: lstm",
                "  vocab_size: 50",
                "  max_length: 6",
                "  embedding_dim: 8",
                "  hidden_dim: 6",
                "  bidirectional: false",
                "  batch_size: 2",
                "  epochs: 2",
                "  dropout: 0.2",
                "  lr: 0.01",
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    vocabulary = build_lstm_vocabulary(["great movie", "awful ending"], max_size=50)
    model = build_lstm_model(config.model)
    for parameter in model.parameters():
        parameter.data.zero_()
    model.classifier.bias.data.fill_(2.0)

    config.paths.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocabulary": vocabulary.token_to_id,
            "max_length": config.model.max_length,
            "family": config.experiment.family,
            "name": config.experiment.name,
        },
        config.paths.model_output,
    )
    (config.paths.model_output.parent / "vocab.json").write_text(
        json.dumps(vocabulary.token_to_id, indent=2),
        encoding="utf-8",
    )
    (config.paths.model_output.parent / "training_config.json").write_text(
        json.dumps(
            {
                "experiment": {
                    "family": config.experiment.family,
                    "name": config.experiment.name,
                },
                "seed": config.seed,
                "model": {
                    "type": "lstm",
                    "vocab_size": config.model.vocab_size,
                    "max_length": config.model.max_length,
                    "embedding_dim": config.model.embedding_dim,
                    "hidden_dim": config.model.hidden_dim,
                    "batch_size": config.model.batch_size,
                    "epochs": config.model.epochs,
                    "dropout": config.model.dropout,
                    "lr": config.model.lr,
                },
                "artifacts": {
                    "model_output": "model.pt",
                    "vocab_output": "vocab.json",
                    "training_config_output": "training_config.json",
                    "val_metrics_output": "val_metrics.json",
                    "test_metrics_output": "test_metrics.json",
                },
                "required_for_inference": [
                    "model.pt",
                    "vocab.json",
                    "training_config.json",
                ],
                "required_for_evaluation": [
                    "model.pt",
                    "vocab.json",
                    "training_config.json",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    artifacts = load_lstm_checkpoint(config.paths.model_output, config.model)
    direct_predictions = predict_lstm_texts(
        artifacts,
        ["great movie", "awful ending"],
    )
    path_predictions = predict_from_model_path(
        config.paths.model_output,
        ["great movie", "awful ending"],
        config=config,
    )

    assert direct_predictions == [1, 1]
    assert path_predictions == [1, 1]
    assert all(isinstance(prediction, int) for prediction in path_predictions)
