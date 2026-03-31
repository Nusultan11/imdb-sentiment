from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, DatasetDict

import imdb_sentiment.inference.predict as predict_module
import imdb_sentiment.pipelines.evaluation as evaluation_module
import imdb_sentiment.pipelines.train_lstm as train_lstm_module
from imdb_sentiment.settings import load_config


def test_lstm_regex_preprocessing_contract_flows_through_train_predict_and_evaluate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "lstm_regex_contract.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  family: lstm",
                "  name: regex_contract_test",
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
                "  pooling: last_hidden",
                "  preprocessing: regex_v2",
                "  batch_size: 2",
                "  epochs: 1",
                "  dropout: 0.2",
                "  lr: 0.01",
            ]
        ),
        encoding="utf-8",
    )

    fake_dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "text": [
                        "Amazing!!! movie",
                        "don't like it",
                        "worst-film-ever",
                        "It was <br /> good",
                        "excellent acting",
                        "terrible pacing",
                        "loved the soundtrack",
                        "boring ending",
                        "funny scenes",
                        "hated the ending",
                    ],
                    "label": [1, 0, 0, 1, 1, 0, 1, 0, 1, 0],
                }
            ),
            "test": Dataset.from_dict(
                {
                    "text": ["worst-film-ever", "Amazing!!! movie"],
                    "label": [0, 1],
                }
            ),
        }
    )
    monkeypatch.setattr(train_lstm_module, "load_imdb_dataset", lambda: fake_dataset)
    monkeypatch.setattr(evaluation_module, "load_imdb_dataset", lambda: fake_dataset)
    monkeypatch.setattr(train_lstm_module, "_train_one_epoch", lambda **kwargs: 0.1)
    monkeypatch.setattr(
        train_lstm_module,
        "_evaluate_lstm_model",
        lambda **kwargs: (
            {
                "loss": 0.2,
                "accuracy": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
            },
            0.6,
        ),
    )

    config = load_config(config_path)
    assert config.model.preprocessing == "regex_v2"

    train_lstm_module.run_lstm_training(config)

    artifact_dir = config.paths.model_output.parent
    vocab_path = artifact_dir / "vocab.json"
    training_config_path = config.paths.model_output.parent / "training_config.json"
    training_history_path = artifact_dir / "training_history.json"
    threshold_tuning_path = artifact_dir / "threshold_tuning.json"
    assert config.paths.model_output.exists()
    assert vocab_path.exists()
    assert training_config_path.exists()
    assert training_history_path.exists()
    assert threshold_tuning_path.exists()

    saved_training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
    assert saved_training_config["model"]["preprocessing"] == "regex_v2"
    saved_threshold_tuning = json.loads(threshold_tuning_path.read_text(encoding="utf-8"))
    assert saved_threshold_tuning["decision_threshold"] == 0.6
    assert saved_threshold_tuning["selection_strategy"] == "validation_best_f1"

    artifacts = predict_module.load_lstm_checkpoint(config.paths.model_output)
    assert artifacts.preprocessing == "regex_v2"
    assert artifacts.decision_threshold == 0.6

    original_encode_lstm_text = predict_module.encode_lstm_text
    predict_preprocessing_calls: list[str | None] = []

    def _recording_encode_lstm_text(*args, **kwargs):
        predict_preprocessing_calls.append(kwargs.get("preprocessing"))
        return original_encode_lstm_text(*args, **kwargs)

    monkeypatch.setattr(predict_module, "encode_lstm_text", _recording_encode_lstm_text)

    predictions = predict_module.predict_lstm_texts(artifacts, ["worst-film-ever"])

    assert len(predictions) == 1
    assert predict_preprocessing_calls == ["regex_v2"]

    original_build_lstm_dataloader = evaluation_module.build_lstm_dataloader
    evaluation_preprocessing_calls: list[str | None] = []
    original_load_restored_lstm_artifacts = evaluation_module.load_restored_lstm_artifacts
    evaluation_threshold_calls: list[float] = []

    def _recording_build_lstm_dataloader(**kwargs):
        evaluation_preprocessing_calls.append(kwargs.get("preprocessing"))
        return original_build_lstm_dataloader(**kwargs)

    def _recording_load_restored_lstm_artifacts(model_path):
        restored_artifacts = original_load_restored_lstm_artifacts(model_path)
        evaluation_threshold_calls.append(restored_artifacts.decision_threshold)
        return restored_artifacts

    monkeypatch.setattr(
        evaluation_module,
        "build_lstm_dataloader",
        _recording_build_lstm_dataloader,
    )
    monkeypatch.setattr(
        evaluation_module,
        "load_restored_lstm_artifacts",
        _recording_load_restored_lstm_artifacts,
    )

    evaluation_metrics = evaluation_module.run_evaluation(config)

    assert {"accuracy", "precision", "recall", "f1"} <= evaluation_metrics.keys()
    assert config.paths.test_metrics_output.exists()
    assert evaluation_preprocessing_calls == ["regex_v2"]
    assert evaluation_threshold_calls == [0.6]
