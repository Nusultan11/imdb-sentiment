from __future__ import annotations

import json
from pathlib import Path

import pytest

from imdb_sentiment.artifacts.lstm_runtime import (
    load_lstm_decision_threshold,
    load_lstm_model_config_from_training_payload,
    load_restored_lstm_artifacts,
)


def _write_lstm_runtime_bundle(
    artifact_dir: Path,
    *,
    preprocessing: str = "regex_v2",
    decision_threshold: float = 0.9,
) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.pt"
    model_path.write_bytes(b"placeholder checkpoint")

    vocabulary_payload = {
        "<pad>": 0,
        "<unk>": 1,
        "worst": 2,
        "film": 3,
        "ever": 4,
    }
    (artifact_dir / "vocab.json").write_text(
        json.dumps(vocabulary_payload, indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "training_config.json").write_text(
        json.dumps(
            {
                "experiment": {
                    "family": "lstm",
                    "name": "runtime_loader_test",
                },
                "seed": 42,
                "model": {
                    "type": "lstm",
                    "vocab_size": 50,
                    "max_length": 6,
                    "embedding_dim": 8,
                    "hidden_dim": 6,
                    "batch_size": 2,
                    "epochs": 2,
                    "dropout": 0.2,
                    "lr": 0.01,
                    "bidirectional": True,
                    "pooling": "masked_mean",
                    "preprocessing": preprocessing,
                },
                "artifacts": {
                    "model_output": "model.pt",
                    "vocab_output": "vocab.json",
                    "training_config_output": "training_config.json",
                    "threshold_tuning_output": "threshold_tuning.json",
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
    (artifact_dir / "threshold_tuning.json").write_text(
        json.dumps(
            {
                "decision_threshold": decision_threshold,
                "selection_strategy": "validation_best_f1",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return model_path


def test_load_restored_lstm_artifacts_reads_saved_config_and_threshold(tmp_path: Path) -> None:
    model_path = _write_lstm_runtime_bundle(
        tmp_path / "artifacts" / "models",
        preprocessing="regex_v2",
        decision_threshold=0.9,
    )

    restored_artifacts = load_restored_lstm_artifacts(model_path)

    assert restored_artifacts.vocabulary.pad_id == 0
    assert restored_artifacts.vocabulary.unk_id == 1
    assert restored_artifacts.vocabulary.token_to_id["worst"] == 2
    assert restored_artifacts.model_config.max_length == 6
    assert restored_artifacts.model_config.bidirectional is True
    assert restored_artifacts.model_config.pooling == "masked_mean"
    assert restored_artifacts.model_config.preprocessing == "regex_v2"
    assert restored_artifacts.decision_threshold == 0.9


def test_load_lstm_model_config_from_training_payload_fills_legacy_defaults() -> None:
    model_config = load_lstm_model_config_from_training_payload(
        {
            "model": {
                "type": "lstm",
                "vocab_size": 50,
                "max_length": 6,
                "embedding_dim": 8,
                "hidden_dim": 6,
                "batch_size": 2,
                "epochs": 2,
                "dropout": 0.2,
                "lr": 0.01,
            }
        }
    )

    assert model_config.bidirectional is False
    assert model_config.pooling == "last_hidden"
    assert model_config.preprocessing == "whitespace_v1"


def test_load_lstm_decision_threshold_defaults_to_point_five_when_sidecar_missing(
    tmp_path: Path,
) -> None:
    threshold = load_lstm_decision_threshold(tmp_path / "missing_threshold_tuning.json")

    assert threshold == 0.5


def test_load_lstm_decision_threshold_rejects_non_numeric_value(tmp_path: Path) -> None:
    threshold_path = tmp_path / "threshold_tuning.json"
    threshold_path.write_text(
        json.dumps(
            {
                "decision_threshold": "high",
                "selection_strategy": "validation_best_f1",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="numeric decision_threshold"):
        load_lstm_decision_threshold(threshold_path)
