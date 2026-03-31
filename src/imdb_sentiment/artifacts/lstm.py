from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from imdb_sentiment.settings import AppConfig, LSTMModelConfig


@dataclass(slots=True)
class LSTMArtifactContract:
    artifact_dir: Path
    model_output: Path
    vocab_output: Path
    training_config_output: Path
    training_history_output: Path
    threshold_tuning_output: Path
    val_metrics_output: Path
    test_metrics_output: Path


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _require_lstm_model_config(config: AppConfig) -> LSTMModelConfig:
    if not isinstance(config.model, LSTMModelConfig):
        raise TypeError("LSTM artifact contract expects LSTMModelConfig.")
    return config.model


def _read_required_json(path: Path, purpose: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"LSTM {purpose} requires artifact file: {path.name}"
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"LSTM artifact file must contain a JSON object: {path.name}")
    return payload


def resolve_lstm_artifact_contract(config: AppConfig) -> LSTMArtifactContract:
    _require_lstm_model_config(config)
    artifact_dir = config.paths.model_output.parent
    return LSTMArtifactContract(
        artifact_dir=artifact_dir,
        model_output=config.paths.model_output,
        vocab_output=artifact_dir / "vocab.json",
        training_config_output=artifact_dir / "training_config.json",
        training_history_output=artifact_dir / "training_history.json",
        threshold_tuning_output=artifact_dir / "threshold_tuning.json",
        val_metrics_output=config.paths.val_metrics_output,
        test_metrics_output=config.paths.test_metrics_output,
    )


def resolve_lstm_artifact_contract_from_model_path(model_path: str | Path) -> LSTMArtifactContract:
    resolved_model_path = Path(model_path)
    artifact_dir = resolved_model_path.parent
    return LSTMArtifactContract(
        artifact_dir=artifact_dir,
        model_output=resolved_model_path,
        vocab_output=artifact_dir / "vocab.json",
        training_config_output=artifact_dir / "training_config.json",
        training_history_output=artifact_dir / "training_history.json",
        threshold_tuning_output=artifact_dir / "threshold_tuning.json",
        val_metrics_output=artifact_dir / "val_metrics.json",
        test_metrics_output=artifact_dir / "test_metrics.json",
    )


def write_json_artifact(path: Path, payload: dict[str, object]) -> None:
    _ensure_parent_dir(path)
    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def build_lstm_training_config_payload(
    config: AppConfig,
    artifact_contract: LSTMArtifactContract,
) -> dict[str, object]:
    model_config = _require_lstm_model_config(config)
    return {
        "experiment": {
            "family": config.experiment.family,
            "name": config.experiment.name,
        },
        "seed": config.seed,
        "model": {
            "type": model_config.type,
            "vocab_size": model_config.vocab_size,
            "max_length": model_config.max_length,
            "embedding_dim": model_config.embedding_dim,
            "hidden_dim": model_config.hidden_dim,
            "batch_size": model_config.batch_size,
            "epochs": model_config.epochs,
            "dropout": model_config.dropout,
            "lr": model_config.lr,
            "bidirectional": model_config.bidirectional,
            "pooling": model_config.pooling,
        },
        "artifacts": {
            "model_output": artifact_contract.model_output.name,
            "vocab_output": artifact_contract.vocab_output.name,
            "training_config_output": artifact_contract.training_config_output.name,
            "training_history_output": artifact_contract.training_history_output.name,
            "threshold_tuning_output": artifact_contract.threshold_tuning_output.name,
            "val_metrics_output": artifact_contract.val_metrics_output.name,
            "test_metrics_output": artifact_contract.test_metrics_output.name,
        },
        "required_for_inference": [
            artifact_contract.model_output.name,
            artifact_contract.vocab_output.name,
            artifact_contract.training_config_output.name,
        ],
        "required_for_evaluation": [
            artifact_contract.model_output.name,
            artifact_contract.vocab_output.name,
            artifact_contract.training_config_output.name,
        ],
    }


def build_lstm_threshold_tuning_payload(decision_threshold: float = 0.5) -> dict[str, object]:
    return {
        "decision_threshold": decision_threshold,
        "selection_strategy": "fixed_default",
    }


def load_lstm_artifact_sidecars(
    model_path: str | Path,
) -> tuple[LSTMArtifactContract, dict[str, int], dict[str, Any]]:
    artifact_contract = resolve_lstm_artifact_contract_from_model_path(model_path)
    vocabulary_payload = _read_required_json(artifact_contract.vocab_output, "inference/evaluation")
    training_config_payload = _read_required_json(
        artifact_contract.training_config_output,
        "inference/evaluation",
    )

    expected_files = {
        "model_output": artifact_contract.model_output.name,
        "vocab_output": artifact_contract.vocab_output.name,
        "training_config_output": artifact_contract.training_config_output.name,
    }
    serialized_artifacts = training_config_payload.get("artifacts")
    if not isinstance(serialized_artifacts, dict):
        raise ValueError("training_config.json must contain an artifacts mapping.")
    for artifact_name, expected_filename in expected_files.items():
        if serialized_artifacts.get(artifact_name) != expected_filename:
            raise ValueError(
                "training_config.json does not match the expected LSTM artifact contract."
            )

    return artifact_contract, dict(vocabulary_payload), training_config_payload
