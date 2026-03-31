from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from imdb_sentiment.artifacts.lstm import load_lstm_artifact_sidecars
from imdb_sentiment.data.lstm import LSTMVocabulary
from imdb_sentiment.settings import LSTMModelConfig, _load_lstm_model_config


@dataclass(slots=True)
class RestoredLSTMArtifacts:
    vocabulary: LSTMVocabulary
    model_config: LSTMModelConfig
    decision_threshold: float


def load_lstm_model_config_from_training_payload(
    training_config_payload: dict[str, object],
) -> LSTMModelConfig:
    serialized_model = training_config_payload.get("model")
    if not isinstance(serialized_model, dict):
        raise ValueError("training_config.json must contain a model mapping.")

    model_payload = dict(serialized_model)
    model_payload.setdefault("bidirectional", False)
    model_payload.setdefault("pooling", "last_hidden")
    model_payload.setdefault("preprocessing", "whitespace_v1")
    return _load_lstm_model_config(model_payload)


def load_lstm_decision_threshold(threshold_tuning_path: Path) -> float:
    if not threshold_tuning_path.exists():
        return 0.5

    payload = json.loads(threshold_tuning_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("threshold_tuning.json must contain a JSON object.")

    decision_threshold = payload.get("decision_threshold", 0.5)
    if not isinstance(decision_threshold, (int, float)) or isinstance(decision_threshold, bool):
        raise ValueError("threshold_tuning.json must contain a numeric decision_threshold.")

    resolved_threshold = float(decision_threshold)
    if resolved_threshold < 0 or resolved_threshold > 1:
        raise ValueError("decision_threshold must be in the range [0, 1].")

    return resolved_threshold


def load_restored_lstm_artifacts(model_path: str | Path) -> RestoredLSTMArtifacts:
    artifact_contract, vocabulary_payload, training_config_payload = load_lstm_artifact_sidecars(model_path)
    return RestoredLSTMArtifacts(
        vocabulary=LSTMVocabulary(token_to_id=vocabulary_payload),
        model_config=load_lstm_model_config_from_training_payload(training_config_payload),
        decision_threshold=load_lstm_decision_threshold(artifact_contract.threshold_tuning_output),
    )
