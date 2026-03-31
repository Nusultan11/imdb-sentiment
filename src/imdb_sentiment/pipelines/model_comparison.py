from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
import zipfile

from imdb_sentiment.artifacts.lstm import resolve_lstm_artifact_contract
from imdb_sentiment.pipelines.evaluation import run_evaluation
from imdb_sentiment.settings import AppConfig, LSTMModelConfig, PROJECT_ROOT, load_config


LSTM_BUNDLE_ALLOWED_FILES = {
    "model.pt",
    "vocab.json",
    "training_config.json",
    "training_history.json",
    "threshold_tuning.json",
    "val_metrics.json",
    "test_metrics.json",
}
LSTM_BUNDLE_REQUIRED_FILES = {
    "model.pt",
    "vocab.json",
    "training_config.json",
}


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _require_lstm_config(config: AppConfig) -> LSTMModelConfig:
    if not isinstance(config.model, LSTMModelConfig):
        raise TypeError("LSTM bundle import expects LSTMModelConfig.")
    return config.model


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path.name}.")
    return payload


def import_lstm_bundle(config: AppConfig, bundle_path: str | Path) -> dict[str, str]:
    _require_lstm_config(config)
    artifact_contract = resolve_lstm_artifact_contract(config)
    resolved_bundle_path = Path(bundle_path)

    if not resolved_bundle_path.exists():
        raise FileNotFoundError(f"LSTM bundle does not exist: {resolved_bundle_path}")

    extracted_paths: dict[str, str] = {}
    extracted_filenames: set[str] = set()

    with zipfile.ZipFile(resolved_bundle_path) as bundle:
        for member in bundle.infolist():
            member_name = Path(member.filename).name
            if member.is_dir() or member_name not in LSTM_BUNDLE_ALLOWED_FILES:
                continue

            target_path = artifact_contract.artifact_dir / member_name
            _ensure_parent_dir(target_path)
            with bundle.open(member) as source, target_path.open("wb") as destination:
                destination.write(source.read())

            extracted_filenames.add(member_name)
            extracted_paths[member_name] = str(target_path)

    missing_files = sorted(LSTM_BUNDLE_REQUIRED_FILES - extracted_filenames)
    if missing_files:
        raise FileNotFoundError(
            "LSTM bundle is missing required files: " + ", ".join(missing_files)
        )

    return extracted_paths


def _load_lstm_runtime_details(config: AppConfig) -> dict[str, object]:
    artifact_contract = resolve_lstm_artifact_contract(config)
    training_config_payload = _read_optional_json(artifact_contract.training_config_output)
    threshold_payload = _read_optional_json(artifact_contract.threshold_tuning_output)

    serialized_model = training_config_payload.get("model") if training_config_payload else None
    if not isinstance(serialized_model, dict):
        serialized_model = {}

    return {
        "preprocessing": serialized_model.get("preprocessing", "whitespace_v1"),
        "pooling": serialized_model.get("pooling", "last_hidden"),
        "bidirectional": serialized_model.get("bidirectional", False),
        "decision_threshold": (
            threshold_payload.get("decision_threshold", 0.5)
            if threshold_payload is not None
            else 0.5
        ),
    }


def _build_comparison_row(config_path: str | Path, metrics: dict[str, float]) -> dict[str, object]:
    config = load_config(config_path)
    row: dict[str, object] = {
        "model": config.experiment.name,
        "family": config.experiment.family,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "test_metrics_output": str(config.paths.test_metrics_output),
    }

    if isinstance(config.model, LSTMModelConfig):
        row.update(_load_lstm_runtime_details(config))
    else:
        row.update(
            {
                "preprocessing": "sklearn_pipeline_internal",
                "pooling": None,
                "bidirectional": None,
                "decision_threshold": None,
            }
        )

    return row


def _default_output_dir() -> Path:
    return PROJECT_ROOT / "artifacts" / "reports" / "model_comparison"


def _write_json(path: Path, payload: object) -> None:
    _ensure_parent_dir(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    _ensure_parent_dir(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sort_key(row: dict[str, object]) -> tuple[float, float, float, float]:
    return (
        float(row["f1"]),
        float(row["accuracy"]),
        float(row["precision"]),
        float(row["recall"]),
    )


def compare_models(
    config_paths: list[str | Path],
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    results: list[dict[str, object]] = []
    missing: list[dict[str, str]] = []

    for config_path in config_paths:
        config = load_config(config_path)
        if not config.paths.model_output.exists():
            missing.append(
                {
                    "model": config.experiment.name,
                    "reason": f"missing artifact: {config.paths.model_output}",
                }
            )
            continue

        metrics = run_evaluation(config)
        results.append(_build_comparison_row(config_path, metrics))

    results.sort(key=_sort_key, reverse=True)

    resolved_output_dir = _default_output_dir() if output_dir is None else Path(output_dir)
    all_metrics_csv = resolved_output_dir / "all_models_test_metrics.csv"
    all_metrics_json = resolved_output_dir / "all_models_test_metrics.json"
    missing_models_json = resolved_output_dir / "missing_models.json"
    winner_summary_json = resolved_output_dir / "winner_summary.json"

    _write_csv(all_metrics_csv, results)
    _write_json(all_metrics_json, results)
    _write_json(missing_models_json, missing)

    winner_summary: dict[str, object] | None = None
    if results:
        winner = results[0]
        winner_summary = {
            "winner_model": winner["model"],
            "winner_family": winner["family"],
            "selection_rule": "highest_test_f1_then_accuracy_then_precision_then_recall",
            "metrics": {
                "accuracy": winner["accuracy"],
                "precision": winner["precision"],
                "recall": winner["recall"],
                "f1": winner["f1"],
            },
            "runtime_details": {
                "preprocessing": winner["preprocessing"],
                "pooling": winner["pooling"],
                "bidirectional": winner["bidirectional"],
                "decision_threshold": winner["decision_threshold"],
            },
        }
    _write_json(winner_summary_json, winner_summary)

    return {
        "results": results,
        "missing": missing,
        "winner": winner_summary,
        "outputs": {
            "all_metrics_csv": str(all_metrics_csv),
            "all_metrics_json": str(all_metrics_json),
            "missing_models_json": str(missing_models_json),
            "winner_summary_json": str(winner_summary_json),
        },
    }
