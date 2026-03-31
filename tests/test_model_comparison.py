from __future__ import annotations

import json
from pathlib import Path
import zipfile

import imdb_sentiment.pipelines.model_comparison as comparison_module
from imdb_sentiment.settings import load_config


def test_import_lstm_bundle_extracts_runtime_files(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts" / "experiments" / "lstm" / "regex"
    config_path = tmp_path / "regex.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  family: lstm",
                "  name: regex_bundle_test",
                "seed: 42",
                "paths:",
                f"  model_output: {(artifact_dir / 'model.pt').as_posix()}",
                f"  val_metrics_output: {(artifact_dir / 'val_metrics.json').as_posix()}",
                f"  test_metrics_output: {(artifact_dir / 'test_metrics.json').as_posix()}",
                "model:",
                "  type: lstm",
                "  vocab_size: 50",
                "  max_length: 6",
                "  embedding_dim: 8",
                "  hidden_dim: 6",
                "  bidirectional: true",
                "  pooling: masked_mean",
                "  preprocessing: regex_v2",
                "  batch_size: 2",
                "  epochs: 2",
                "  dropout: 0.2",
                "  lr: 0.01",
            ]
        ),
        encoding="utf-8",
    )

    bundle_path = tmp_path / "regex_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w") as bundle:
        bundle.writestr("model.pt", b"checkpoint")
        bundle.writestr("vocab.json", json.dumps({"<pad>": 0, "<unk>": 1}))
        bundle.writestr(
            "training_config.json",
            json.dumps(
                {
                    "experiment": {"family": "lstm", "name": "regex_bundle_test"},
                    "model": {"preprocessing": "regex_v2"},
                }
            ),
        )
        bundle.writestr("notes.txt", "ignored")

    imported_paths = comparison_module.import_lstm_bundle(load_config(config_path), bundle_path)

    assert set(imported_paths) == {"model.pt", "vocab.json", "training_config.json"}
    assert (artifact_dir / "model.pt").exists()
    assert (artifact_dir / "vocab.json").exists()
    assert (artifact_dir / "training_config.json").exists()
    assert not (artifact_dir / "notes.txt").exists()


def test_compare_models_writes_report_and_picks_best_winner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tfidf_config_path = tmp_path / "tfidf.yaml"
    tfidf_config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  family: tfidf",
                "  name: tfidf_best",
                "seed: 42",
                "paths:",
                f"  model_output: {(tmp_path / 'artifacts' / 'tfidf.joblib').as_posix()}",
                f"  val_metrics_output: {(tmp_path / 'artifacts' / 'tfidf_val.json').as_posix()}",
                f"  test_metrics_output: {(tmp_path / 'artifacts' / 'tfidf_test.json').as_posix()}",
                "model:",
                "  type: logistic_regression",
                "  max_features: 100",
                "  ngram_range: [1, 2]",
                "  max_iter: 100",
            ]
        ),
        encoding="utf-8",
    )
    lstm_artifact_dir = tmp_path / "artifacts" / "lstm"
    lstm_config_path = tmp_path / "lstm.yaml"
    lstm_config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  family: lstm",
                "  name: regex_bilstm",
                "seed: 42",
                "paths:",
                f"  model_output: {(lstm_artifact_dir / 'model.pt').as_posix()}",
                f"  val_metrics_output: {(lstm_artifact_dir / 'val_metrics.json').as_posix()}",
                f"  test_metrics_output: {(lstm_artifact_dir / 'test_metrics.json').as_posix()}",
                "model:",
                "  type: lstm",
                "  vocab_size: 50",
                "  max_length: 6",
                "  embedding_dim: 8",
                "  hidden_dim: 6",
                "  bidirectional: true",
                "  pooling: masked_mean",
                "  preprocessing: regex_v2",
                "  batch_size: 2",
                "  epochs: 2",
                "  dropout: 0.2",
                "  lr: 0.01",
            ]
        ),
        encoding="utf-8",
    )

    load_config(tfidf_config_path).paths.model_output.parent.mkdir(parents=True, exist_ok=True)
    load_config(tfidf_config_path).paths.model_output.write_text("placeholder", encoding="utf-8")
    lstm_artifact_dir.mkdir(parents=True, exist_ok=True)
    (lstm_artifact_dir / "model.pt").write_text("placeholder", encoding="utf-8")
    (lstm_artifact_dir / "training_config.json").write_text(
        json.dumps(
            {
                "model": {
                    "preprocessing": "regex_v2",
                    "pooling": "masked_mean",
                    "bidirectional": True,
                }
            }
        ),
        encoding="utf-8",
    )
    (lstm_artifact_dir / "threshold_tuning.json").write_text(
        json.dumps({"decision_threshold": 0.86}),
        encoding="utf-8",
    )

    def _fake_run_evaluation(config, output_path=None):
        metrics = {
            "tfidf_best": {"accuracy": 0.90, "precision": 0.89, "recall": 0.91, "f1": 0.90},
            "regex_bilstm": {"accuracy": 0.88, "precision": 0.88, "recall": 0.87, "f1": 0.875},
        }[config.experiment.name]
        resolved_output = config.paths.test_metrics_output if output_path is None else Path(output_path)
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        resolved_output.write_text(json.dumps(metrics), encoding="utf-8")
        return metrics

    monkeypatch.setattr(comparison_module, "run_evaluation", _fake_run_evaluation)

    report = comparison_module.compare_models(
        [tfidf_config_path, lstm_config_path],
        output_dir=tmp_path / "reports",
    )

    assert report["winner"] == {
        "winner_model": "tfidf_best",
        "winner_family": "tfidf",
        "selection_rule": "highest_test_f1_then_accuracy_then_precision_then_recall",
        "metrics": {
            "accuracy": 0.9,
            "precision": 0.89,
            "recall": 0.91,
            "f1": 0.9,
        },
        "runtime_details": {
            "preprocessing": "sklearn_pipeline_internal",
            "pooling": None,
            "bidirectional": None,
            "decision_threshold": None,
        },
    }
    assert (tmp_path / "reports" / "all_models_test_metrics.csv").exists()
    assert (tmp_path / "reports" / "all_models_test_metrics.json").exists()
    assert (tmp_path / "reports" / "winner_summary.json").exists()
