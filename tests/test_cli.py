import json
from pathlib import Path

from datasets import Dataset, DatasetDict
import joblib

import imdb_sentiment.pipelines.evaluation as evaluation_module
import imdb_sentiment.pipelines.prepare_lstm_data as prepare_lstm_data_module
from imdb_sentiment.cli import main
from imdb_sentiment.models.baseline import build_baseline_model


def test_cli_predict_outputs_json_predictions(tmp_path, monkeypatch, capsys) -> None:
    model = build_baseline_model(
        max_features=100,
        ngram_range=(1, 2),
        max_iter=100,
        random_state=42,
    )
    train_texts = ["excellent movie", "wonderful acting", "boring movie", "terrible ending"]
    train_labels = [1, 1, 0, 0]
    model.fit(train_texts, train_labels)

    model_dir = tmp_path / "artifacts" / "models"
    reports_dir = tmp_path / "artifacts" / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "baseline.joblib"
    joblib.dump(model, model_path)

    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  family: tfidf",
                "  name: baseline_test",
                "seed: 42",
                "",
                "paths:",
                f"  model_output: {model_path.as_posix()}",
                f"  val_metrics_output: {(reports_dir / 'val_metrics.json').as_posix()}",
                f"  test_metrics_output: {(reports_dir / 'test_metrics.json').as_posix()}",
                "",
                "model:",
                "  type: logistic_regression",
                "  max_features: 100",
                "  ngram_range: [1, 2]",
                "  max_iter: 100",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "imdb-sentiment",
            "predict",
            "--config",
            str(config_path),
            "--text",
            "I absolutely loved this movie.",
            "--text",
            "This was dull and disappointing.",
        ],
    )

    main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert "predictions" in payload
    assert len(payload["predictions"]) == 2
    assert all(pred in [0, 1] for pred in payload["predictions"])


def test_cli_evaluate_writes_test_metrics(tmp_path, monkeypatch, capsys) -> None:
    model = build_baseline_model(
        max_features=100,
        ngram_range=(1, 2),
        max_iter=100,
        random_state=42,
    )
    model.fit(
        ["excellent movie", "wonderful acting", "boring movie", "terrible ending"],
        [1, 1, 0, 0],
    )

    model_dir = tmp_path / "artifacts" / "models"
    reports_dir = tmp_path / "artifacts" / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "baseline.joblib"
    joblib.dump(model, model_path)
    test_metrics_output = reports_dir / "test_metrics.json"

    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  family: tfidf",
                "  name: baseline_test",
                "seed: 42",
                "",
                "paths:",
                f"  model_output: {model_path.as_posix()}",
                f"  val_metrics_output: {(reports_dir / 'val_metrics.json').as_posix()}",
                f"  test_metrics_output: {test_metrics_output.as_posix()}",
                "",
                "model:",
                "  type: logistic_regression",
                "  max_features: 100",
                "  ngram_range: [1, 2]",
                "  max_iter: 100",
            ]
        ),
        encoding="utf-8",
    )

    fake_dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"text": ["placeholder"], "label": [1]}),
            "test": Dataset.from_dict(
                {
                    "text": ["excellent film", "terrible movie"],
                    "label": [1, 0],
                }
            ),
        }
    )
    monkeypatch.setattr(evaluation_module, "load_imdb_dataset", lambda: fake_dataset)
    monkeypatch.setattr(
        "sys.argv",
        [
            "imdb-sentiment",
            "evaluate",
            "--config",
            str(config_path),
        ],
    )

    main()
    captured = capsys.readouterr()

    assert "Evaluation finished." in captured.out
    assert f"Test metrics saved to: {test_metrics_output}" in captured.out
    assert test_metrics_output.exists()


def test_cli_prepare_data_outputs_lstm_split_paths(tmp_path, monkeypatch, capsys) -> None:
    output_dir = tmp_path / "prepared"
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
                "  vocab_size: 20000",
                "  max_length: 64",
                "  embedding_dim: 128",
                "  hidden_dim: 128",
                "  batch_size: 32",
                "  epochs: 5",
                "  dropout: 0.3",
                "  lr: 0.001",
            ]
        ),
        encoding="utf-8",
    )

    fake_dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "text": [
                        "Great movie",
                        "Excellent plot",
                        "Amazing acting",
                        "Loved every scene",
                        "Bad acting",
                        "Terrible ending",
                        "Boring script",
                        "Hated every minute",
                        "Wonderful soundtrack",
                        "Awful pacing",
                    ],
                    "label": [1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
                }
            ),
            "test": Dataset.from_dict(
                {
                    "text": ["Wonderful film", "Awful film"],
                    "label": [1, 0],
                }
            ),
        }
    )
    monkeypatch.setattr(prepare_lstm_data_module, "load_imdb_dataset", lambda: fake_dataset)
    monkeypatch.setattr(
        "sys.argv",
        [
            "imdb-sentiment",
            "prepare-data",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert set(payload) == {"train_path", "val_path", "test_path", "metadata_path"}
    assert Path(payload["train_path"]).exists()
    assert Path(payload["val_path"]).exists()
    assert Path(payload["test_path"]).exists()
    assert Path(payload["metadata_path"]).exists()


def test_cli_serve_web_uses_config_model_path(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "artifacts" / "models" / "baseline.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("placeholder", encoding="utf-8")

    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  family: tfidf",
                "  name: baseline_test",
                "seed: 42",
                "paths:",
                f"  model_output: {model_path.as_posix()}",
                f"  val_metrics_output: {(tmp_path / 'artifacts' / 'reports' / 'val_metrics.json').as_posix()}",
                f"  test_metrics_output: {(tmp_path / 'artifacts' / 'reports' / 'test_metrics.json').as_posix()}",
                "model:",
                "  type: logistic_regression",
                "  max_features: 100",
                "  ngram_range: [1, 2]",
                "  max_iter: 100",
            ]
        ),
        encoding="utf-8",
    )

    observed_call: dict[str, object] = {}

    def _fake_serve_review_classifier(model_path_arg, host, port) -> None:
        observed_call["model_path"] = model_path_arg
        observed_call["host"] = host
        observed_call["port"] = port

    monkeypatch.setattr("imdb_sentiment.cli.serve_review_classifier", _fake_serve_review_classifier)
    monkeypatch.setattr(
        "sys.argv",
        [
            "imdb-sentiment",
            "serve-web",
            "--config",
            str(config_path),
            "--host",
            "0.0.0.0",
            "--port",
            "9001",
        ],
    )

    main()

    assert observed_call == {
        "model_path": model_path,
        "host": "0.0.0.0",
        "port": 9001,
    }
