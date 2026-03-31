import json
from pathlib import Path

from datasets import Dataset, DatasetDict
import joblib
import torch

import imdb_sentiment.pipelines.evaluation as evaluation_module
import imdb_sentiment.pipelines.model_comparison as comparison_module
import imdb_sentiment.pipelines.prepare_lstm_data as prepare_lstm_data_module
import imdb_sentiment.cli as cli_module
from imdb_sentiment.data.lstm import build_lstm_vocabulary
from imdb_sentiment.models.baseline import build_baseline_model
from imdb_sentiment.models.lstm.model import build_lstm_model


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

    cli_module.main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert "predictions" in payload
    assert len(payload["predictions"]) == 2
    assert all(pred in [0, 1] for pred in payload["predictions"])


def test_cli_predict_supports_lstm_checkpoints(tmp_path, monkeypatch, capsys) -> None:
    model_dir = tmp_path / "artifacts" / "models"
    reports_dir = tmp_path / "artifacts" / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    config_path = tmp_path / "lstm.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  family: lstm",
                "  name: baseline_test",
                "seed: 42",
                "",
                "paths:",
                f"  model_output: {(model_dir / 'model.pt').as_posix()}",
                f"  val_metrics_output: {(reports_dir / 'val_metrics.json').as_posix()}",
                f"  test_metrics_output: {(reports_dir / 'test_metrics.json').as_posix()}",
                "",
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

    from imdb_sentiment.settings import load_config

    config = load_config(config_path)
    vocabulary = build_lstm_vocabulary(["great movie", "awful ending"], max_size=50)
    model = build_lstm_model(config.model)
    for parameter in model.parameters():
        parameter.data.zero_()
    model.classifier.bias.data.fill_(2.0)
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

    cli_module.main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload == {"predictions": [1, 1]}


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

    cli_module.main()
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
                "  bidirectional: false",
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

    cli_module.main()
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

    cli_module.main()

    assert observed_call == {
        "model_path": model_path,
        "host": "0.0.0.0",
        "port": 9001,
    }


def test_cli_import_lstm_bundle_outputs_imported_paths(tmp_path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "lstm.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  family: lstm",
                "  name: regex_bundle_test",
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

    monkeypatch.setattr(
        cli_module,
        "import_lstm_bundle",
        lambda config, bundle_path: {"model.pt": bundle_path},
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "imdb-sentiment",
            "import-lstm-bundle",
            "--config",
            str(config_path),
            "--bundle",
            str(tmp_path / "bundle.zip"),
        ],
    )

    cli_module.main()
    payload = json.loads(capsys.readouterr().out)

    assert payload == {"model.pt": str(tmp_path / "bundle.zip")}


def test_cli_compare_models_outputs_report_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_module,
        "compare_models",
        lambda config_paths, output_dir: {
            "results": [{"model": "tfidf_best"}],
            "missing": [],
            "winner": {"winner_model": "tfidf_best"},
            "outputs": {"winner_summary_json": "reports/winner_summary.json"},
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "imdb-sentiment",
            "compare-models",
            "--config",
            "configs/experiments/tfidf_tuned_v2_final.yaml",
            "--config",
            "configs/experiments/lstm_bidirectional_masked_mean_optuna_regexprep_v1.yaml",
        ],
    )

    cli_module.main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["winner"] == {"winner_model": "tfidf_best"}


def test_cli_predict_routes_tfidf_family_to_sklearn_predict(tmp_path, monkeypatch, capsys) -> None:
    model_path = tmp_path / "artifacts" / "models" / "baseline.joblib"
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

    observed_calls: list[str] = []

    class _FakeModel:
        pass

    def _fake_load_model(path):
        observed_calls.append(f"load_model:{Path(path).name}")
        return _FakeModel()

    def _fake_predict_texts(model, texts):
        assert isinstance(model, _FakeModel)
        observed_calls.append(f"predict_texts:{len(list(texts))}")
        return [1, 0]

    monkeypatch.setattr(cli_module, "load_model", _fake_load_model)
    monkeypatch.setattr(cli_module, "predict_texts", _fake_predict_texts)
    monkeypatch.setattr(
        "sys.argv",
        [
            "imdb-sentiment",
            "predict",
            "--config",
            str(config_path),
            "--text",
            "great movie",
            "--text",
            "bad movie",
        ],
    )

    cli_module.main()
    payload = json.loads(capsys.readouterr().out)

    assert observed_calls == ["load_model:baseline.joblib", "predict_texts:2"]
    assert payload == {"predictions": [1, 0]}


def test_cli_predict_routes_lstm_family_to_torch_predict(tmp_path, monkeypatch, capsys) -> None:
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

    observed_calls: list[str] = []

    def _fake_load_lstm_checkpoint(path):
        observed_calls.append(f"load_lstm_checkpoint:{Path(path).name}")
        return "artifacts"

    def _fake_predict_lstm_texts(artifacts, texts):
        assert artifacts == "artifacts"
        observed_calls.append(f"predict_lstm_texts:{len(list(texts))}")
        return [0, 1]

    monkeypatch.setattr(cli_module, "load_lstm_checkpoint", _fake_load_lstm_checkpoint)
    monkeypatch.setattr(cli_module, "predict_lstm_texts", _fake_predict_lstm_texts)
    monkeypatch.setattr(
        "sys.argv",
        [
            "imdb-sentiment",
            "predict",
            "--config",
            str(config_path),
            "--text",
            "great movie",
            "--text",
            "bad movie",
        ],
    )

    cli_module.main()
    payload = json.loads(capsys.readouterr().out)

    assert observed_calls == ["load_lstm_checkpoint:model.pt", "predict_lstm_texts:2"]
    assert payload == {"predictions": [0, 1]}
