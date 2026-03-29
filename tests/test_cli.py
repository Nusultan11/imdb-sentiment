import json

from datasets import Dataset, DatasetDict
import joblib

import imdb_sentiment.pipelines.evaluation as evaluation_module
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
