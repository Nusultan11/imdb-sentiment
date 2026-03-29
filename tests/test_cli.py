import json

import joblib

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
                "seed: 42",
                "",
                "paths:",
                f"  model_output: {model_path.as_posix()}",
                f"  metrics_output: {(reports_dir / 'metrics.json').as_posix()}",
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
