import json
from pathlib import Path

from datasets import Dataset, DatasetDict
import joblib

import imdb_sentiment.pipelines.evaluation as evaluation_module
from imdb_sentiment.models.baseline import build_baseline_model
from imdb_sentiment.settings import load_config


def test_run_evaluation_writes_test_metrics(tmp_path: Path, monkeypatch) -> None:
    model = build_baseline_model(
        max_features=100,
        ngram_range=(1, 2),
        max_iter=100,
        random_state=42,
    )
    model.fit(
        ["great movie", "excellent acting", "bad plot", "awful ending"],
        [1, 1, 0, 0],
    )

    model_path = tmp_path / "artifacts" / "models" / "baseline.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 42",
                "paths:",
                f"  model_output: {model_path.as_posix()}",
                f"  metrics_output: {(tmp_path / 'artifacts' / 'reports' / 'metrics.json').as_posix()}",
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
                    "text": ["great film", "terrible movie"],
                    "label": [1, 0],
                }
            ),
        }
    )
    monkeypatch.setattr(evaluation_module, "load_imdb_dataset", lambda: fake_dataset)

    config = load_config(config_path)
    output_path = tmp_path / "artifacts" / "reports" / "test_metrics.json"
    metrics = evaluation_module.run_evaluation(config, output_path=output_path)

    assert {"accuracy", "precision", "recall", "f1"} <= metrics.keys()
    assert output_path.exists()
    saved_metrics = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved_metrics == metrics
