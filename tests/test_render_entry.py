from __future__ import annotations

from pathlib import Path

import pytest

import imdb_sentiment.render_entry as render_entry_module


def test_render_entry_uses_env_config_host_and_port(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  family: tfidf",
                "  name: baseline_test",
                "seed: 42",
                "paths:",
                f"  model_output: {(tmp_path / 'artifacts' / 'models' / 'baseline.joblib').as_posix()}",
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

    def _fake_serve_review_classifier(model_path, host, port) -> None:
        observed_call["model_path"] = model_path
        observed_call["host"] = host
        observed_call["port"] = port

    monkeypatch.setenv("IMDB_CONFIG", str(config_path))
    monkeypatch.setenv("RENDER_HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "11000")
    monkeypatch.setattr(render_entry_module, "serve_review_classifier", _fake_serve_review_classifier)

    render_entry_module.main()

    assert observed_call == {
        "model_path": tmp_path / "artifacts" / "models" / "baseline.joblib",
        "host": "0.0.0.0",
        "port": 11000,
    }


def test_render_entry_rejects_non_integer_port(monkeypatch) -> None:
    monkeypatch.setenv("PORT", "not-a-number")

    with pytest.raises(ValueError, match="PORT environment variable must be an integer."):
        render_entry_module._read_render_port()
