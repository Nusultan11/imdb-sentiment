import joblib
import pytest
from sklearn.exceptions import InconsistentVersionWarning

from imdb_sentiment.inference.predict import load_model, predict_from_model_path, predict_texts
from imdb_sentiment.models.baseline import build_baseline_model


def test_inference_returns_binary_predictions(tmp_path) -> None:
    model = build_baseline_model(
        max_features=100,
        ngram_range=(1, 2),
        max_iter=100,
        random_state=42,
    )
    train_texts = ["amazing movie", "great acting", "bad movie", "awful plot"]
    train_labels = [1, 1, 0, 0]
    model.fit(train_texts, train_labels)

    model_path = tmp_path / "baseline.joblib"
    joblib.dump(model, model_path)
    loaded_model = load_model(model_path)

    texts = [
        "This movie was amazing, I loved it.",
        "Terrible film. Waste of time.",
    ]

    preds = predict_texts(loaded_model, texts)

    assert isinstance(preds, list)
    assert len(preds) == 2
    assert all(pred in [0, 1] for pred in preds)


def test_load_model_raises_runtime_error_on_sklearn_version_mismatch(monkeypatch, tmp_path) -> None:
    model_path = tmp_path / "baseline.joblib"
    model_path.write_text("placeholder", encoding="utf-8")

    def _load_with_warning(_path):
        import warnings

        warnings.warn(
            InconsistentVersionWarning(
                estimator_name="Pipeline",
                current_sklearn_version="1.7.1",
                original_sklearn_version="1.6.1",
            )
        )
        return object()

    monkeypatch.setattr("imdb_sentiment.inference.predict.joblib.load", _load_with_warning)

    with pytest.raises(RuntimeError, match="scikit-learn versions do not match"):
        load_model(model_path)


def test_predict_from_model_path_returns_binary_predictions(tmp_path) -> None:
    model = build_baseline_model(
        max_features=100,
        ngram_range=(1, 2),
        max_iter=100,
        random_state=42,
    )
    train_texts = ["excellent film", "loved the acting", "boring movie", "hated the ending"]
    train_labels = [1, 1, 0, 0]
    model.fit(train_texts, train_labels)

    model_path = tmp_path / "baseline.joblib"
    joblib.dump(model, model_path)

    preds = predict_from_model_path(
        model_path,
        ["What a wonderful movie.", "This was a terrible waste of time."],
    )

    assert isinstance(preds, list)
    assert len(preds) == 2
    assert all(pred in [0, 1] for pred in preds)


def test_predict_texts_accepts_generator_input(tmp_path) -> None:
    model = build_baseline_model(
        max_features=100,
        ngram_range=(1, 2),
        max_iter=100,
        random_state=42,
    )
    train_texts = ["excellent movie", "great acting", "boring movie", "terrible acting"]
    train_labels = [1, 1, 0, 0]
    model.fit(train_texts, train_labels)

    text_stream = (
        text
        for text in [
            "excellent story",
            "terrible story",
        ]
    )

    preds = predict_texts(model, text_stream)

    assert preds == [1, 0]
    assert all(isinstance(pred, int) for pred in preds)
