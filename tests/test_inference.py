import joblib

from imdb_sentiment.inference.predict import load_model, predict_texts
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
