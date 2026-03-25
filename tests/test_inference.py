from imdb_sentiment.inference.predict import load_model, predict_texts
from imdb_sentiment.settings import load_config


def test_inference_returns_binary_predictions() -> None:
    config = load_config()
    model = load_model(config.paths.model_output)

    texts = [
        "This movie was amazing, I loved it.",
        "Terrible film. Waste of time.",
    ]

    preds = predict_texts(model, texts)

    assert isinstance(preds, list)
    assert len(preds) == 2
    assert all(pred in [0, 1] for pred in preds)
