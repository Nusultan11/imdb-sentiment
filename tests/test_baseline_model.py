from imdb_sentiment.features.preprocess import normalize_review_text
from imdb_sentiment.models.baseline import build_baseline_model


def test_baseline_vectorizer_uses_explicit_preprocessing_contract() -> None:
    model = build_baseline_model(
        max_features=100,
        ngram_range=(1, 2),
        max_iter=100,
        random_state=42,
    )
    vectorizer = model.named_steps["vectorizer"]

    assert vectorizer.preprocessor is normalize_review_text
    assert vectorizer.lowercase is False
