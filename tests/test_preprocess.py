from imdb_sentiment.features.preprocess import normalize_text


def test_normalize_text_removes_noise() -> None:
    assert normalize_text("  Amazing MOVIE!!! 10/10  ") == "amazing movie 10 10"
