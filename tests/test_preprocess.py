from imdb_sentiment.features.preprocess import normalize_review_text


def test_normalize_review_text_removes_html_and_extra_spaces() -> None:
    raw_text = "Hello<br /><br />world   "
    expected = "hello world"
    assert normalize_review_text(raw_text) == expected
