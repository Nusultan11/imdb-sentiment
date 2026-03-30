from imdb_sentiment.features.preprocess import normalize_review_text


def test_normalize_review_text_removes_html_and_extra_spaces() -> None:
    raw_text = "Hello<br /><br />world   "
    expected = "hello world"
    assert normalize_review_text(raw_text) == expected


def test_normalize_review_text_is_explicitly_lowercase() -> None:
    raw_text = "Great<br /> MOVIE"
    expected = "great movie"
    assert normalize_review_text(raw_text) == expected


def test_normalize_review_text_removes_generic_html_tags() -> None:
    raw_text = "<div><b>Great</b> movie</div>"
    expected = "great movie"
    assert normalize_review_text(raw_text) == expected


def test_normalize_review_text_decodes_html_entities() -> None:
    raw_text = "It&#39;s &quot;great&quot; &amp; intense."
    expected = "it's \"great\" & intense."
    assert normalize_review_text(raw_text) == expected
