from __future__ import annotations

import pytest

from imdb_sentiment.features.lstm_preprocessing import tokenize_lstm_text


@pytest.mark.parametrize(
    ("text", "expected_tokens"),
    [
        ("amazing!!!", ["amazing", "!", "!", "!"]),
        ("don't like it", ["don't", "like", "it"]),
        ("worst-film-ever", ["worst", "film", "ever"]),
        ("It was <br /> good", ["it", "was", "good"]),
    ],
)
def test_tokenize_lstm_text_regex_v2_matches_expected_tokens(
    text: str,
    expected_tokens: list[str],
) -> None:
    tokens = tokenize_lstm_text(text, preprocessing="regex_v2")

    assert tokens == expected_tokens


def test_tokenize_lstm_text_regex_v2_filters_weird_unicode_punctuation() -> None:
    text = "It \u0096 was \u201creally\u201d \u0091bad\u0092 ???"

    tokens = tokenize_lstm_text(text, preprocessing="regex_v2")

    assert tokens == ["it", "was", "really", "bad", "?", "?", "?"]
    assert all(token not in {"-", '"', "\u0091", "\u0092", "\u0096"} for token in tokens)
