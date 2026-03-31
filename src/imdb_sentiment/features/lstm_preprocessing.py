from __future__ import annotations

from collections.abc import Callable
from html import unescape
import re
import unicodedata

from imdb_sentiment.features.preprocess import normalize_review_text


HTML_BREAK_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"\.\.\.|[a-z0-9]+(?:'[a-z0-9]+)*|[!?]")

DIRTY_TEXT_TRANSLATION = str.maketrans(
    {
        "\u0091": "'",
        "\u0092": "'",
        "\u0093": '"',
        "\u0094": '"',
        "\u0096": "-",
        "\u0097": "-",
        "\u00a0": " ",
        "\u200b": " ",
        "\ufeff": " ",
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2026": " ... ",
    }
)


LSTMTokenizer = Callable[[str], list[str]]


def tokenize_lstm_text_v1(text: str) -> list[str]:
    normalized_text = normalize_review_text(text)
    if not normalized_text:
        return []
    return normalized_text.split(" ")


def normalize_lstm_text_v2(text: str) -> str:
    text = unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(DIRTY_TEXT_TRANSLATION)
    text = text.lower()
    text = HTML_BREAK_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def tokenize_lstm_text_v2(text: str) -> list[str]:
    normalized_text = normalize_lstm_text_v2(text)
    if not normalized_text:
        return []
    return TOKEN_RE.findall(normalized_text)


TOKENIZER_REGISTRY: dict[str, LSTMTokenizer] = {
    "whitespace_v1": tokenize_lstm_text_v1,
    "regex_v2": tokenize_lstm_text_v2,
}


def get_lstm_tokenizer(preprocessing: str = "whitespace_v1") -> LSTMTokenizer:
    tokenizer = TOKENIZER_REGISTRY.get(preprocessing)
    if tokenizer is None:
        raise ValueError(
            "Unsupported LSTM preprocessing strategy. "
            f"Expected one of: {sorted(TOKENIZER_REGISTRY)}"
        )
    return tokenizer


def tokenize_lstm_text(text: str, preprocessing: str = "whitespace_v1") -> list[str]:
    return get_lstm_tokenizer(preprocessing)(text)


__all__ = [
    "get_lstm_tokenizer",
    "LSTMTokenizer",
    "tokenize_lstm_text",
    "tokenize_lstm_text_v1",
    "normalize_lstm_text_v2",
    "tokenize_lstm_text_v2",
]
