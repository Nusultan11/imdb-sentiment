from __future__ import annotations

from html import unescape
import re


HTML_BREAK_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_review_text(text: str) -> str:
    text = unescape(text)
    text = text.lower()
    text = HTML_BREAK_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text
