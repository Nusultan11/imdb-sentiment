from __future__ import annotations

import re


HTML_BREAK_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


def normalize_review_text(text: str) -> str:
    text = text.lower()
    text = HTML_BREAK_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text