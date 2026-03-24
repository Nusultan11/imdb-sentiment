from __future__ import annotations

import re


WHITESPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")


def normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    cleaned = NON_ALNUM_RE.sub(" ", lowered)
    return WHITESPACE_RE.sub(" ", cleaned).strip()
