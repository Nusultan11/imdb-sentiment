from __future__ import annotations

from typing import Iterable


def predict_texts(model, texts: Iterable[str]):
    return model.predict(list(texts))
