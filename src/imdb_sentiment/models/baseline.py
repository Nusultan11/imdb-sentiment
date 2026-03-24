from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_baseline_model(max_features: int, ngram_range: tuple[int, int], max_iter: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(max_features=max_features, ngram_range=ngram_range),
            ),
            ("classifier", LogisticRegression(max_iter=max_iter)),
        ]
    )
