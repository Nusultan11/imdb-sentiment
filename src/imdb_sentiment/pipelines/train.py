from __future__ import annotations

from sklearn.metrics import accuracy_score
from datasets import load_dataset

from imdb_sentiment.features.preprocess import normalize_review_text
from imdb_sentiment.models.baseline import build_baseline_model
from imdb_sentiment.settings import AppConfig


def run_training(config: AppConfig) -> dict[str, float]:
    dataset=load_dataset("imdb")
    train_split=dataset["train"]
    test_split=dataset["test"]

    X_train=[normalize_review_text(text) for text in train_split["text"]]
    y_train=train_split["label"]

    X_test=[normalize_review_text(text) for text in test_split["text"]]
    y_test=test_split["label"]

    model=build_baseline_model(
        max_features=config.model.max_features,
        ngram_range=config.model.ngram_range,
        max_iter=config.model.max_iter
    )
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)
    accuracy=accuracy_score(y_test, y_pred)
    return {"accuracy": accuracy}