from __future__ import annotations

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from imdb_sentiment.data.dataset import load_dataset
from imdb_sentiment.features.preprocess import normalize_text
from imdb_sentiment.models.baseline import build_baseline_model
from imdb_sentiment.settings import AppConfig


def run_training(config: AppConfig) -> dict[str, float]:
    df = load_dataset(config.paths.raw_data)
    texts = df[config.data.text_column].astype(str).map(normalize_text)
    labels = df[config.data.target_column]

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=config.data.test_size,
        random_state=config.seed,
        stratify=labels,
    )

    model = build_baseline_model(
        max_features=config.model.max_features,
        ngram_range=config.model.ngram_range,
        max_iter=config.model.max_iter,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return {"accuracy": float(accuracy_score(y_test, predictions))}
