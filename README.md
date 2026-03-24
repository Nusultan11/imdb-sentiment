# imdb-sentiment

ML-проект для классификации тональности отзывов IMDb.

## Архитектура

```text
imdb-sentiment/
├── README.md
├── .gitignore
├── pyproject.toml
├── configs/
│   └── baseline.yaml
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   ├── interim/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
├── notebooks/
│   └── .gitkeep
├── src/
│   └── imdb_sentiment/
│       ├── __init__.py
│       ├── cli.py
│       ├── settings.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── dataset.py
│       ├── features/
│       │   ├── __init__.py
│       │   └── preprocess.py
│       ├── inference/
│       │   ├── __init__.py
│       │   └── predict.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── baseline.py
│       └── pipelines/
│           ├── __init__.py
│           └── train.py
├── tests/
│   ├── __init__.py
│   ├── test_preprocess.py
│   └── test_train.py
└── artifacts/
    ├── models/
    │   └── .gitkeep
    └── reports/
        └── .gitkeep
```

## Идея слоев

- `configs/`: параметры экспериментов и путей.
- `data/`: сырые, промежуточные и подготовленные данные.
- `src/imdb_sentiment/data`: загрузка и валидация датасета.
- `src/imdb_sentiment/features`: очистка текста и feature engineering.
- `src/imdb_sentiment/models`: baseline и будущие модели.
- `src/imdb_sentiment/pipelines`: orchestration обучения и оценки.
- `src/imdb_sentiment/inference`: предсказания на новых отзывах.
- `artifacts/`: сохраненные модели, метрики и отчеты.

## Быстрый старт

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
pytest
python -m imdb_sentiment.cli
```
