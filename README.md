# imdb-sentiment

Baseline sentiment analysis project for IMDb reviews.

## Project layout

```text
imdb-sentiment/
|-- README.md
|-- .gitignore
|-- pyproject.toml
|-- configs/
|   `-- baseline.yaml
|-- data/
|   |-- raw/
|   |   `-- .gitkeep
|   |-- interim/
|   |   `-- .gitkeep
|   `-- processed/
|       `-- .gitkeep
|-- notebooks/
|   |-- .gitkeep
|   `-- baseline_eda_imdb.ipynb
|-- src/
|   `-- imdb_sentiment/
|       |-- cli.py
|       |-- settings.py
|       |-- data/
|       |-- features/
|       |-- inference/
|       |-- models/
|       `-- pipelines/
|-- tests/
`-- artifacts/
```

## Current pipeline

The baseline training pipeline downloads the IMDb dataset through the Hugging Face
`datasets` package, normalizes review text, trains a TF-IDF + Logistic Regression
model, and stores:

- the trained model in `artifacts/models/baseline.joblib`
- evaluation metrics in `artifacts/reports/metrics.json`

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
python -m pytest
```

## Train the model

```powershell
$env:PYTHONPATH="src"
python -m imdb_sentiment.cli
```

After training, check:

- `artifacts/models/baseline.joblib`
- `artifacts/reports/metrics.json`

## Notes

- Training needs network access the first time because the IMDb dataset is downloaded
  from Hugging Face.
- Tests do not need network access because the dataset loader is mocked in
  `tests/test_train.py`.
