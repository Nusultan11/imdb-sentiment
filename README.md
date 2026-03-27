# imdb-sentiment

Baseline ML project for sentiment analysis of IMDb movie reviews.

The project trains a **TF-IDF + Logistic Regression** baseline, evaluates it on the IMDb test split, saves the trained model and evaluation metrics, and provides a simple inference layer for loading the saved model and predicting sentiment for new texts.

---

## Features

- clean `src/` project layout
- YAML-based configuration
- dataset loading from Hugging Face with local CSV fallback
- dataset validation (`train` / `test`, required columns)
- text normalization integrated into the model pipeline
- baseline model: **TF-IDF + Logistic Regression**
- saved artifacts:
  - trained model
  - evaluation metrics
- inference utilities for loading the saved model and running predictions
- pytest-based tests for preprocessing, training, and inference

---

## Project layout

```text
imdb-sentiment/
|-- README.md
|-- AGENTS.md
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
|-- artifacts/
|   |-- models/
|   |   `-- .gitkeep
|   `-- reports/
|       `-- .gitkeep
|-- src/
|   `-- imdb_sentiment/
|       |-- cli.py
|       |-- settings.py
|       |-- data/
|       |   `-- dataset.py
|       |-- features/
|       |   `-- preprocess.py
|       |-- inference/
|       |   `-- predict.py
|       |-- models/
|       |   `-- baseline.py
|       `-- pipelines/
|           `-- train.py
`-- tests/