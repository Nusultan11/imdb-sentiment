# imdb-sentiment

Experiment-ready ML project for sentiment analysis on IMDb reviews.

The repository keeps a clean, stable **TF-IDF + Logistic Regression** baseline while also providing a scaffold for future **TF-IDF**, **LSTM**, and **transformer** experiments. The current tracked baseline includes:

- a saved model artifact in `artifacts/models/baseline.joblib`
- tracked baseline metrics in `artifacts/reports/metrics.json`
- CLI commands for training and inference
- an evaluation pipeline for test-set scoring

---

## Features

- clean `src/` project layout
- YAML-based configuration
- offline-capable dataset loading with Hugging Face first and local CSV fallback second
- deterministic TF-IDF preprocessing inside the sklearn pipeline
- fail-fast model loading when `scikit-learn` versions do not match
- CLI entrypoints for both `train` and `predict`
- experiment config scaffolds for TF-IDF, LSTM, and transformer work
- tracked baseline artifacts for reproducible local inference

---

## Project layout

```text
imdb-sentiment/
|-- README.md
|-- AGENTS.md
|-- .gitignore
|-- pyproject.toml
|
|-- configs/
|   |-- baseline.yaml
|   `-- experiments/
|       |-- tfidf_baseline_v1.yaml
|       |-- tfidf_max_features_10000.yaml
|       |-- lstm_baseline_v1.yaml
|       `-- transformer_distilbert_v1.yaml
|
|-- docs/
|   `-- experiments.md
|
|-- data/
|   |-- raw/
|   |   `-- .gitkeep
|   |-- interim/
|   |   `-- .gitkeep
|   `-- processed/
|       `-- .gitkeep
|
|-- notebooks/
|   |-- .gitkeep
|   |-- baseline_eda_imdb.ipynb
|   `-- tf_idf_baseline_v1.ipynb
|
|-- artifacts/
|   |-- models/
|   |   |-- .gitkeep
|   |   `-- baseline.joblib
|   |-- reports/
|   |   |-- .gitkeep
|   |   `-- metrics.json
|   `-- experiments/
|       |-- tfidf/
|       |   |-- .gitkeep
|       |   |-- baseline_v1/
|       |   |   `-- .gitkeep
|       |   `-- max_features_10000/
|       |       `-- .gitkeep
|       |-- lstm/
|       |   |-- .gitkeep
|       |   `-- baseline_v1/
|       |       `-- .gitkeep
|       `-- transformer/
|           |-- .gitkeep
|           `-- distilbert_v1/
|               `-- checkpoint/
|                   `-- .gitkeep
|
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
|       |   |-- baseline.py
|       |   |-- tfidf/
|       |   |   `-- baseline.py
|       |   |-- lstm/
|       |   |   `-- model.py
|       |   `-- transformer/
|       |       `-- model.py
|       `-- pipelines/
|           |-- train.py
|           `-- evaluation.py
|
`-- tests/
    |-- test_baseline_model.py
    |-- test_cli.py
    |-- test_dataset.py
    |-- test_evaluation.py
    |-- test_inference.py
    |-- test_preprocess.py
    `-- test_train.py
```

---

## Data ingestion behavior

The dataset loader follows this order:

1. try to load the IMDb dataset from Hugging Face
2. if network access fails, try local CSV fallback files
3. validate that both `train` and `test` splits exist
4. validate that each split contains `text` and `label`

Expected fallback files:

- `data/raw/imdb_train.csv`
- `data/raw/imdb_test.csv`

Expected CSV schema:

```text
text,label
This movie was great,1
This movie was terrible,0
```

---

## Environment

The committed baseline artifact was created with `scikit-learn 1.6.1`.

If you want to run inference on the tracked `baseline.joblib`, use the local project environment:

```powershell
.\.venv\Scripts\python -c "import sklearn; print(sklearn.__version__)"
```

Expected output:

```text
1.6.1
```

If the runtime version does not match the artifact version, the inference layer now fails fast with a clear `RuntimeError` instead of silently continuing.

---

## CLI usage

Train the baseline:

```powershell
$env:PYTHONPATH="src"
python -m imdb_sentiment.cli train --config configs/baseline.yaml
```

Run inference with a saved model:

```powershell
$env:PYTHONPATH="src"
python -m imdb_sentiment.cli predict --config configs/baseline.yaml --text "This movie was amazing." --text "This film was awful."
```

Expected output shape:

```json
{"predictions": [1, 0]}
```

---

## Baseline artifacts

Current tracked baseline files:

- `artifacts/models/baseline.joblib`
- `artifacts/reports/metrics.json`

Current tracked baseline metrics:

```json
{
  "accuracy": 0.8976,
  "precision": 0.8994391025641025,
  "recall": 0.8958499600957701,
  "f1": 0.897640943622551
}
```

---

## Evaluation

Test-set evaluation lives in `src/imdb_sentiment/pipelines/evaluation.py`.

Current evaluation flow:

1. load a saved model
2. load the IMDb dataset
3. score the test split
4. write test metrics to `artifacts/reports/test_metrics.json`

---

## Verification

Useful checks:

```powershell
$env:PYTHONPATH="src"
python -m pytest tests/test_dataset.py -q
python -m pytest tests/test_inference.py -q
python -m pytest tests/test_cli.py -q
python -m pytest tests/test_evaluation.py -q
python -m pytest -q
```
