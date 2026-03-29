# Experiments

This document tracks the current experiment-oriented project scaffold.

## Active baseline

The tracked baseline artifact in the repository is:

```text
family=tfidf
name=baseline_v1
model=TF-IDF + Logistic Regression
seed=42
artifact=artifacts/models/baseline.joblib
metrics=artifacts/reports/metrics.json
```

Current tracked metrics:

```text
accuracy=0.8976
precision=0.8994391025641025
recall=0.8958499600957701
f1=0.897640943622551
```

## Experiment configs

Available experiment config scaffolds:

- `configs/experiments/tfidf_baseline_v1.yaml`
- `configs/experiments/tfidf_max_features_10000.yaml`
- `configs/experiments/lstm_baseline_v1.yaml`
- `configs/experiments/transformer_distilbert_v1.yaml`

## Expected experiment outputs

Each experiment family stores outputs under `artifacts/experiments/`:

- TF-IDF:
  - `model.joblib`
  - `val_metrics.json`
  - `test_metrics.json`
  - `config_snapshot.yaml`
- LSTM:
  - `model.pt`
  - `val_metrics.json`
  - `test_metrics.json`
  - `config_snapshot.yaml`
- Transformer:
  - `checkpoint/`
  - `val_metrics.json`
  - `test_metrics.json`
  - `config_snapshot.yaml`

## Evaluation flow

Current baseline workflow:

1. train with `python -m imdb_sentiment.cli train --config ...`
2. predict with `python -m imdb_sentiment.cli predict --config ... --text ...`
3. evaluate with `imdb_sentiment.pipelines.evaluation.run_evaluation(...)`
