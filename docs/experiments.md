# Experiments

This document tracks the experiment-oriented scaffold for the repository.

## Baseline contract

The current baseline is:

```text
family=tfidf
name=baseline
model=TF-IDF + Logistic Regression
seed=42
config=configs/baseline.yaml
```

The baseline repository contract is:

- model artifacts are local outputs, not tracked source files
- validation metrics are written separately from test metrics
- test-set scoring happens only through the evaluation flow

## Config schema

Every experiment config should include:

```yaml
experiment:
  family: ...
  name: ...

seed: 42

paths:
  model_output: ...
  val_metrics_output: ...
  test_metrics_output: ...

model:
  type: ...
```

This keeps experiment identity, validation outputs, and test outputs explicit.

## Available config scaffolds

- `configs/baseline.yaml`
- `configs/experiments/tfidf_baseline_v1.yaml`
- `configs/experiments/tfidf_max_features_10000.yaml`
- `configs/experiments/lstm_baseline_v1.yaml`
- `configs/experiments/transformer_distilbert_v1.yaml`

## Expected output structure

Typical output files:

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

## Workflow

Baseline and experiment workflow:

1. train with `python -m imdb_sentiment.cli train --config ...`
2. run local predictions with `python -m imdb_sentiment.cli predict --config ... --text ...`
3. score the test split with `python -m imdb_sentiment.cli evaluate --config ...`

## CI guardrails

The repository now checks the workflow automatically in `.github/workflows/ci.yml`:

- `ruff check .`
- `pytest -q`
