# imdb-sentiment

Experiment-ready ML project for sentiment analysis on IMDb reviews.

The repository keeps a clean **TF-IDF + Logistic Regression** baseline and separates the workflow into:

- `train`: fit on the IMDb training split and report validation metrics from an internal train/validation split
- `prepare-data`: export explicit `train/val/test` files for LSTM experiments
- `predict`: run local inference with a saved family-specific model artifact
- `evaluate`: score a saved model on the IMDb test split and write test metrics separately

This separation keeps the baseline honest:

- no fitting on the test split
- no leakage from test-time evaluation into training-time metrics
- explicit experiment metadata in YAML configs

---

## Features

- clean `src/` project layout
- YAML-based configuration with experiment metadata and family-specific model fields
- deterministic TF-IDF preprocessing inside the sklearn pipeline
- LSTM text pipeline with tokenizer, vocabulary, token ids, padding, dataset, and dataloader
- shared LSTM artifact contract for checkpoint sidecars and runtime loading
- separate LSTM trainer with batch training, validation loop, and best-checkpoint saving
- family-aware evaluation for TF-IDF artifacts and LSTM checkpoints
- family-aware prediction for TF-IDF artifacts and LSTM checkpoints
- fail-fast model loading when `scikit-learn` versions do not match
- Hugging Face first, local CSV fallback second for dataset loading
- CLI entrypoints for `train`, `prepare-data`, `predict`, and `evaluate`
- family-aware experiment orchestration for TF-IDF, LSTM, and transformer experiments
- GitHub Actions CI for `ruff` and `pytest`

---

## LSTM Experiments

- Built a baseline 1-layer LSTM for IMDb sentiment classification
- Improved it with bidirectional encoding
- Replaced last hidden aggregation with `masked_mean` pooling
- Tuned key hyperparameters with Optuna on the strongest masked-mean BiLSTM branch
- Best archived validation result for the LSTM family: `F1 = 0.8952`

See: [lstm_experiments.md](/C:/Users/nurs/OneDrive/Рабочий%20стол/imdb-sentiment/docs/experiments/lstm_experiments.md)

---

## Project layout

```text
imdb-sentiment/
|-- README.md
|-- AGENTS.md
|-- .gitignore
|-- pyproject.toml
|-- .github/
|   `-- workflows/
|       `-- ci.yml
|
|-- configs/
|   |-- baseline.yaml
|   `-- experiments/
|       |-- tfidf_baseline_v1.yaml
|       |-- tfidf_max_features_10000.yaml
|       |-- tfidf_tuned_v2_final.yaml
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
|   |-- tf_idf_baseline_v1.ipynb
|   `-- tfidf_tuned_v2_final.ipynb
|
|-- artifacts/
|   |-- models/
|   |   `-- .gitkeep
|   |-- reports/
|   |   `-- .gitkeep
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
|       |-- artifacts/
|       |   |-- __init__.py
|       |   `-- lstm.py
|       |-- data/
|       |   |-- dataset.py
|       |   `-- lstm.py
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
|           |-- prepare_data.py
|           |-- prepare_lstm_data.py
|           |-- train_lstm.py
|           |-- train_tfidf.py
|           `-- evaluation.py
|
`-- tests/
    |-- test_baseline_model.py
    |-- test_cli.py
    |-- test_dataset.py
    |-- test_evaluation.py
    |-- test_inference.py
    |-- test_lstm_data.py
    |-- test_lstm_model.py
    |-- test_preprocess.py
    |-- test_settings.py
    |-- test_train.py
    `-- test_train_lstm.py
```

---

## Configuration

Each config now carries experiment identity as well as separate artifact paths for validation and test outputs.
The schema is now family-aware: TF-IDF, LSTM, and transformer configs do not need to pretend
they share the same model hyperparameters.

Example from `configs/baseline.yaml`:

```yaml
experiment:
  family: tfidf
  name: baseline

seed: 42

paths:
  model_output: artifacts/models/baseline.joblib
  val_metrics_output: artifacts/reports/val_metrics.json
  test_metrics_output: artifacts/reports/test_metrics.json

model:
  type: logistic_regression
  max_features: 20000
  ngram_range: [1, 2]
  max_iter: 1000
```

Why this shape is useful:

- `experiment.family` and `experiment.name` make runs easier to organize
- validation and test metrics no longer overwrite each other
- experiment configs can scale without reusing one flat baseline schema forever
- TF-IDF keeps vectorizer/classifier params, while neural families can define their own fields

Example LSTM config:

```yaml
experiment:
  family: lstm
  name: baseline_v1

seed: 42

paths:
  model_output: artifacts/experiments/lstm/baseline_v1/model.pt
  val_metrics_output: artifacts/experiments/lstm/baseline_v1/val_metrics.json
  test_metrics_output: artifacts/experiments/lstm/baseline_v1/test_metrics.json

model:
  type: lstm
  vocab_size: 20000
  max_length: 256
  embedding_dim: 128
  hidden_dim: 128
  batch_size: 32
  epochs: 5
  dropout: 0.3
  lr: 0.001
```

For the current local LSTM trainer, this config produces an artifact directory centered around `paths.model_output`.
The standardized local contract is:

- required for inference: `model.pt`, `vocab.json`, `training_config.json`
- required for evaluation input: `model.pt`, `vocab.json`, `training_config.json`
- written during training: `model.pt`, `vocab.json`, `training_config.json`, `val_metrics.json`
- written during test evaluation: `test_metrics.json`

`training_config.json` also serializes the expected artifact filenames so inference and evaluation can
fail fast if the checkpoint directory no longer matches the expected LSTM bundle.

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

Core runtime dependencies currently declared in `pyproject.toml`:

- `scikit-learn==1.6.1`
- `numpy>=1.26`
- `torch>=2.2`
- `datasets>=2.0`
- `PyYAML>=6.0.1`
- `pandas>=2.2`
- `joblib>=1.3`

Install the project with dev tooling:

```powershell
.\.venv\Scripts\python -m pip install -e .[dev]
```

If you want to run local inference on a saved artifact, use the project environment:

```powershell
.\.venv\Scripts\python -c "import sklearn; print(sklearn.__version__)"
```

Expected output:

```text
1.6.1
```

If the runtime version does not match the artifact version, the inference layer raises a clear `RuntimeError` instead of silently continuing.

---

## CLI usage

Train the TF-IDF baseline and write validation metrics:

```powershell
$env:PYTHONPATH="src"
python -m imdb_sentiment.cli train --config configs/baseline.yaml
```

Train the LSTM experiment and write the best checkpoint plus validation metrics:

```powershell
$env:PYTHONPATH="src"
python -m imdb_sentiment.cli train --config configs/experiments/lstm_baseline_v1.yaml
```

Run inference with a saved TF-IDF model:

```powershell
$env:PYTHONPATH="src"
python -m imdb_sentiment.cli predict --config configs/baseline.yaml --text "This movie was amazing." --text "This film was awful."
```

Run inference with a saved LSTM checkpoint bundle:

```powershell
$env:PYTHONPATH="src"
python -m imdb_sentiment.cli predict --config configs/experiments/lstm_baseline_v1.yaml --text "This movie was amazing." --text "This film was awful."
```

Evaluate a saved TF-IDF model on the IMDb test split:

```powershell
$env:PYTHONPATH="src"
python -m imdb_sentiment.cli evaluate --config configs/baseline.yaml
```

Evaluate a saved LSTM checkpoint on the IMDb test split:

```powershell
$env:PYTHONPATH="src"
python -m imdb_sentiment.cli evaluate --config configs/experiments/lstm_baseline_v1.yaml
```

Prepare LSTM data artifacts:

```powershell
$env:PYTHONPATH="src"
python -m imdb_sentiment.cli prepare-data --config configs/experiments/lstm_baseline_v1.yaml
```

Expected prediction output shape:

```json
{"predictions": [1, 0]}
```

---

## Training and evaluation semantics

The repository now makes the metric split explicit:

- `train.py` fits on `dataset["train"]`
- `train.py` is now a family-aware orchestration layer
- TF-IDF training lives in its own family-specific runner
- LSTM training lives in its own family-specific runner
- LSTM data preparation lives in its own family-specific pipeline
- TF-IDF training creates an internal validation split from `dataset["train"]`
- LSTM training creates its own train/validation split from `dataset["train"]`
- LSTM data preparation exports explicit `train`, `val`, and `test` JSONL files
- TF-IDF training writes validation metrics to `paths.val_metrics_output`
- `evaluation.py` is family-aware and scores only `dataset["test"]`
- `evaluation.py` writes test metrics to `paths.test_metrics_output`

This means the README, config schema, and runtime logic now describe the same system.

Current implementation status:

- TF-IDF training runner is implemented
- TF-IDF inference path is implemented through `.joblib` artifacts
- LSTM config schema is implemented
- LSTM local data preparation pipeline is implemented
- LSTM local training runner is implemented
- LSTM local evaluation path is implemented for `.pt` checkpoints
- LSTM local predict/inference path is implemented for standardized checkpoint bundles
- transformer config schema is implemented
- transformer training runner is not implemented yet and fails fast with `NotImplementedError`

---

## Artifacts policy

Model and metrics files are treated as local outputs, not as committed source files.

Typical TF-IDF local outputs:

- `artifacts/models/baseline.joblib`
- `artifacts/reports/val_metrics.json`
- `artifacts/reports/test_metrics.json`

Typical LSTM local outputs:

- `artifacts/experiments/lstm/baseline_v1/model.pt`
- `artifacts/experiments/lstm/baseline_v1/vocab.json`
- `artifacts/experiments/lstm/baseline_v1/training_config.json`
- `artifacts/experiments/lstm/baseline_v1/val_metrics.json`
- `artifacts/experiments/lstm/baseline_v1/test_metrics.json`

Standardized LSTM contract:

- `model.pt` stores the torch `state_dict`
- `vocab.json` stores the token-to-id vocabulary used by training and inference
- `training_config.json` stores model hyperparameters plus the expected artifact filenames
- `val_metrics.json` is produced by training
- `test_metrics.json` is produced by evaluation

Why the LSTM bundle is larger than TF-IDF:

- TF-IDF keeps preprocessing and classifier state inside one `.joblib` pipeline
- LSTM needs model weights plus vocabulary and text-shape metadata
- separate JSON artifacts make debugging and later inference wiring easier

---

## Comparing TF-IDF and LSTM

Use the TF-IDF baseline first as the reference point, then compare the LSTM branch against it.

| Area | TF-IDF baseline | LSTM branch |
|---|---|---|
| Training command | `train --config configs/baseline.yaml` | `train --config configs/experiments/lstm_baseline_v1.yaml` |
| Main artifact | `.joblib` pipeline | `.pt` checkpoint + JSON bundle |
| Text handling | sklearn vectorizer inside pipeline | explicit tokenizer/vocab/padding pipeline |
| Validation | internal split inside TF-IDF trainer | internal split inside LSTM trainer |
| Test evaluation | supported | supported |
| Local predict CLI | supported | supported |

The repository keeps placeholder directories with `.gitkeep`, but real trained binaries and report files are ignored by git.

---

## CI

The repository includes GitHub Actions CI in `.github/workflows/ci.yml`.

On each push to `master` and on pull requests, CI runs:

1. dependency installation
2. `ruff check .`
3. `pytest -q`

---

## Verification

Useful local checks:

```powershell
$env:PYTHONPATH="src"
python -m pytest -q
ruff check .
python -m imdb_sentiment.cli train --config configs/baseline.yaml
python -m imdb_sentiment.cli evaluate --config configs/baseline.yaml
python -m imdb_sentiment.cli train --config configs/experiments/lstm_baseline_v1.yaml
python -m imdb_sentiment.cli predict --config configs/experiments/lstm_baseline_v1.yaml --text "This movie was amazing."
python -m imdb_sentiment.cli evaluate --config configs/experiments/lstm_baseline_v1.yaml
```
