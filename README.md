# imdb-sentiment

Experiment-ready ML project for sentiment analysis on IMDb reviews.

The repository keeps a clean TF-IDF + Logistic Regression baseline, a family-aware LSTM pipeline, and explicit separation between training, prediction, evaluation, and auxiliary export workflows.

## Workflow

- `train`: fit on the IMDb training split and report validation metrics from an internal train/validation split
- `prepare-data`: export explicit `train/val/test` files for LSTM experiments; this is an auxiliary export tool and the main LSTM trainer does not read these JSONL files back
- `predict`: run local inference with a saved family-specific model artifact
- `evaluate`: score a saved model on the IMDb test split and write test metrics separately

This keeps the pipeline honest:

- no fitting on the test split
- no leakage from test-time evaluation into training-time metrics
- explicit experiment metadata in YAML configs

## Features

- clean `src/` project layout
- YAML-based configuration with experiment metadata and family-specific model fields
- deterministic TF-IDF preprocessing inside the sklearn pipeline
- LSTM text pipeline with tokenizer, vocabulary, token ids, padding, dataset, and dataloader
- shared LSTM artifact contract for checkpoint sidecars and runtime loading
- separate LSTM trainer with batch training, validation loop, and best-checkpoint saving
- family-aware evaluation for TF-IDF artifacts and LSTM checkpoints
- family-aware prediction for TF-IDF artifacts and LSTM checkpoints
- local LSTM bundle import for Colab-trained checkpoints
- saved model-comparison reports with automatic winner selection
- Hugging Face first, local CSV fallback second for dataset loading
- GitHub Actions CI for `ruff` and `pytest`

## LSTM Experiments

The LSTM branch evolved in this order:

- baseline 1-layer LSTM
- bidirectional LSTM
- BiLSTM with `masked_mean` pooling
- Optuna-tuned BiLSTM + `masked_mean`
- final independent `regex_v2 + BiLSTM + masked_mean` run

Available LSTM preprocessing modes:

- `whitespace_v1`: legacy whitespace-oriented normalization path
- `regex_v2`: regex-driven tokenization path for the main local LSTM experiments

Main local LSTM configs such as [lstm_baseline_v1.yaml](/C:/Users/nurs/OneDrive/Рабочий%20стол/imdb-sentiment/configs/experiments/lstm_baseline_v1.yaml), [lstm_bidirectional_v1.yaml](/C:/Users/nurs/OneDrive/Рабочий%20стол/imdb-sentiment/configs/experiments/lstm_bidirectional_v1.yaml), and [lstm_bidirectional_masked_mean_optuna_regexprep_v1.yaml](/C:/Users/nurs/OneDrive/Рабочий%20стол/imdb-sentiment/configs/experiments/lstm_bidirectional_masked_mean_optuna_regexprep_v1.yaml) use explicit LSTM runtime settings instead of hidden defaults.

Strongest saved local LSTM run:

- experiment: `bidirectional_masked_mean_optuna_regexprep_v1`
- architecture: `bidirectional=True`, `pooling=masked_mean`
- preprocessing: `regex_v2`
- validation metrics from [val_metrics.json](/C:/Users/nurs/OneDrive/Рабочий%20стол/imdb-sentiment/artifacts/experiments/lstm/bidirectional_masked_mean_optuna_regexprep_v1/val_metrics.json):
  - `accuracy = 0.9046`
  - `precision = 0.8844`
  - `recall = 0.9314`
  - `f1 = 0.9073`
- test metrics from [test_metrics.json](/C:/Users/nurs/OneDrive/Рабочий%20стол/imdb-sentiment/artifacts/experiments/lstm/bidirectional_masked_mean_optuna_regexprep_v1/test_metrics.json):
  - `accuracy = 0.8848`
  - `precision = 0.8651`
  - `recall = 0.9118`
  - `f1 = 0.8878`
- tuned threshold from [threshold_tuning.json](/C:/Users/nurs/OneDrive/Рабочий%20стол/imdb-sentiment/artifacts/experiments/lstm/bidirectional_masked_mean_optuna_regexprep_v1/threshold_tuning.json): `0.47`

## Final Local Comparison

Saved comparison artifacts:

- [all_models_test_metrics.csv](/C:/Users/nurs/OneDrive/Рабочий%20стол/imdb-sentiment/artifacts/reports/model_comparison/all_models_test_metrics.csv)
- [all_models_test_metrics.json](/C:/Users/nurs/OneDrive/Рабочий%20стол/imdb-sentiment/artifacts/reports/model_comparison/all_models_test_metrics.json)
- [missing_models.json](/C:/Users/nurs/OneDrive/Рабочий%20стол/imdb-sentiment/artifacts/reports/model_comparison/missing_models.json)
- [winner_summary.json](/C:/Users/nurs/OneDrive/Рабочий%20стол/imdb-sentiment/artifacts/reports/model_comparison/winner_summary.json)
- [evaluation_report_snapshot.json](/C:/Users/nurs/OneDrive/Рабочий%20стол/imdb-sentiment/artifacts/reports/model_comparison/evaluation_report_snapshot.json)

Current final winner on the IMDb test split:

- model: `tuned_v2_final`
- family: `tfidf`
- metrics:
  - `accuracy = 0.8979`
  - `precision = 0.8966`
  - `recall = 0.8994`
  - `f1 = 0.8980`

Strongest saved local LSTM competitor:

- model: `bidirectional_masked_mean_optuna_regexprep_v1`
- family: `lstm`
- preprocessing: `regex_v2`
- metrics:
  - `accuracy = 0.8848`
  - `precision = 0.8651`
  - `recall = 0.9118`
  - `f1 = 0.8878`

Current project conclusion:

- best overall saved model: TF-IDF `tuned_v2_final`
- best saved LSTM model: `regex_v2 + BiLSTM + masked_mean`

## Project Layout

```text
imdb-sentiment/
|-- README.md
|-- AGENTS.md
|-- pyproject.toml
|-- configs/
|   |-- baseline.yaml
|   `-- experiments/
|       |-- tfidf_baseline_v1.yaml
|       |-- tfidf_max_features_10000.yaml
|       |-- tfidf_tuned_v2_final.yaml
|       |-- lstm_baseline_v1.yaml
|       |-- lstm_bidirectional_v1.yaml
|       |-- lstm_bidirectional_masked_mean_v1.yaml
|       `-- lstm_bidirectional_masked_mean_optuna_regexprep_v1.yaml
|-- notebooks/
|   |-- baseline_eda_imdb.ipynb
|   |-- tf_idf_baseline_v1.ipynb
|   |-- tfidf_tuned_v2_final.ipynb
|   |-- evaluation.ipynb
|   `-- lstm/
|       |-- bilstm_masked_mean_v1.ipynb
|       `-- regex_v2_bilstm_masked_pooling.ipynb
|-- artifacts/
|   |-- reports/
|   |   `-- model_comparison/
|   |       |-- all_models_test_metrics.csv
|   |       |-- all_models_test_metrics.json
|   |       |-- missing_models.json
|   |       `-- winner_summary.json
|   `-- experiments/
|       |-- tfidf/
|       |   `-- tuned_v2_final/
|       `-- lstm/
|           |-- baseline_v1/
|           |-- bidirectional_v1/
|           |-- bidirectional_masked_mean_v1/
|           `-- bidirectional_masked_mean_optuna_regexprep_v1/
|-- src/
|   `-- imdb_sentiment/
|       |-- cli.py
|       |-- settings.py
|       |-- artifacts/
|       |   |-- lstm.py
|       |   `-- lstm_runtime.py
|       |-- inference/
|       |   `-- predict.py
|       `-- pipelines/
|           |-- train.py
|           |-- train_tfidf.py
|           |-- train_lstm.py
|           |-- prepare_data.py
|           |-- prepare_lstm_data.py
|           |-- evaluation.py
|           `-- model_comparison.py
`-- tests/
    |-- test_cli.py
    |-- test_evaluation.py
    |-- test_inference.py
    |-- test_lstm_runtime.py
    `-- test_model_comparison.py
```

## Configuration

Each config carries experiment identity as well as separate artifact paths for validation and test outputs.

Example TF-IDF config:

```yaml
experiment:
  family: tfidf
  name: tuned_v2_final

seed: 42

paths:
  model_output: artifacts/experiments/tfidf/tuned_v2_final/model.joblib
  val_metrics_output: artifacts/experiments/tfidf/tuned_v2_final/val_metrics.json
  test_metrics_output: artifacts/experiments/tfidf/tuned_v2_final/test_metrics.json

model:
  type: logistic_regression
  max_features: 38000
  ngram_range: [1, 2]
  max_iter: 1000
```

Example LSTM config:

```yaml
experiment:
  family: lstm
  name: bidirectional_masked_mean_optuna_regexprep_v1

seed: 42

paths:
  model_output: artifacts/experiments/lstm/bidirectional_masked_mean_optuna_regexprep_v1/model.pt
  val_metrics_output: artifacts/experiments/lstm/bidirectional_masked_mean_optuna_regexprep_v1/val_metrics.json
  test_metrics_output: artifacts/experiments/lstm/bidirectional_masked_mean_optuna_regexprep_v1/test_metrics.json

model:
  type: lstm
  vocab_size: 30000
  max_length: 512
  embedding_dim: 128
  hidden_dim: 128
  bidirectional: true
  pooling: masked_mean
  preprocessing: regex_v2
  batch_size: 16
  epochs: 5
  dropout: 0.5
  lr: 0.002074566765675252
```

## Data Ingestion Behavior

The dataset loader follows this order:

1. try to load the IMDb dataset from Hugging Face
2. if network access fails, try local CSV fallback files
3. validate that both `train` and `test` splits exist
4. validate that each split contains `text` and `label`

Expected fallback files:

- `data/raw/imdb_train.csv`
- `data/raw/imdb_test.csv`

## CLI Usage

Train the TF-IDF baseline:

```powershell
$env:PYTHONPATH="src"
python -m imdb_sentiment.cli train --config configs/baseline.yaml
```

Train the final regex LSTM:

```powershell
$env:PYTHONPATH="src"
python -m imdb_sentiment.cli train --config configs/experiments/lstm_bidirectional_masked_mean_optuna_regexprep_v1.yaml
```

Run inference with a saved LSTM checkpoint bundle:

```powershell
$env:PYTHONPATH="src"
python -m imdb_sentiment.cli predict --config configs/experiments/lstm_bidirectional_masked_mean_optuna_regexprep_v1.yaml --text "This movie was amazing." --text "This film was awful."
```

Evaluate a saved model:

```powershell
$env:PYTHONPATH="src"
python -m imdb_sentiment.cli evaluate --config configs/experiments/lstm_bidirectional_masked_mean_optuna_regexprep_v1.yaml
```

Prepare LSTM export-only JSONL artifacts:

```powershell
$env:PYTHONPATH="src"
python -m imdb_sentiment.cli prepare-data --config configs/experiments/lstm_baseline_v1.yaml
```

Import a downloaded LSTM bundle from Colab into the configured experiment directory:

```powershell
$env:PYTHONPATH="src"
python -m imdb_sentiment.cli import-lstm-bundle --config configs/experiments/lstm_bidirectional_masked_mean_optuna_regexprep_v1.yaml --bundle "C:\Users\<you>\Downloads\regex_v2_bilstm_masked_pooling_trained_bundle.zip"
```

Compare all saved local models and write the final winner report:

```powershell
$env:PYTHONPATH="src"
python -m imdb_sentiment.cli compare-models --config configs/experiments/tfidf_tuned_v2_final.yaml --config configs/experiments/lstm_baseline_v1.yaml --config configs/experiments/lstm_bidirectional_v1.yaml --config configs/experiments/lstm_bidirectional_masked_mean_v1.yaml --config configs/experiments/lstm_bidirectional_masked_mean_optuna_regexprep_v1.yaml
```

Expected prediction output:

```json
{"predictions": [1, 0]}
```

## Training and Evaluation Semantics

- TF-IDF training creates an internal validation split from `dataset["train"]`
- LSTM training creates its own train/validation split from `dataset["train"]`
- LSTM `prepare-data` exports explicit `train`, `val`, and `test` JSONL files
- those exported JSONL files are for inspection or external reuse only; the main LSTM trainer still loads IMDb directly and makes its own split
- evaluation reads only `dataset["test"]`
- LSTM evaluation and inference restore runtime settings from saved artifacts, not from guessed external defaults

Current implementation status:

- TF-IDF training/evaluation/predict: implemented
- LSTM training/evaluation/predict: implemented
- transformer config schema: implemented
- transformer training runner: not implemented yet and fails fast with `NotImplementedError`

## Artifacts Policy

Model and metrics files are treated as local outputs, not committed source files.

Standardized LSTM artifact contract:

- `model.pt` stores the torch `state_dict`
- `vocab.json` stores the token-to-id vocabulary used by training and inference
- `training_config.json` stores model hyperparameters plus expected artifact filenames
- `training_history.json` stores epoch-wise training history
- `threshold_tuning.json` stores the tuned decision threshold used by prediction and evaluation
- `val_metrics.json` is produced by training
- `test_metrics.json` is produced by evaluation

Why the LSTM bundle is larger than TF-IDF:

- TF-IDF keeps preprocessing and classifier state inside one `.joblib` pipeline
- LSTM needs model weights plus vocabulary and text-shape metadata
- separate JSON artifacts make debugging and later inference wiring easier

## Comparing TF-IDF and LSTM

Use the TF-IDF branch as the baseline reference point, then compare LSTM checkpoints against it.

| Area | TF-IDF branch | LSTM branch |
|---|---|---|
| Main artifact | `.joblib` pipeline | `.pt` checkpoint + JSON bundle |
| Text handling | sklearn vectorizer inside pipeline | explicit tokenizer/vocab/padding pipeline |
| Validation | internal split inside TF-IDF trainer | internal split inside LSTM trainer |
| Test evaluation | supported | supported |
| Local predict CLI | supported | supported |

## CI

GitHub Actions CI runs:

1. dependency installation
2. `ruff check .`
3. `pytest -q`

## Verification

Useful local checks:

```powershell
$env:PYTHONPATH="src"
python -m pytest -q
ruff check .
python -m imdb_sentiment.cli evaluate --config configs/experiments/tfidf_tuned_v2_final.yaml
python -m imdb_sentiment.cli evaluate --config configs/experiments/lstm_bidirectional_masked_mean_optuna_regexprep_v1.yaml
python -m imdb_sentiment.cli compare-models --config configs/experiments/tfidf_tuned_v2_final.yaml --config configs/experiments/lstm_baseline_v1.yaml --config configs/experiments/lstm_bidirectional_v1.yaml --config configs/experiments/lstm_bidirectional_masked_mean_v1.yaml --config configs/experiments/lstm_bidirectional_masked_mean_optuna_regexprep_v1.yaml
```
