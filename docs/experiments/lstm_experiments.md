# LSTM Experiments Summary

## 1. Goal

- Task: binary sentiment classification on IMDb reviews.
- Project strategy: compare several model families, with TF-IDF serving as the baseline family and LSTM explored as the neural branch.
- Evaluation rule for this branch: keep the test split untouched during model iteration and compare LSTM variants on validation only.
- LSTM branch question: does a stronger sequence encoder, better pooling, and hyperparameter tuning improve validation F1 over the original LSTM baseline?

## Quick Comparison Table

| Experiment | Main change | Val F1 | Notes |
|---|---|---|---|
| `lstm_baseline_v1` | 1-layer LSTM, `last_hidden` pooling | `0.8377` | baseline |
| `lstm_bidirectional_v1` | bidirectional encoder | `0.8293` | better context, but not better F1 |
| `lstm_bidirectional_masked_mean_v1` | `masked_mean` pooling | `n/a` | stronger sequence aggregation, separate branch prepared |
| `lstm_bidirectional_masked_mean_optuna_v1` | Optuna-tuned hyperparameters on BiLSTM + `masked_mean` | `0.8952` | current best archived LSTM |

## 2. LSTM Evolution

### `lstm_baseline_v1`

- Architecture: unidirectional LSTM with `last_hidden` pooling.
- Config: [lstm_baseline_v1.yaml](/C:/Users/nurs/OneDrive/Рабочий%20стол/imdb-sentiment/configs/experiments/lstm_baseline_v1.yaml)
- Archived validation result:
  - best epoch: `4`
  - val accuracy: `0.8408`
  - val precision: `0.8565`
  - val recall: `0.8196`
  - val f1: `0.8377`

### `lstm_bidirectional_v1`

- Architecture: bidirectional LSTM with `last_hidden` pooling.
- Config: [lstm_bidirectional_v1.yaml](/C:/Users/nurs/OneDrive/Рабочий%20стол/imdb-sentiment/configs/experiments/lstm_bidirectional_v1.yaml)
- Archived validation result:
  - best epoch: `5`
  - val accuracy: `0.8418`
  - val precision: `0.9031`
  - val recall: `0.7666`
  - val f1: `0.8293`

### `lstm_bidirectional_masked_mean_v1`

- Architecture: bidirectional LSTM with `masked_mean` pooling.
- Config: [lstm_bidirectional_masked_mean_v1.yaml](/C:/Users/nurs/OneDrive/Рабочий%20стол/imdb-sentiment/configs/experiments/lstm_bidirectional_masked_mean_v1.yaml)
- What changed relative to `bidirectional_v1`:
  - kept bidirectional encoding
  - replaced `last_hidden` pooling with `masked_mean`
  - separated the experiment into its own config and artifact directory for honest validation comparison
- Validation status in the current repo snapshot:
  - architecture and experiment contract are in place
  - a standalone archived `masked_mean_v1` validation bundle is not stored separately here
  - the next archived result for this branch is the Optuna-tuned continuation below

### `lstm_bidirectional_masked_mean_optuna_v1`

- Architecture family: bidirectional LSTM with `masked_mean` pooling plus Optuna tuning over training hyperparameters.
- External artifacts stored in:
  - [bidirectional_masked_mean_v1](/C:/Users/nurs/OneDrive/Рабочий%20стол/imdb-sentiment/artifacts/experiments/lstm/bidirectional_masked_mean_v1)
- Notebook export:
  - [bilstm+mean_polling.ipynb](/C:/Users/nurs/OneDrive/Рабочий%20стол/imdb-sentiment/artifacts/experiments/lstm/bidirectional_masked_mean_v1/bilstm+mean_polling.ipynb)
- Archived artifact bundle:
  - [bidirectional_masked_mean_optuna_v1_artifacts.zip](/C:/Users/nurs/OneDrive/Рабочий%20стол/imdb-sentiment/artifacts/experiments/lstm/bidirectional_masked_mean_v1/bidirectional_masked_mean_optuna_v1_artifacts.zip)

## 3. What Changed Across Versions

| Version | What changed | Why | Validation result |
|---|---|---|---|
| `lstm_baseline_v1` | Started with a unidirectional LSTM and `last_hidden` pooling. | Establish a clean LSTM baseline before architectural changes. | `val_f1 = 0.8377` |
| `lstm_bidirectional_v1` | Switched encoder to bidirectional while keeping `last_hidden` pooling. | Test whether richer left-right context helps IMDb sentiment classification. | `val_f1 = 0.8293` |
| `lstm_bidirectional_masked_mean_v1` | Changed pooling from `last_hidden` to `masked_mean` and kept the experiment separate. | Make pooling less sensitive to the final timestep and compare the branch honestly on validation. | Separate archived validation metrics are not stored here; this branch continues into the tuned Optuna run. |
| `lstm_bidirectional_masked_mean_optuna_v1` | Tuned `embedding_dim`, `hidden_dim`, `dropout`, `lr`, and `batch_size` with Optuna on top of the masked-mean BiLSTM branch. | Push the strongest LSTM branch further after the pooling change. | Optuna study best trial reached `val_f1 = 0.8984`; the archived final retrain bundle reports `val_f1 = 0.8952`. |

## 4. Current Best Result

Current best LSTM branch in this repo snapshot: `bidirectional_masked_mean_optuna_v1`

- Architecture:
  - model type: `lstm`
  - bidirectional: `true`
  - pooling: `masked_mean`
  - embedding dim: `128`
  - hidden dim: `128`
  - batch size: `16`
  - dropout: `0.5`
  - learning rate: `0.002074566765675252`
  - max length: `512`
- Best epoch in the archived final training run: `2`
- Validation metrics from the archived final artifact:
  - val accuracy: `0.8954`
  - val precision: `0.8990`
  - val recall: `0.8915`
  - val f1: `0.8952`
- Chosen threshold in the archived final artifact:
  - `0.86`
- Extra recruiter-friendly note:
  - the Optuna notebook export also records a best study trial with `val_f1 = 0.8984`, `best_epoch = 2`, and `best_threshold = 0.43`
  - the zipped production-style artifact that was saved after retraining currently reports the slightly lower but still strongest archived LSTM result `val_f1 = 0.8952`

## 5. Reading Guide

- If someone wants the full story quickly:
  - baseline LSTM established the neural reference point
  - simply making the encoder bidirectional did not improve validation F1
  - changing pooling to `masked_mean` defined the more promising branch
  - hyperparameter tuning on the masked-mean BiLSTM produced the strongest LSTM validation result in the project so far
