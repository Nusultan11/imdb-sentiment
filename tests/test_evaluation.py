import json
from pathlib import Path

from datasets import Dataset, DatasetDict
import joblib
import torch

import imdb_sentiment.pipelines.evaluation as evaluation_module
from imdb_sentiment.data.lstm import build_lstm_vocabulary
from imdb_sentiment.models.baseline import build_baseline_model
from imdb_sentiment.models.lstm.model import build_lstm_model
from imdb_sentiment.settings import load_config


def test_evaluation_resolve_torch_device_prefers_cuda_when_available(monkeypatch) -> None:
    monkeypatch.setattr(evaluation_module.torch.cuda, "is_available", lambda: True)

    device = evaluation_module._resolve_torch_device()

    assert device.type == "cuda"


def test_run_evaluation_writes_test_metrics(tmp_path: Path, monkeypatch) -> None:
    model = build_baseline_model(
        max_features=100,
        ngram_range=(1, 2),
        max_iter=100,
        random_state=42,
    )
    model.fit(
        ["great movie", "excellent acting", "bad plot", "awful ending"],
        [1, 1, 0, 0],
    )

    model_path = tmp_path / "artifacts" / "models" / "baseline.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  family: tfidf",
                "  name: baseline_test",
                "seed: 42",
                "paths:",
                f"  model_output: {model_path.as_posix()}",
                f"  val_metrics_output: {(tmp_path / 'artifacts' / 'reports' / 'val_metrics.json').as_posix()}",
                f"  test_metrics_output: {(tmp_path / 'artifacts' / 'reports' / 'test_metrics.json').as_posix()}",
                "model:",
                "  type: logistic_regression",
                "  max_features: 100",
                "  ngram_range: [1, 2]",
                "  max_iter: 100",
            ]
        ),
        encoding="utf-8",
    )

    fake_dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"text": ["placeholder"], "label": [1]}),
            "test": Dataset.from_dict(
                {
                    "text": ["great film", "terrible movie"],
                    "label": [1, 0],
                }
            ),
        }
    )
    monkeypatch.setattr(evaluation_module, "load_imdb_dataset", lambda: fake_dataset)

    config = load_config(config_path)
    output_path = tmp_path / "artifacts" / "reports" / "test_metrics.json"
    metrics = evaluation_module.run_evaluation(config, output_path=output_path)

    assert {"accuracy", "precision", "recall", "f1"} <= metrics.keys()
    assert output_path.exists()
    saved_metrics = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved_metrics == metrics


def test_run_evaluation_uses_config_test_metrics_output_by_default(tmp_path: Path, monkeypatch) -> None:
    model = build_baseline_model(
        max_features=100,
        ngram_range=(1, 2),
        max_iter=100,
        random_state=42,
    )
    model.fit(
        ["great movie", "excellent acting", "bad plot", "awful ending"],
        [1, 1, 0, 0],
    )

    model_path = tmp_path / "artifacts" / "models" / "baseline.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    test_metrics_output = tmp_path / "artifacts" / "reports" / "from_config_test_metrics.json"
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  family: tfidf",
                "  name: baseline_test",
                "seed: 42",
                "paths:",
                f"  model_output: {model_path.as_posix()}",
                f"  val_metrics_output: {(tmp_path / 'artifacts' / 'reports' / 'val_metrics.json').as_posix()}",
                f"  test_metrics_output: {test_metrics_output.as_posix()}",
                "model:",
                "  type: logistic_regression",
                "  max_features: 100",
                "  ngram_range: [1, 2]",
                "  max_iter: 100",
            ]
        ),
        encoding="utf-8",
    )

    fake_dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"text": ["placeholder"], "label": [1]}),
            "test": Dataset.from_dict(
                {
                    "text": ["great film", "terrible movie"],
                    "label": [1, 0],
                }
            ),
        }
    )
    monkeypatch.setattr(evaluation_module, "load_imdb_dataset", lambda: fake_dataset)

    config = load_config(config_path)
    metrics = evaluation_module.run_evaluation(config)

    assert {"accuracy", "precision", "recall", "f1"} <= metrics.keys()
    assert test_metrics_output.exists()


def test_run_evaluation_supports_lstm_checkpoints(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "lstm.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  family: lstm",
                "  name: baseline_test",
                "seed: 42",
                "paths:",
                f"  model_output: {(tmp_path / 'artifacts' / 'models' / 'model.pt').as_posix()}",
                f"  val_metrics_output: {(tmp_path / 'artifacts' / 'reports' / 'val_metrics.json').as_posix()}",
                f"  test_metrics_output: {(tmp_path / 'artifacts' / 'reports' / 'test_metrics.json').as_posix()}",
                "model:",
                "  type: lstm",
                "  vocab_size: 50",
                "  max_length: 6",
                "  embedding_dim: 8",
                "  hidden_dim: 6",
                "  bidirectional: false",
                "  batch_size: 2",
                "  epochs: 2",
                "  dropout: 0.2",
                "  lr: 0.01",
            ]
        ),
        encoding="utf-8",
    )

    fake_dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "text": ["great movie", "bad ending"],
                    "label": [1, 0],
                }
            ),
            "test": Dataset.from_dict(
                {
                    "text": ["great film", "awful film"],
                    "label": [1, 0],
                }
            ),
        }
    )
    monkeypatch.setattr(evaluation_module, "load_imdb_dataset", lambda: fake_dataset)

    config = load_config(config_path)
    vocabulary = build_lstm_vocabulary(["great movie", "bad ending"], max_size=50)
    model = build_lstm_model(config.model)
    config.paths.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocabulary": vocabulary.token_to_id,
            "max_length": config.model.max_length,
            "family": config.experiment.family,
            "name": config.experiment.name,
        },
        config.paths.model_output,
    )
    (config.paths.model_output.parent / "vocab.json").write_text(
        json.dumps(vocabulary.token_to_id, indent=2),
        encoding="utf-8",
    )
    (config.paths.model_output.parent / "training_config.json").write_text(
        json.dumps(
            {
                "experiment": {
                    "family": config.experiment.family,
                    "name": config.experiment.name,
                },
                "seed": config.seed,
                "model": {
                    "type": "lstm",
                    "vocab_size": config.model.vocab_size,
                    "max_length": config.model.max_length,
                    "embedding_dim": config.model.embedding_dim,
                    "hidden_dim": config.model.hidden_dim,
                    "batch_size": config.model.batch_size,
                    "epochs": config.model.epochs,
                    "dropout": config.model.dropout,
                    "lr": config.model.lr,
                },
                "artifacts": {
                    "model_output": "model.pt",
                    "vocab_output": "vocab.json",
                    "training_config_output": "training_config.json",
                    "val_metrics_output": "val_metrics.json",
                    "test_metrics_output": "test_metrics.json",
                },
                "required_for_inference": [
                    "model.pt",
                    "vocab.json",
                    "training_config.json",
                ],
                "required_for_evaluation": [
                    "model.pt",
                    "vocab.json",
                    "training_config.json",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    metrics = evaluation_module.run_evaluation(config)

    assert {"accuracy", "precision", "recall", "f1"} <= metrics.keys()
    assert config.paths.test_metrics_output.exists()
    saved_metrics = json.loads(config.paths.test_metrics_output.read_text(encoding="utf-8"))
    assert saved_metrics == metrics


def test_run_evaluation_uses_saved_lstm_decision_threshold(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "lstm_threshold.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  family: lstm",
                "  name: threshold_eval_test",
                "seed: 42",
                "paths:",
                f"  model_output: {(tmp_path / 'artifacts' / 'models' / 'model.pt').as_posix()}",
                f"  val_metrics_output: {(tmp_path / 'artifacts' / 'reports' / 'val_metrics.json').as_posix()}",
                f"  test_metrics_output: {(tmp_path / 'artifacts' / 'reports' / 'test_metrics.json').as_posix()}",
                "model:",
                "  type: lstm",
                "  vocab_size: 50",
                "  max_length: 6",
                "  embedding_dim: 8",
                "  hidden_dim: 6",
                "  bidirectional: false",
                "  batch_size: 2",
                "  epochs: 2",
                "  dropout: 0.2",
                "  lr: 0.01",
            ]
        ),
        encoding="utf-8",
    )

    fake_dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "text": ["placeholder train row"],
                    "label": [1],
                }
            ),
            "test": Dataset.from_dict(
                {
                    "text": ["great film", "awful film"],
                    "label": [1, 0],
                }
            ),
        }
    )
    monkeypatch.setattr(evaluation_module, "load_imdb_dataset", lambda: fake_dataset)

    config = load_config(config_path)
    vocabulary = build_lstm_vocabulary(["great movie", "awful ending"], max_size=50)
    config.paths.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {},
            "vocabulary": vocabulary.token_to_id,
            "max_length": config.model.max_length,
            "family": config.experiment.family,
            "name": config.experiment.name,
        },
        config.paths.model_output,
    )
    (config.paths.model_output.parent / "vocab.json").write_text(
        json.dumps(vocabulary.token_to_id, indent=2),
        encoding="utf-8",
    )
    (config.paths.model_output.parent / "training_config.json").write_text(
        json.dumps(
            {
                "experiment": {
                    "family": config.experiment.family,
                    "name": config.experiment.name,
                },
                "seed": config.seed,
                "model": {
                    "type": "lstm",
                    "vocab_size": config.model.vocab_size,
                    "max_length": config.model.max_length,
                    "embedding_dim": config.model.embedding_dim,
                    "hidden_dim": config.model.hidden_dim,
                    "batch_size": config.model.batch_size,
                    "epochs": config.model.epochs,
                    "dropout": config.model.dropout,
                    "lr": config.model.lr,
                },
                "artifacts": {
                    "model_output": "model.pt",
                    "vocab_output": "vocab.json",
                    "training_config_output": "training_config.json",
                    "val_metrics_output": "val_metrics.json",
                    "test_metrics_output": "test_metrics.json",
                },
                "required_for_inference": [
                    "model.pt",
                    "vocab.json",
                    "training_config.json",
                ],
                "required_for_evaluation": [
                    "model.pt",
                    "vocab.json",
                    "training_config.json",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (config.paths.model_output.parent / "threshold_tuning.json").write_text(
        json.dumps(
            {
                "decision_threshold": 0.6,
                "selection_strategy": "validation_best_f1",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        evaluation_module,
        "build_lstm_dataloader",
        lambda **kwargs: [
            (
                torch.tensor([[1, 2, 0], [1, 2, 0]], dtype=torch.long),
                torch.tensor([1.0, 0.0], dtype=torch.float32),
            )
        ],
    )

    class _FakeEvaluationModel(torch.nn.Module):
        def load_state_dict(self, state_dict) -> None:
            self._loaded_state_dict = state_dict

        def to(self, device):
            return self

        def eval(self):
            return self

        def forward(self, token_ids):  # noqa: D401 - tiny test double
            del token_ids
            return torch.tensor([0.6, 0.2], dtype=torch.float32)

    monkeypatch.setattr(
        evaluation_module,
        "build_lstm_model",
        lambda model_config: _FakeEvaluationModel(),
    )

    metrics = evaluation_module.run_evaluation(config)

    assert metrics == {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }


def test_run_evaluation_uses_saved_lstm_preprocessing_from_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "lstm_regex_eval.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  family: lstm",
                "  name: regex_eval_test",
                "seed: 42",
                "paths:",
                f"  model_output: {(tmp_path / 'artifacts' / 'models' / 'model.pt').as_posix()}",
                f"  val_metrics_output: {(tmp_path / 'artifacts' / 'reports' / 'val_metrics.json').as_posix()}",
                f"  test_metrics_output: {(tmp_path / 'artifacts' / 'reports' / 'test_metrics.json').as_posix()}",
                "model:",
                "  type: lstm",
                "  vocab_size: 50",
                "  max_length: 6",
                "  embedding_dim: 8",
                "  hidden_dim: 6",
                "  bidirectional: false",
                "  pooling: last_hidden",
                "  batch_size: 2",
                "  epochs: 2",
                "  dropout: 0.2",
                "  lr: 0.01",
            ]
        ),
        encoding="utf-8",
    )

    fake_dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "text": ["placeholder train row"],
                    "label": [1],
                }
            ),
            "test": Dataset.from_dict(
                {
                    "text": ["worst-film-ever", "Amazing!!! movie"],
                    "label": [0, 1],
                }
            ),
        }
    )
    monkeypatch.setattr(evaluation_module, "load_imdb_dataset", lambda: fake_dataset)

    config = load_config(config_path)
    vocabulary = build_lstm_vocabulary(
        ["don't like it", "worst-film-ever", "Amazing!!! movie"],
        max_size=50,
        preprocessing="regex_v2",
    )
    config.paths.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {},
            "vocabulary": vocabulary.token_to_id,
            "max_length": 6,
            "family": config.experiment.family,
            "name": config.experiment.name,
        },
        config.paths.model_output,
    )
    (config.paths.model_output.parent / "vocab.json").write_text(
        json.dumps(vocabulary.token_to_id, indent=2),
        encoding="utf-8",
    )
    (config.paths.model_output.parent / "training_config.json").write_text(
        json.dumps(
            {
                "experiment": {
                    "family": config.experiment.family,
                    "name": config.experiment.name,
                },
                "seed": config.seed,
                "model": {
                    "type": "lstm",
                    "vocab_size": 50,
                    "max_length": 6,
                    "embedding_dim": 8,
                    "hidden_dim": 6,
                    "batch_size": 2,
                    "epochs": 2,
                    "dropout": 0.2,
                    "lr": 0.01,
                    "bidirectional": False,
                    "pooling": "last_hidden",
                    "preprocessing": "regex_v2",
                },
                "artifacts": {
                    "model_output": "model.pt",
                    "vocab_output": "vocab.json",
                    "training_config_output": "training_config.json",
                    "threshold_tuning_output": "threshold_tuning.json",
                    "val_metrics_output": "val_metrics.json",
                    "test_metrics_output": "test_metrics.json",
                },
                "required_for_inference": [
                    "model.pt",
                    "vocab.json",
                    "training_config.json",
                ],
                "required_for_evaluation": [
                    "model.pt",
                    "vocab.json",
                    "training_config.json",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (config.paths.model_output.parent / "threshold_tuning.json").write_text(
        json.dumps(
            {
                "decision_threshold": 0.5,
                "selection_strategy": "validation_best_f1",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    dataloader_calls: list[dict[str, object]] = []

    def _recording_build_lstm_dataloader(**kwargs):
        dataloader_calls.append(kwargs)
        return [
            (
                torch.tensor([[1, 2, 3, 0, 0, 0], [4, 5, 6, 0, 0, 0]], dtype=torch.long),
                torch.tensor([0.0, 1.0], dtype=torch.float32),
            )
        ]

    monkeypatch.setattr(
        evaluation_module,
        "build_lstm_dataloader",
        _recording_build_lstm_dataloader,
    )

    class _FakeEvaluationModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.loaded_state_dict: dict[str, object] | None = None

        def load_state_dict(self, state_dict) -> None:
            self.loaded_state_dict = state_dict

        def to(self, device):
            return self

        def eval(self):
            return self

        def forward(self, token_ids):  # noqa: D401 - tiny test double
            del token_ids
            return torch.tensor([-0.2, 0.3], dtype=torch.float32)

    built_model_configs: list[object] = []

    def _recording_build_lstm_model(model_config):
        built_model_configs.append(model_config)
        return _FakeEvaluationModel()

    monkeypatch.setattr(
        evaluation_module,
        "build_lstm_model",
        _recording_build_lstm_model,
    )

    metrics = evaluation_module.run_evaluation(config)

    assert {"accuracy", "precision", "recall", "f1"} <= metrics.keys()
    assert dataloader_calls[0]["preprocessing"] == "regex_v2"
    assert built_model_configs[0].preprocessing == "regex_v2"


def test_run_evaluation_ignores_external_lstm_yaml_model_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_config_path = tmp_path / "lstm_checkpoint.yaml"
    checkpoint_config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  family: lstm",
                "  name: checkpoint_eval_test",
                "seed: 42",
                "paths:",
                f"  model_output: {(tmp_path / 'artifacts' / 'models' / 'model.pt').as_posix()}",
                f"  val_metrics_output: {(tmp_path / 'artifacts' / 'reports' / 'val_metrics.json').as_posix()}",
                f"  test_metrics_output: {(tmp_path / 'artifacts' / 'reports' / 'test_metrics.json').as_posix()}",
                "model:",
                "  type: lstm",
                "  vocab_size: 50",
                "  max_length: 6",
                "  embedding_dim: 8",
                "  hidden_dim: 6",
                "  bidirectional: true",
                "  pooling: masked_mean",
                "  batch_size: 2",
                "  epochs: 2",
                "  dropout: 0.2",
                "  lr: 0.01",
            ]
        ),
        encoding="utf-8",
    )
    checkpoint_config = load_config(checkpoint_config_path)

    wrong_external_config_path = tmp_path / "lstm_external_wrong.yaml"
    wrong_external_config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  family: lstm",
                "  name: wrong_external_eval_yaml",
                "seed: 42",
                "paths:",
                f"  model_output: {checkpoint_config.paths.model_output.as_posix()}",
                f"  val_metrics_output: {(tmp_path / 'other' / 'val_metrics.json').as_posix()}",
                f"  test_metrics_output: {(tmp_path / 'other' / 'test_metrics.json').as_posix()}",
                "model:",
                "  type: lstm",
                "  vocab_size: 50",
                "  max_length: 6",
                "  embedding_dim: 8",
                "  hidden_dim: 4",
                "  bidirectional: false",
                "  pooling: last_hidden",
                "  preprocessing: whitespace_v1",
                "  batch_size: 2",
                "  epochs: 2",
                "  dropout: 0.2",
                "  lr: 0.01",
            ]
        ),
        encoding="utf-8",
    )
    wrong_external_config = load_config(wrong_external_config_path)

    fake_dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "text": ["placeholder train row"],
                    "label": [1],
                }
            ),
            "test": Dataset.from_dict(
                {
                    "text": ["great film", "awful film"],
                    "label": [1, 0],
                }
            ),
        }
    )
    monkeypatch.setattr(evaluation_module, "load_imdb_dataset", lambda: fake_dataset)

    vocabulary = build_lstm_vocabulary(
        ["great movie", "awful ending"],
        max_size=50,
        preprocessing="regex_v2",
    )
    checkpoint_config.paths.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {},
            "vocabulary": vocabulary.token_to_id,
            "max_length": checkpoint_config.model.max_length,
            "family": checkpoint_config.experiment.family,
            "name": checkpoint_config.experiment.name,
        },
        checkpoint_config.paths.model_output,
    )
    (checkpoint_config.paths.model_output.parent / "vocab.json").write_text(
        json.dumps(vocabulary.token_to_id, indent=2),
        encoding="utf-8",
    )
    (checkpoint_config.paths.model_output.parent / "training_config.json").write_text(
        json.dumps(
            {
                "experiment": {
                    "family": checkpoint_config.experiment.family,
                    "name": checkpoint_config.experiment.name,
                },
                "seed": checkpoint_config.seed,
                "model": {
                    "type": "lstm",
                    "vocab_size": 50,
                    "max_length": 6,
                    "embedding_dim": 8,
                    "hidden_dim": 6,
                    "batch_size": 2,
                    "epochs": 2,
                    "dropout": 0.2,
                    "lr": 0.01,
                    "bidirectional": True,
                    "pooling": "masked_mean",
                    "preprocessing": "regex_v2",
                },
                "artifacts": {
                    "model_output": "model.pt",
                    "vocab_output": "vocab.json",
                    "training_config_output": "training_config.json",
                    "threshold_tuning_output": "threshold_tuning.json",
                    "val_metrics_output": "val_metrics.json",
                    "test_metrics_output": "test_metrics.json",
                },
                "required_for_inference": [
                    "model.pt",
                    "vocab.json",
                    "training_config.json",
                ],
                "required_for_evaluation": [
                    "model.pt",
                    "vocab.json",
                    "training_config.json",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (checkpoint_config.paths.model_output.parent / "threshold_tuning.json").write_text(
        json.dumps(
            {
                "decision_threshold": 0.5,
                "selection_strategy": "validation_best_f1",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        evaluation_module,
        "build_lstm_dataloader",
        lambda **kwargs: [
            (
                torch.tensor([[1, 2, 0], [1, 2, 0]], dtype=torch.long),
                torch.tensor([1.0, 0.0], dtype=torch.float32),
            )
        ],
    )

    built_model_configs: list[object] = []

    class _FakeEvaluationModel(torch.nn.Module):
        def load_state_dict(self, state_dict) -> None:
            self.loaded_state_dict = state_dict

        def to(self, device):
            return self

        def eval(self):
            return self

        def forward(self, token_ids):  # noqa: D401 - tiny test double
            del token_ids
            return torch.tensor([0.2, -0.2], dtype=torch.float32)

    def _recording_build_lstm_model(model_config):
        built_model_configs.append(model_config)
        return _FakeEvaluationModel()

    monkeypatch.setattr(
        evaluation_module,
        "build_lstm_model",
        _recording_build_lstm_model,
    )

    metrics = evaluation_module.run_evaluation(wrong_external_config)

    assert {"accuracy", "precision", "recall", "f1"} <= metrics.keys()
    assert built_model_configs[0].hidden_dim == 6
    assert built_model_configs[0].bidirectional is True
    assert built_model_configs[0].pooling == "masked_mean"
    assert built_model_configs[0].preprocessing == "regex_v2"
