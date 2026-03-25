from imdb_sentiment.pipelines.train import run_training
from imdb_sentiment.settings import load_config


def main() -> None:
    config = load_config()
    metrics = run_training(config)
    print("Training finished.")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Model saved to: {config.paths.model_output}")
    print(f"Metrics saved to: {config.paths.metrics_output}")


if __name__ == "__main__":
    main()
