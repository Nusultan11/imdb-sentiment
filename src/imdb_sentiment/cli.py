from imdb_sentiment.settings import load_config


def main() -> None:
    config = load_config()
    print("IMDb sentiment project scaffold is ready.")
    print(f"Model output: {config.paths.model_output}")


if __name__ == "__main__":
    main()
