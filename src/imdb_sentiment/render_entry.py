from __future__ import annotations

import os

from imdb_sentiment.settings import load_config
from imdb_sentiment.webapp import serve_review_classifier


DEFAULT_RENDER_CONFIG = "configs/experiments/tfidf_tuned_v2_final.yaml"
DEFAULT_RENDER_HOST = "0.0.0.0"
DEFAULT_RENDER_PORT = 10000


def _read_render_port() -> int:
    raw_port = os.getenv("PORT")
    if raw_port is None:
        return DEFAULT_RENDER_PORT

    try:
        return int(raw_port)
    except ValueError as exc:
        raise ValueError("PORT environment variable must be an integer.") from exc


def main() -> None:
    config_path = os.getenv("IMDB_CONFIG", DEFAULT_RENDER_CONFIG)
    host = os.getenv("RENDER_HOST", DEFAULT_RENDER_HOST)
    port = _read_render_port()

    config = load_config(config_path)
    serve_review_classifier(config.paths.model_output, host=host, port=port)


if __name__ == "__main__":
    main()
