from __future__ import annotations

from io import BytesIO
from urllib.parse import urlencode
from wsgiref.util import setup_testing_defaults

from imdb_sentiment.webapp import ReviewClassifierApp


class DummyModel:
    def __init__(self, predictions: list[int]) -> None:
        self.predictions = predictions

    def predict(self, texts: list[str]) -> list[int]:
        return self.predictions[: len(texts)]


def _run_wsgi_app(
    app: ReviewClassifierApp,
    *,
    method: str,
    path: str,
    form_data: dict[str, str] | None = None,
) -> tuple[str, str]:
    request_body = urlencode(form_data or {}).encode("utf-8")
    environ: dict[str, object] = {}
    setup_testing_defaults(environ)
    environ["REQUEST_METHOD"] = method
    environ["PATH_INFO"] = path
    environ["CONTENT_LENGTH"] = str(len(request_body))
    environ["wsgi.input"] = BytesIO(request_body)

    status: dict[str, str] = {}

    def start_response(status_line, _headers):
        status["line"] = status_line

    body = b"".join(app(environ, start_response)).decode("utf-8")
    return status["line"], body


def test_review_classifier_homepage_renders_form() -> None:
    app = ReviewClassifierApp("artifacts/models/baseline.joblib")
    app._model = DummyModel([1])

    status_line, body = _run_wsgi_app(app, method="GET", path="/")

    assert status_line == "200 OK"
    assert "Detect the type of review" in body
    assert "<form method=\"post\" action=\"/predict\">" in body


def test_review_classifier_predict_route_renders_prediction() -> None:
    app = ReviewClassifierApp("artifacts/models/baseline.joblib")
    app._model = DummyModel([1])

    status_line, body = _run_wsgi_app(
        app,
        method="POST",
        path="/predict",
        form_data={"review_text": "I loved this movie."},
    )

    assert status_line == "200 OK"
    assert "Predicted review type" in body
    assert "Positive" in body
    assert "I loved this movie." in body
