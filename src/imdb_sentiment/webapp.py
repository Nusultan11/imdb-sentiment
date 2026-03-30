from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server

from sklearn.pipeline import Pipeline

from imdb_sentiment.inference.predict import load_model, predict_texts


PREDICTION_LABELS = {
    0: "Negative",
    1: "Positive",
}


class ReviewClassifierApp:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self._model: Pipeline | None = None

    def _get_model(self) -> Pipeline:
        if self._model is None:
            self._model = load_model(self.model_path)
        return self._model

    def predict_review_type(self, review_text: str) -> str:
        normalized_text = review_text.strip()
        if not normalized_text:
            raise ValueError("Review text must not be empty.")

        prediction = predict_texts(self._get_model(), [normalized_text])[0]
        return PREDICTION_LABELS[prediction]

    def render_page(
        self,
        review_text: str = "",
        prediction: str | None = None,
        error: str | None = None,
    ) -> bytes:
        escaped_review = escape(review_text)
        escaped_prediction = escape(prediction) if prediction is not None else ""
        escaped_error = escape(error) if error is not None else ""
        result_block = ""

        if prediction is not None:
            result_block = (
                "<section class='card result'>"
                "<h2>Predicted review type</h2>"
                f"<p class='badge'>{escaped_prediction}</p>"
                "</section>"
            )
        elif error is not None:
            result_block = (
                "<section class='card result error'>"
                "<h2>Prediction error</h2>"
                f"<p>{escaped_error}</p>"
                "</section>"
            )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>IMDb Review Type</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f1e8;
      --panel: #fffaf2;
      --ink: #1b1f23;
      --muted: #5f6b76;
      --accent: #c8553d;
      --accent-soft: #f3d7cf;
      --border: #e3d5c5;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top left, #f7dfc8 0, transparent 26%),
        linear-gradient(180deg, #f4ede1 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    main {{
      max-width: 780px;
      margin: 0 auto;
      padding: 48px 20px 64px;
    }}
    .hero {{
      margin-bottom: 28px;
    }}
    .eyebrow {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    h1 {{
      font-size: clamp(34px, 6vw, 58px);
      line-height: 0.95;
      margin: 14px 0 12px;
    }}
    .lead {{
      max-width: 56ch;
      color: var(--muted);
      font-size: 18px;
      line-height: 1.6;
    }}
    .card {{
      background: rgba(255, 250, 242, 0.92);
      border: 1px solid var(--border);
      border-radius: 24px;
      padding: 24px;
      box-shadow: 0 18px 40px rgba(81, 59, 40, 0.08);
      backdrop-filter: blur(8px);
    }}
    form {{
      display: grid;
      gap: 16px;
    }}
    label {{
      font-weight: 700;
    }}
    textarea {{
      width: 100%;
      min-height: 220px;
      resize: vertical;
      border-radius: 18px;
      border: 1px solid var(--border);
      padding: 18px;
      font: inherit;
      color: inherit;
      background: #fffdf8;
    }}
    button {{
      justify-self: start;
      border: 0;
      border-radius: 999px;
      padding: 14px 22px;
      font: inherit;
      font-weight: 700;
      color: white;
      background: linear-gradient(135deg, #b84a33 0%, var(--accent) 100%);
      cursor: pointer;
    }}
    .result {{
      margin-top: 18px;
    }}
    .result.error {{
      border-color: #d27a6d;
    }}
    .badge {{
      display: inline-block;
      margin: 0;
      padding: 10px 16px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <span class="eyebrow">IMDb sentiment</span>
      <h1>Detect the type of review</h1>
      <p class="lead">
        Paste an IMDb-style review below and the site will classify it as a positive or negative review.
      </p>
    </section>
    <section class="card">
      <form method="post" action="/predict">
        <label for="review_text">Review text</label>
        <textarea id="review_text" name="review_text" placeholder="Write or paste a review here...">{escaped_review}</textarea>
        <button type="submit">Classify review</button>
      </form>
    </section>
    {result_block}
  </main>
</body>
</html>
"""
        return html.encode("utf-8")

    def __call__(
        self,
        environ: dict[str, Any],
        start_response,
    ) -> list[bytes]:
        method = environ.get("REQUEST_METHOD", "GET").upper()
        path = environ.get("PATH_INFO", "/")

        if method == "GET" and path == "/":
            start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
            return [self.render_page()]

        if method == "POST" and path == "/predict":
            body_size = int(environ.get("CONTENT_LENGTH") or "0")
            body = environ["wsgi.input"].read(body_size).decode("utf-8")
            form_data = parse_qs(body)
            review_text = form_data.get("review_text", [""])[0]

            try:
                prediction = self.predict_review_type(review_text)
                response_body = self.render_page(review_text=review_text, prediction=prediction)
                start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
                return [response_body]
            except Exception as exc:
                response_body = self.render_page(review_text=review_text, error=str(exc))
                start_response("400 Bad Request", [("Content-Type", "text/html; charset=utf-8")])
                return [response_body]

        start_response("404 Not Found", [("Content-Type", "text/plain; charset=utf-8")])
        return [b"Not found"]


def serve_review_classifier(
    model_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    app = ReviewClassifierApp(model_path)
    with make_server(host, port, app) as server:
        print(f"Serving review classifier on http://{host}:{port}")
        server.serve_forever()
