import json
import os
from collections.abc import Iterator

from dotenv import load_dotenv
from flask import Flask, Response, request, stream_with_context
from openai import OpenAI

load_dotenv()

MAX_MESSAGES = 50
MAX_TOTAL_CHARS = 100_000

app = Flask(__name__)


def _validate_messages(messages: object) -> tuple[list[dict[str, str]] | None, str | None]:
    if not isinstance(messages, list) or not messages:
        return None, "messages must be a non-empty list"
    if len(messages) > MAX_MESSAGES:
        return None, f"too many messages (max {MAX_MESSAGES})"
    total = 0
    normalized: list[dict[str, str]] = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            return None, f"message {i} must be an object"
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant", "system"):
            return None, f"message {i} has invalid role"
        if not isinstance(content, str):
            return None, f"message {i} content must be a string"
        total += len(content)
        normalized.append({"role": role, "content": content})
    if total > MAX_TOTAL_CHARS:
        return None, "total message length too large"
    if normalized[-1]["role"] != "user":
        return None, "last message must be from user"
    return normalized, None


def _ndjson_error_line(message: str) -> str:
    return json.dumps({"error": message}) + "\n"


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/chat")
def chat() -> Response:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _json_error("invalid JSON body", 400)

    messages, err = _validate_messages(payload.get("messages"))
    if err:
        return _json_error(err, 400)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return _json_error("OPENAI_API_KEY is not set", 503)

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    def generate() -> Iterator[str]:
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield json.dumps({"t": delta}) + "\n"
            yield json.dumps({"done": True}) + "\n"
        except Exception as e:
            yield _ndjson_error_line(str(e))

    return Response(
        stream_with_context(generate()),
        mimetype="application/x-ndjson",
        headers={"Cache-Control": "no-cache"},
    )


def _json_error(message: str, status: int) -> Response:
    return Response(
        json.dumps({"error": message}),
        status=status,
        mimetype="application/json",
    )
