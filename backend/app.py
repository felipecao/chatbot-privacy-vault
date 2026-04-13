import json
import os
from collections.abc import Iterator

import httpx
from dotenv import load_dotenv
from flask import Flask, Response, request, stream_with_context
from openai import OpenAI

load_dotenv()

MAX_MESSAGES = 50
MAX_TOTAL_CHARS = 100_000

VAULT_URL     = os.environ.get("VAULT_URL", "http://127.0.0.1:5001")
VAULT_TIMEOUT = 10  # seconds

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


def _anonymize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Replace PII in every message with vault tokens before sending to OpenAI.
    All messages (user and assistant) are anonymized so the full conversation
    context reaching OpenAI is free of real PII.
    """
    with httpx.Client(base_url=VAULT_URL, timeout=VAULT_TIMEOUT) as client:
        result = []
        for msg in messages:
            resp = client.post("/anonymize", json={"message": msg["content"]})
            resp.raise_for_status()
            result.append({"role": msg["role"], "content": resp.json()["anonymizedMessage"]})
        return result


def _deanonymize(text: str) -> str:
    """Restore original PII values in the assembled OpenAI response."""
    with httpx.Client(base_url=VAULT_URL, timeout=VAULT_TIMEOUT) as client:
        resp = client.post("/deanonymize", json={"anonymizedMessage": text})
        resp.raise_for_status()
        return resp.json()["message"]


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
            # 1. Anonymize — replace PII with vault tokens in the full conversation.
            anon_messages = _anonymize_messages(messages)

            # 2. Stream the response from OpenAI using anonymized messages.
            stream = client.chat.completions.create(
                model=model,
                messages=anon_messages,
                stream=True,
            )

            # 3. Buffer the complete response — tokens like NAME_abc123 can be
            #    split across chunks, so deanonymization must run on the full text.
            chunks: list[str] = []
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)

            # 4. Deanonymize — restore original PII values in the assembled reply.
            reply = _deanonymize("".join(chunks))

            yield json.dumps({"t": reply}) + "\n"
            yield json.dumps({"done": True}) + "\n"

        except httpx.HTTPError as e:
            yield _ndjson_error_line(f"vault error: {e}")
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
