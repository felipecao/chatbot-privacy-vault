"""
Data Privacy Vault — Anonymization Service
==========================================
Default port: 5001

Endpoints
---------
POST /anonymize
    Body:     {"message": "<text containing PII>"}
    Response: {"anonymizedMessage": "<text with PII replaced by tokens>"}

POST /deanonymize
    Body:     {"anonymizedMessage": "<text containing tokens>"}
    Response: {"message": "<text with original PII restored>"}

GET /health
    Response: {"ok": true}

PII detected
------------
- Email addresses  — regex
- Phone numbers    — regex
- Person names     — spaCy NER (PERSON entities) with a capitalized-word
                     heuristic as a complementary pass for non-English text

Each word of a multi-word name is replaced by its own independent token.
The same PII value always maps to the same token (idempotent, backed by MongoDB).

MongoDB collection: vault_entries
----------------------------------
{
  "token": "NAME_5a1fe53e9b67",   # unique index — deanonymize lookup key
  "pii":   "Dago",                # unique index — idempotency key
  "type":  "NAME",                # NAME | EMAIL | PHONE
  "created_at": ISODate
}

Environment variables
---------------------
SPACY_MODEL   spaCy model to load (default: en_core_web_sm).
              For Spanish text use: es_core_news_sm
              Install with: python -m spacy download <model>
MONGO_URI     MongoDB connection string (default: mongodb://admin:secret@localhost:27017/).
MONGO_DB      Database name (default: privacy_vault).
"""

import logging
import os
import re
import uuid
from datetime import datetime, timezone

import spacy
from flask import Flask, jsonify, request
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError
from pymongo.collection import ReturnDocument

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# MongoDB — vault_entries collection
# ---------------------------------------------------------------------------

_MONGO_URI = os.getenv("MONGO_URI", "mongodb://admin:secret@localhost:27017/")
_MONGO_DB  = os.getenv("MONGO_DB",  "privacy_vault")

_mongo  = MongoClient(_MONGO_URI)
_db     = _mongo[_MONGO_DB]
_vault  = _db["vault_entries"]

# Ensure uniqueness on both lookup paths.
_vault.create_index([("token", ASCENDING)], unique=True, background=True)
_vault.create_index([("pii",   ASCENDING)], unique=True, background=True)

logger.info("Connected to MongoDB: db=%s collection=vault_entries", _MONGO_DB)

# Matches any vault token in a piece of text, used during deanonymization.
_TOKEN_RE = re.compile(r"\b(NAME|EMAIL|PHONE)_[0-9a-f]{12}\b")


def _get_or_create_token(value: str, prefix: str) -> str:
    """
    Return the existing token for *value*, or atomically mint and store a new one.

    Uses find_one_and_update with upsert=True and $setOnInsert so that:
    - First call for a given PII value → inserts a new doc, returns new token.
    - Subsequent calls for the same value → returns existing token unchanged.
    """
    new_token = f"{prefix}_{uuid.uuid4().hex[:12]}"
    doc = _vault.find_one_and_update(
        filter={"pii": value},
        update={"$setOnInsert": {
            "token":      new_token,
            "pii":        value,
            "type":       prefix,
            "created_at": datetime.now(timezone.utc),
        }},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return doc["token"]


# ---------------------------------------------------------------------------
# NLP model
# ---------------------------------------------------------------------------

_SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")

try:
    _nlp = spacy.load(_SPACY_MODEL)
    logger.info("Loaded spaCy model '%s'", _SPACY_MODEL)
except OSError:
    logger.warning(
        "spaCy model '%s' not found — NER-based name detection disabled. "
        "Install with: python -m spacy download %s",
        _SPACY_MODEL,
        _SPACY_MODEL,
    )
    _nlp = None

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)

# Matches common phone formats:
#   International  +xx xxx-xxxx
#   US-style       (xxx) xxx-xxxx  /  xxx-xxx-xxxx
#   Plain digits   10-15 consecutive digits (e.g. Colombian mobile numbers)
#
# Each alternative starts with a non-space character ('+', '(', or digit)
# so the match never swallows a preceding space.
# Lookbehind/lookahead prevent matching a substring of a longer digit run.
_PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:"
    r"\+\d{1,3}[\s.\-]?\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}"  # +x (xxx) xxx-xxxx
    r"|\+\d{1,3}(?:[\s.\-]\d{2,6}){2,4}"                      # +xx xxx xxxx (grouped)
    r"|\+\d{1,3}[\s.\-]?\d{6,12}"                             # +xx xxxxxxxxx (compact)
    r"|\(\d{3}\)[\s.\-]?\d{3}[\s.\-]\d{4}"                   # (xxx) xxx-xxxx
    r"|\d{3}[\s.\-]\d{3}[\s.\-]\d{4}"                        # xxx-xxx-xxxx
    r"|\d{10,15}"                                              # plain 10-15 digits
    r")"
    r"(?!\d)"
)

# Two or more consecutive Title-Case words (including accented characters).
# Language-agnostic heuristic that complements spaCy for non-English input.
_NAME_RE = re.compile(
    r"\b[A-ZÁÉÍÓÚÀÈÌÒÙÄËÏÖÜÑ][a-záéíóúàèìòùäëïöüñ]+"
    r"(?:\s+[A-ZÁÉÍÓÚÀÈÌÒÙÄËÏÖÜÑ][a-záéíóúàèìòùäëïöüñ]+)+\b"
)

# ---------------------------------------------------------------------------
# Core anonymization logic
# ---------------------------------------------------------------------------

Spans = list[tuple[int, int, str]]  # (start, end, replacement_token)


def _overlaps(start: int, end: int, spans: Spans) -> bool:
    return any(s < end and start < e for s, e, _ in spans)


def _add_word_spans(text: str, entity_text: str, entity_start: int, spans: Spans) -> None:
    """
    Split a detected name into individual words and append one span per word.
    Each word receives its own independent token.
    """
    cursor = entity_start
    for word in entity_text.split():
        try:
            word_start = text.index(word, cursor)
        except ValueError:
            continue  # safety: skip if position tracking drifts
        word_end = word_start + len(word)
        if not _overlaps(word_start, word_end, spans):
            spans.append((word_start, word_end, _get_or_create_token(word, "NAME")))
        cursor = word_end


def _build_result(text: str, spans: Spans) -> str:
    """Rebuild *text* with every span replaced by its token."""
    parts: list[str] = []
    cursor = 0
    for start, end, token in sorted(spans, key=lambda x: x[0]):
        parts.append(text[cursor:start])
        parts.append(token)
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def _anonymize(message: str) -> str:
    spans: Spans = []

    # Pass 1 — Emails (before phones to prevent digits in email addresses
    #           from being matched by the phone pattern)
    for m in _EMAIL_RE.finditer(message):
        spans.append((m.start(), m.end(), _get_or_create_token(m.group(), "EMAIL")))

    # Pass 2 — Phone numbers
    for m in _PHONE_RE.finditer(message):
        if not _overlaps(m.start(), m.end(), spans):
            spans.append((m.start(), m.end(), _get_or_create_token(m.group(), "PHONE")))

    # Pass 3 — Person names via spaCy NER
    # Only accept PERSON entities where every alphabetic word is Title-Case.
    # This filters out false positives where the English model mis-classifies
    # lowercase foreign words (e.g. Spanish "oferta", "trabajo") as names.
    if _nlp is not None:
        doc = _nlp(message)
        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue
            words = [w for w in ent.text.split() if w.isalpha()]
            if not words or not all(w[0].isupper() for w in words):
                continue
            if not _overlaps(ent.start_char, ent.end_char, spans):
                _add_word_spans(message, ent.text, ent.start_char, spans)

    # Pass 4 — Capitalized-word heuristic (catches names missed by spaCy,
    #           e.g. when surrounding context is non-English)
    for m in _NAME_RE.finditer(message):
        if not _overlaps(m.start(), m.end(), spans):
            _add_word_spans(message, m.group(), m.start(), spans)

    return _build_result(message, spans)


def _deanonymize(message: str) -> str:
    """
    Replace every vault token in *message* with its original PII value.

    Collects all tokens in one pass, fetches them from MongoDB in a single
    query, then substitutes in a second pass — one round-trip regardless of
    how many tokens are present.
    """
    token_set = {m.group() for m in _TOKEN_RE.finditer(message)}
    if not token_set:
        return message

    docs = _vault.find({"token": {"$in": list(token_set)}}, {"token": 1, "pii": 1})
    lookup = {doc["token"]: doc["pii"] for doc in docs}

    return _TOKEN_RE.sub(lambda m: lookup.get(m.group(), m.group()), message)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.post("/anonymize")
def anonymize():
    body = request.get_json(silent=True)
    if not body or "message" not in body:
        return jsonify({"error": "'message' field is required"}), 400
    if not isinstance(body["message"], str):
        return jsonify({"error": "'message' must be a string"}), 400

    return jsonify({"anonymizedMessage": _anonymize(body["message"])})


@app.post("/deanonymize")
def deanonymize():
    body = request.get_json(silent=True)
    if not body or "anonymizedMessage" not in body:
        return jsonify({"error": "'anonymizedMessage' field is required"}), 400
    if not isinstance(body["anonymizedMessage"], str):
        return jsonify({"error": "'anonymizedMessage' must be a string"}), 400

    return jsonify({"message": _deanonymize(body["anonymizedMessage"])})
