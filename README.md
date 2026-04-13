# chatbot-privacy-vault

> Starter project to demonstrate how a privacy vault fits into a chatbot

## Architecture

The app runs as **two processes**: a **Streamlit** front end ([`ui/streamlit_app.py`](ui/streamlit_app.py)) and a **Flask** API ([`backend/app.py`](backend/app.py)). The UI keeps the full conversation in `st.session_state["messages"]` and sends that list to the backend on every turn. The backend is **stateless**; it does not store chats server-side.

**Streaming path:** Flask calls OpenAI `chat.completions` with `stream=True`, forwards each text delta to the client as **NDJSON** (`application/x-ndjson`: one JSON object per line, e.g. `{"t":"..."}`, then `{"done":true}`). Streamlit reads the response with **httpx** in streaming mode, parses lines, and feeds token strings to **`st.write_stream`**. When the stream finishes, the assistant reply is appended to session state and the app rerenders so history shows a single assistant bubble per turn.

Errors **before** the response body streams use normal JSON (`{"error":"..."}`) and an HTTP error status. Errors **during** the stream can appear as an NDJSON line `{"error":"..."}`.

```mermaid
flowchart LR
  subgraph streamlit [Streamlit]
    SessionState[session_state messages]
    ChatInput[chat_input]
    WriteStream[write_stream]
  end
  subgraph flask [Flask]
    ChatPost["POST /chat"]
    NdjsonOut[NDJSON body]
  end
  subgraph openaiSvc [OpenAI]
    StreamChat[chat.completions stream]
  end
  ChatInput --> SessionState
  SessionState -->|POST JSON messages| ChatPost
  ChatPost --> StreamChat
  StreamChat -->|delta content| NdjsonOut
  NdjsonOut -->|httpx iter_lines| WriteStream
  WriteStream --> SessionState
```

**Endpoints:** `GET /health` returns `{"ok": true}`. `POST /chat` accepts `{"messages":[{"role":"user"|"assistant"|"system","content":"..."}]}`; the last message must be from the `user`.

**Configuration:** see [`.env.example`](.env.example) (`OPENAI_API_KEY`, optional `OPENAI_MODEL`, `CHAT_API_URL` for the Streamlit → Flask URL).

## Local env setup

1. Copy [`.env.example`](.env.example) to `.env` and set `OPENAI_API_KEY`.

2. Start the Flask API (terminal 1):
   `uv run flask --app backend.app run --host 127.0.0.1 --port 5000`

3. Start the Streamlit UI (terminal 2):
   `uv run streamlit run ui/streamlit_app.py`

4. Start the Privacy Vault (terminal 3):
   `uv run flask --app vault.app run --host 127.0.0.1 --port 5001`

   The vault exposes two endpoints:
   - `POST /anonymize` — replaces names, emails, and phone numbers with 8-char hex tokens
   - `POST /deanonymize` — looks up tokens and restores the original PII values

   By default the vault uses spaCy's English model (`en_core_web_sm`) for name detection. For Spanish or other languages, install the matching model and point to it via `SPACY_MODEL`:
   ```bash
   python -m spacy download es_core_news_sm
   SPACY_MODEL=es_core_news_sm uv run flask --app vault.app run --host 127.0.0.1 --port 5001
   ```