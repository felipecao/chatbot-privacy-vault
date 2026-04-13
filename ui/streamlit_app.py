import json
import os
from collections.abc import Iterator

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

CHAT_URL = os.environ.get("CHAT_API_URL", "http://127.0.0.1:5000/chat")


def _assistant_delta_stream(messages: list[dict[str, str]]):
    with httpx.Client(timeout=120.0) as client:
        with client.stream("POST", CHAT_URL, json={"messages": messages}) as response:
            if response.status_code != 200:
                body = response.read().decode()
                try:
                    msg = json.loads(body).get("error", body)
                except json.JSONDecodeError:
                    msg = body
                raise RuntimeError(msg)
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("done"):
                    break
                if obj.get("error"):
                    raise RuntimeError(obj["error"])
                if "t" in obj:
                    yield obj["t"]


def main() -> None:
    st.set_page_config(page_title="Chat", page_icon="")
    st.title("Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    prompt = st.chat_input("Message")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt:
        try:

            def deltas() -> Iterator[str]:
                yield from _assistant_delta_stream(list(st.session_state.messages))

            with st.chat_message("assistant"):
                streamed = st.write_stream(deltas)

            assistant_text = (
                streamed if isinstance(streamed, str) else "".join(str(x) for x in streamed)
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_text}
            )
            st.rerun()
        except (RuntimeError, httpx.HTTPError, json.JSONDecodeError) as e:
            st.error(str(e))


if __name__ == "__main__":
    main()
