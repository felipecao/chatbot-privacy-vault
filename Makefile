.DEFAULT_GOAL := help

.PHONY: help backend frontend

help:
	@echo "Targets:"
	@echo "  make backend   - Flask chat API (http://127.0.0.1:5000)"
	@echo "  make frontend  - Streamlit UI"

backend:
	uv run flask --app backend.app run --host 127.0.0.1 --port 5000

frontend:
	uv run streamlit run ui/streamlit_app.py
