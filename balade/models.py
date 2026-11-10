"""Ollama models installed on this machine, for the model picker."""
from __future__ import annotations

import requests

# Embedding-only models cannot answer questions: hide them from the list.
_EMBEDDING_HINTS = ("embed", "bge", "nomic", "e5", "minilm")


def installed_chat_models(base_url: str) -> list[str]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=3)
        response.raise_for_status()
        names = [model["name"] for model in response.json().get("models", [])]
    except Exception:
        return []
    return sorted(name for name in names if not any(hint in name.lower() for hint in _EMBEDDING_HINTS))
