"""Thumbs up / down on answers, saved locally (JSON Lines) to find the questions the bot answers badly."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def save_feedback(path: Path, question: str, answer: str, rating: str, sources: list[dict],
                  model: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rating": rating,                        # "up" or "down"
        "model": model,
        "question": question,
        "answer": answer,
        "sources": [f"{s['video_id']}@{int(s['start'])}" for s in sources],
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_feedback(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
