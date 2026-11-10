"""Conversation export as a Markdown file."""
from __future__ import annotations

from datetime import datetime

from balade.i18n import t
from balade.rag import Source


def conversation_markdown(messages: list[dict], lang: str, now: datetime | None = None) -> str:
    now = now or datetime.now()
    lines = [f"# Balade.AI · {now:%Y-%m-%d %H:%M}", ""]
    for message in messages:
        if message["role"] == "user":
            lines += [f"**{t('export.you', lang)}:** {message['content']}", ""]
            continue
        lines += [f"**Balade.AI:** {message['content']}", ""]
        sources = [Source(**s) for s in message.get("sources", [])]
        if sources:
            lines.append(f"*{t('export.sources', lang)}:*")
            lines += [f"- [{s.display_title(lang)} · {s.timestamp}]({s.url})" for s in sources]
            lines.append("")
    return "\n".join(lines).strip() + "\n"
