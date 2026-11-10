"""Video summary with key moments: the whole transcript fits in the 32k context window."""
from __future__ import annotations

import re

from balade.transcripts import format_timestamp, video_url

SUMMARY_PROMPT = """You are Balade.AI. Below is the transcript of the video "{title}" from the YouTube channel
Balade Mentale. Each passage starts with the time at which it is said in the video.

Write in {language}, using only what the transcript says:
1. A summary of the video in 3 to 5 sentences.
2. A line "**{moments_title}**", then 5 to 8 key moments as a list, one per line. Each line starts
   with the time of that moment in square brackets, copied from the transcript (for example [3:42]),
   followed by a short description.

TRANSCRIPT
{transcript}"""

_TIME_RE = re.compile(r"\[(\d{1,2}(?::\d{2}){1,2})\]")


def remove_overlap(previous: str, current: str, max_overlap: int = 400) -> str:
    """Consecutive chunks share a few sentences: drop the part of `current` already in `previous`."""
    for size in range(min(len(previous), len(current), max_overlap), 19, -1):
        if previous.endswith(current[:size]):
            return current[size:].lstrip()
    return current


def build_transcript(timeline: list[tuple[float, str]], budget_chars: int) -> str:
    """'[m:ss] text' lines from the chunks of a video, without the overlaps. If the video is too
    long for the context window, every passage is shortened by the same ratio."""
    passages: list[tuple[float, str]] = []
    previous = ""
    for start, text in timeline:
        passages.append((start, remove_overlap(previous, text)))
        previous = text
    total = sum(len(text) for _, text in passages)
    if total > budget_chars and passages:
        keep = max(80, budget_chars // len(passages))
        passages = [(start, text[:keep]) for start, text in passages]
    return "\n".join(f"[{format_timestamp(start)}] {text}" for start, text in passages if text)


def parse_time(value: str) -> int:
    seconds = 0
    for part in value.split(":"):
        seconds = seconds * 60 + int(part)
    return seconds


def link_timestamps(text: str, video_id: str) -> str:
    """Turns every [3:42] of the summary into a link that opens the video at that moment."""
    return _TIME_RE.sub(lambda m: f"[▶ {m.group(1)}]({video_url(video_id, parse_time(m.group(1)))})", text)
