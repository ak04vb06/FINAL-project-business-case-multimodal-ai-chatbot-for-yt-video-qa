"""Splits a transcript into overlapping chunks that remember where they start in the video."""
from __future__ import annotations

from dataclasses import dataclass

from balade.transcripts import Segment


@dataclass
class Chunk:
    text: str
    start: float          # start time (seconds) of the first caption line in the chunk


def chunk_segments(segments: list[Segment], size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Groups caption lines into chunks of about `size` characters.

    Lines are never cut in half, so every chunk starts on a real caption line and we know
    its exact time in the video. Two consecutive chunks share the last lines of the first one
    (about `overlap` characters), so an idea split across two chunks is not lost.
    """
    chunks: list[Chunk] = []
    current: list[Segment] = []
    length = 0

    for segment in segments:
        if current and length + len(segment.text) + 1 > size:
            chunks.append(_join(current))
            # Start the next chunk with the tail of this one (the overlap).
            kept: list[Segment] = []
            kept_length = 0
            for previous in reversed(current):
                if kept_length + len(previous.text) + 1 > overlap:
                    break
                kept.insert(0, previous)
                kept_length += len(previous.text) + 1
            current, length = kept, kept_length
        current.append(segment)
        length += len(segment.text) + 1

    if current:
        chunks.append(_join(current))
    return chunks


def _join(segments: list[Segment]) -> Chunk:
    return Chunk(" ".join(s.text for s in segments), segments[0].start)
