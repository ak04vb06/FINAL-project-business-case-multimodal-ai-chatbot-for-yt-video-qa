"""YouTube side: video ids, transcripts with timestamps, video titles and the local transcript cache."""
from __future__ import annotations

import html
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests

VIDEO_ID_RE = re.compile(r"(?:[?&]v=|youtu\.be/|/shorts/|/embed/|/live/)([\w-]{11})(?![\w-])")
BARE_ID_RE = re.compile(r"^[\w-]{11}$")


@dataclass
class Segment:
    """One caption line of a video."""
    text: str
    start: float          # seconds from the start of the video


@dataclass
class Transcript:
    video_id: str
    language: str
    segments: list[Segment] = field(default_factory=list)


class TranscriptError(Exception):
    """Raised with a short reason the interface can translate: no_transcript, blocked, unavailable, error."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# -- ids, links, times ------------------------------------------------------

def parse_video_id(url_or_id: str) -> str | None:
    """Accepts youtu.be links, watch?v= links, shorts, embeds, or a bare 11-character id."""
    value = (url_or_id or "").strip()
    if BARE_ID_RE.match(value):
        return value
    match = VIDEO_ID_RE.search(value)
    return match.group(1) if match else None


def video_url(video_id: str, start: float = 0) -> str:
    """Link to the video, opening at `start` seconds."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    return f"{url}&t={int(start)}s" if start >= 1 else url


def thumbnail_url(video_id: str) -> str:
    return f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg"


def format_timestamp(seconds: float) -> str:
    """222 -> '3:42', 3723 -> '1:02:03'."""
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


# -- cleaning ---------------------------------------------------------------

def clean_text(text: str) -> str:
    """Removes HTML entities and tags, sound cues like [Musique], and extra whitespace."""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    return re.sub(r"\s+", " ", text).strip()


# -- fetching ---------------------------------------------------------------

# Exception names of youtube-transcript-api, grouped into the reasons shown to the user.
_REASONS = {
    "no_transcript": {"TranscriptsDisabled", "NoTranscriptFound"},
    "blocked": {"RequestBlocked", "IpBlocked", "PoTokenRequired", "YouTubeRequestFailed"},
    "unavailable": {"VideoUnavailable", "InvalidVideoId", "VideoUnplayable", "AgeRestricted"},
}


def fetch_transcript(video_id: str, languages=("fr", "en")) -> Transcript:
    """Downloads the captions of a video (manual or auto-generated), keeping each line's start time."""
    from youtube_transcript_api import YouTubeTranscriptApi

    try:
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=list(languages))
    except Exception as error:
        name = type(error).__name__
        reason = next((r for r, names in _REASONS.items() if name in names), "error")
        raise TranscriptError(reason) from error

    segments = [Segment(clean_text(s.text), float(s.start)) for s in fetched]
    segments = [s for s in segments if s.text]
    if not segments:
        raise TranscriptError("no_transcript")
    return Transcript(video_id, getattr(fetched, "language_code", "") or languages[0], segments)


def fetch_video_details(video_id: str) -> dict:
    """Title and channel from YouTube's public oEmbed endpoint (no API key). Empty dict if it fails."""
    try:
        response = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": video_url(video_id), "format": "json"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        return {"title": data.get("title", ""), "channel": data.get("author_name", "")}
    except Exception:
        return {}


# -- cache ------------------------------------------------------------------

def load_cached(video_id: str, cache_dir: Path) -> Transcript | None:
    path = Path(cache_dir) / f"{video_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return Transcript(data["video_id"], data.get("language", ""),
                      [Segment(s["text"], float(s["start"])) for s in data["segments"]])


def save_cache(transcript: Transcript, cache_dir: Path) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{transcript.video_id}.json").write_text(
        json.dumps(asdict(transcript), ensure_ascii=False), encoding="utf-8")


def get_transcript(video_id: str, cache_dir: Path, languages=("fr", "en")) -> Transcript:
    """Cached transcript if we already have it, otherwise download it and cache it.
    YouTube blocks repeated requests, so each video is only fetched once."""
    cached = load_cached(video_id, cache_dir)
    if cached:
        return cached
    transcript = fetch_transcript(video_id, languages)
    save_cache(transcript, cache_dir)
    return transcript
