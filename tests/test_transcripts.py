import pytest

from balade import transcripts
from balade.transcripts import (Segment, Transcript, TranscriptError, clean_text, format_timestamp,
                                get_transcript, parse_video_id, video_url)


@pytest.mark.parametrize("value, expected", [
    ("https://youtu.be/300eyzsqang?si=YMrPUffy3kqjz6bK", "300eyzsqang"),
    ("https://www.youtube.com/watch?v=BXebijm_Mao&t=42s", "BXebijm_Mao"),
    ("https://www.youtube.com/watch?feature=share&v=2--n3nStqTk", "2--n3nStqTk"),
    ("https://youtube.com/shorts/NDeyeTQ-ZAM", "NDeyeTQ-ZAM"),
    ("https://www.youtube.com/embed/86-T834tOU8", "86-T834tOU8"),
    ("NFJk2fmIidU", "NFJk2fmIidU"),
    ("https://example.com/video", None),
    ("", None),
])
def test_parse_video_id(value, expected):
    assert parse_video_id(value) == expected


def test_timestamps_and_links():
    assert format_timestamp(0) == "0:00"
    assert format_timestamp(222.7) == "3:42"
    assert format_timestamp(3723) == "1:02:03"
    assert video_url("abcdefghijk", 222.7) == "https://www.youtube.com/watch?v=abcdefghijk&t=222s"
    assert video_url("abcdefghijk") == "https://www.youtube.com/watch?v=abcdefghijk"


def test_clean_text():
    assert clean_text("[Musique] les &amp; <i>pieuvres</i>\n  vampires ") == "les & pieuvres vampires"


def test_cache_is_used_before_youtube(tmp_path, monkeypatch):
    calls = []

    def fake_fetch(video_id, languages):
        calls.append(video_id)
        return Transcript(video_id, "fr", [Segment("bonjour", 1.5)])

    monkeypatch.setattr(transcripts, "fetch_transcript", fake_fetch)
    first = get_transcript("abcdefghijk", tmp_path)
    second = get_transcript("abcdefghijk", tmp_path)
    assert calls == ["abcdefghijk"]                     # downloaded once, then read from the cache
    assert second.segments == first.segments == [Segment("bonjour", 1.5)]


def test_fetch_errors_are_translated(monkeypatch):
    import youtube_transcript_api

    class TranscriptsDisabled(Exception):
        pass

    class FakeApi:
        def fetch(self, video_id, languages):
            raise TranscriptsDisabled(video_id)

    monkeypatch.setattr(youtube_transcript_api, "YouTubeTranscriptApi", FakeApi)
    with pytest.raises(TranscriptError) as error:
        transcripts.fetch_transcript("abcdefghijk")
    assert error.value.reason == "no_transcript"
