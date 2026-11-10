import sys
import types
from datetime import datetime

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from balade import models, voice
from balade.export import conversation_markdown
from balade.feedback import load_feedback, save_feedback
from balade.index import VideoInfo, add_video, chunk_count, indexed_videos, open_store, remove_video, video_timeline
from balade.rag import RagEngine
from balade.search import KeywordIndex, reciprocal_rank_fusion, tokenize
from balade.summary import build_transcript, link_timestamps, remove_overlap
from balade.transcripts import Segment


# -- hybrid search ----------------------------------------------------------

def test_tokenize_removes_accents_and_stopwords():
    assert tokenize("Qu'est-ce que l'Été à -273,15 °C ?") == ["ete", "273", "15"]


def test_reciprocal_rank_fusion_favours_documents_found_twice():
    a, b, c = (Document(id=i, page_content=i) for i in "abc")
    fused = reciprocal_rank_fusion([[a, b], [c, b]], k=3)
    assert fused[0].id == "b"
    assert {d.id for d in fused} == {"a", "b", "c"}


@pytest.fixture
def store(tmp_path):
    store = open_store(DeterministicFakeEmbedding(size=32), persist_dir=tmp_path, collection="features")
    zero = [Segment(f"phrase {i} sur le froid extrême", i * 10.0) for i in range(30)]
    zero[12] = Segment("le zéro absolu vaut moins 273,15 degrés Celsius", 120.0)
    add_video(store, VideoInfo("zeroabsolu1", "Le froid maximal", topic="physics"), zero, size=120, overlap=30)
    squid = [Segment(f"la pieuvre vampire phrase {i}", i * 5.0) for i in range(30)]
    add_video(store, VideoInfo("pieuvre0001", "Les pieuvres vampires", topic="marine_biology"), squid, size=120, overlap=30)
    return store


def test_keyword_search_finds_exact_numbers_and_respects_the_video_filter(store):
    index = KeywordIndex(store)
    hits = index.search("273,15", k=3)
    assert hits and "273,15" in hits[0].page_content
    limited = index.search("273,15", k=3, video_id="pieuvre0001")
    assert all(d.metadata["video_id"] == "pieuvre0001" for d in limited)


def test_keyword_index_follows_additions_and_removals(store):
    index = KeywordIndex(store)
    assert index.search("pieuvre", k=1)
    remove_video(store, "pieuvre0001")
    assert index.search("pieuvre", k=1) == []
    assert [v.video_id for v in indexed_videos(store)] == ["zeroabsolu1"]


def test_hybrid_retrieval_puts_the_exact_match_first_and_can_be_limited_to_one_video(store):
    engine = RagEngine(store, llm=None, k=4, fetch_k=20)
    docs = engine.retrieve("Combien vaut le zéro absolu, 273,15 ?")
    assert "273,15" in docs[0].page_content
    only_squid = engine.retrieve("zéro absolu 273,15", video_id="pieuvre0001")
    assert only_squid and all(d.metadata["video_id"] == "pieuvre0001" for d in only_squid)


def test_index_is_persisted_on_disk(store, tmp_path):
    count = chunk_count(store)
    reopened = open_store(DeterministicFakeEmbedding(size=32), persist_dir=tmp_path, collection="features")
    assert chunk_count(reopened) == count


def test_answer_limited_to_one_video_only_cites_that_video(store):
    llm = GenericFakeChatModel(messages=iter([AIMessage(content="La pieuvre vampire vit dans les abysses.")]))
    answer = RagEngine(store, llm, k=3, fetch_k=10).answer("Parle-moi du froid", [], "fr", video_id="pieuvre0001")
    "".join(answer)
    assert answer.sources and {s.video_id for s in answer.sources} == {"pieuvre0001"}


# -- summary ----------------------------------------------------------------

def test_remove_overlap():
    assert remove_overlap("a b c. Le froid extrême est partout.", "Le froid extrême est partout. Suite.") == "Suite."
    assert remove_overlap("rien à voir", "Texte nouveau sans recouvrement") == "Texte nouveau sans recouvrement"


def test_build_transcript_uses_times_and_fits_the_budget(store):
    timeline = video_timeline(store, "zeroabsolu1")
    assert timeline == sorted(timeline)
    full = build_transcript(timeline, budget_chars=100_000)
    assert full.startswith("[0:00] ")
    assert "273,15" in full
    short = build_transcript(timeline, budget_chars=400)
    assert len(short) < len(full)
    assert short.count("\n") == full.count("\n")      # every passage kept, each one shortened


def test_link_timestamps():
    text = link_timestamps("- [3:42] Le zéro absolu\n- [1:02:03] Fin", "abcdefghijk")
    assert "[▶ 3:42](https://www.youtube.com/watch?v=abcdefghijk&t=222s)" in text
    assert "&t=3723s" in text


# -- export, feedback, models, voice ----------------------------------------

def test_conversation_markdown():
    messages = [
        {"role": "user", "content": "C'est quoi le zéro absolu ?"},
        {"role": "assistant", "content": "0 K.", "sources": [
            {"video_id": "abcdefghijk", "title": "Le froid", "title_en": "Cold", "start": 222.0, "excerpt": "..."}]},
    ]
    md = conversation_markdown(messages, "fr", now=datetime(2026, 11, 10, 2, 0))
    assert md.startswith("# Balade.AI · 2026-11-10 02:00")
    assert "**Toi:** C'est quoi le zéro absolu ?" in md
    assert "- [Le froid · 3:42](https://www.youtube.com/watch?v=abcdefghijk&t=222s)" in md


def test_feedback_is_appended(tmp_path):
    path = tmp_path / "feedback.jsonl"
    save_feedback(path, "Q1", "A1", "up", [{"video_id": "abcdefghijk", "start": 12.5}], "llama3.1:8b")
    save_feedback(path, "Q2", "A2", "down", [], "llama3.1:8b")
    records = load_feedback(path)
    assert [r["rating"] for r in records] == ["up", "down"]
    assert records[0]["sources"] == ["abcdefghijk@12"]


def test_installed_chat_models_hides_embedding_models(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"models": [{"name": "llama3.1:8b"}, {"name": "bge-m3:latest"},
                               {"name": "nomic-embed-text:latest"}, {"name": "qwen2.5:14b"}]}

    monkeypatch.setattr(models.requests, "get", lambda *a, **k: Response())
    assert models.installed_chat_models("http://localhost:11434") == ["llama3.1:8b", "qwen2.5:14b"]

    def down(*a, **k):
        raise ConnectionError

    monkeypatch.setattr(models.requests, "get", down)
    assert models.installed_chat_models("http://localhost:11434") == []


def test_voice_transcriber_with_a_fake_whisper(monkeypatch):
    class FakeWhisperModel:
        def __init__(self, size, device, compute_type):
            self.args = (size, device, compute_type)

        def transcribe(self, audio, vad_filter, beam_size):
            assert audio.read() == b"RIFF-fake-wav"
            return iter([types.SimpleNamespace(text=" C'est quoi "), types.SimpleNamespace(text="le zéro absolu ? ")]), None

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisperModel))
    transcriber = voice.Transcriber("small", "cpu")
    assert transcriber.model.args == ("small", "cpu", "int8")
    assert transcriber.transcribe(b"RIFF-fake-wav") == "C'est quoi le zéro absolu ?"
