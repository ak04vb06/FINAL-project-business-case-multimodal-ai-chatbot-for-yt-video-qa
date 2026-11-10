import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from balade.index import VideoInfo, add_video, chunk_count, indexed_videos, is_indexed, open_store
from balade.rag import NOT_FOUND, RagEngine, classify, detect_language, fit_history, retrieval_query
from balade.transcripts import Segment


@pytest.mark.parametrize("text, expected", [
    ("Qu'est-ce que la bioluminescence ?", "fr"),
    ("C'est quoi le zéro absolu ?", "fr"),
    ("What is absolute zero?", "en"),
    ("How do these creatures defend themselves?", "en"),
    ("ok", "fr"),                       # unsure -> interface language
])
def test_detect_language(text, expected):
    assert detect_language(text, default="fr") == expected


@pytest.mark.parametrize("text, expected", [
    ("Salut !", "greeting"),
    ("hello", "greeting"),
    ("Merci beaucoup", "thanks"),
    ("Quels sujets sont disponibles ?", "topics"),
    ("What videos do you have?", "topics"),
    # These contain "hi" / "vidéo" but are real questions (the old substring matching got them wrong):
    ("What is this creature?", "question"),
    ("Which planets are habitable?", "question"),
    ("Dans la vidéo, pourquoi la pieuvre vampire vit-elle si profond ?", "question"),
    ("Salut, comment se forment les exoplanètes autour des étoiles ?", "question"),
])
def test_classify(text, expected):
    assert classify(text) == expected


def test_follow_up_questions_are_searched_with_the_previous_one():
    history = [{"role": "user", "content": "Qu'est-ce que la bioluminescence ?"},
               {"role": "assistant", "content": "..."}]
    assert retrieval_query("Et pourquoi ?", history) == "Qu'est-ce que la bioluminescence ?\nEt pourquoi ?"
    long_question = "Comment les planètes du système solaire se sont-elles formées ?"
    assert retrieval_query(long_question, history) == long_question


def test_history_keeps_the_most_recent_messages_that_fit():
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i} " + "x" * 90}
               for i in range(30)]
    text = fit_history(history, budget_chars=450, max_messages=20)
    assert "message 29" in text and "message 26" in text
    assert "message 25" not in text                   # older messages dropped first
    assert fit_history([], 1000) == "(none)"


# -- end to end with a fake model and fake embeddings -----------------------

@pytest.fixture
def store(tmp_path):
    store = open_store(DeterministicFakeEmbedding(size=32), persist_dir=tmp_path, collection="test")
    segs = [Segment(f"phrase numéro {i} sur la bioluminescence des abysses", float(i * 10)) for i in range(40)]
    add_video(store, VideoInfo("abcdefghijk", "Les pieuvres vampires", "Vampire squid", "marine_biology"), segs)
    return store


def engine_with(store, *answers):
    llm = GenericFakeChatModel(messages=iter(AIMessage(content=a) for a in answers))
    return RagEngine(store, llm, k=3, fetch_k=10)


def test_add_video_is_idempotent_and_listed(store):
    count = chunk_count(store)
    assert count > 1
    assert is_indexed(store, "abcdefghijk")
    assert add_video(store, VideoInfo("abcdefghijk", "x"), [Segment("y", 0)]) == 0
    assert chunk_count(store) == count
    videos = indexed_videos(store)
    assert [v.video_id for v in videos] == ["abcdefghijk"]
    assert videos[0].display_title("en") == "Vampire squid"


def test_answer_from_the_videos_is_streamed_with_sources(store):
    answer = engine_with(store, "La pieuvre vampire produit de la lumière.").answer(
        "Qu'est-ce que la bioluminescence ?", [], "fr")
    tokens = list(answer)
    assert len(tokens) > 3                               # streamed in several pieces
    assert answer.text == "La pieuvre vampire produit de la lumière."
    assert answer.from_videos is True
    assert len(answer.sources) == 3
    assert all(s.video_id == "abcdefghijk" for s in answer.sources)
    assert "&t=" in answer.sources[0].url or answer.sources[0].start == 0


def test_marker_switches_to_general_knowledge_and_is_never_shown(store):
    answer = engine_with(store, NOT_FOUND, "Le football se joue à onze.").answer(
        "Combien de joueurs dans une équipe de football ?", [], "fr")
    text = "".join(answer)
    assert NOT_FOUND not in text
    assert text.startswith("Les vidéos indexées n'abordent pas ce point.")
    assert text.endswith("Le football se joue à onze.")
    assert answer.from_videos is False and answer.sources == []


def test_model_own_not_found_sentence_also_falls_back(store):
    answer = engine_with(store, "This information is not in the video.", "Paris is in France.").answer(
        "Where is Paris?", [], "en")
    text = "".join(answer)
    assert text.startswith("This information is not in the video.")
    assert "Paris is in France." in text
    assert answer.from_videos is False


def test_empty_index_goes_straight_to_general_knowledge(tmp_path):
    empty = open_store(DeterministicFakeEmbedding(size=32), persist_dir=tmp_path, collection="empty")
    answer = engine_with(empty, "Une réponse générale.").answer("Question ?", [], "fr")
    assert "".join(answer).endswith("Une réponse générale.")
    assert answer.from_videos is False
