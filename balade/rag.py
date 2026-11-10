"""Answering a question: language, intent, retrieval (MMR), streamed answer and fallback to general knowledge."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterator

from langchain_core.documents import Document

from balade import config
from balade.i18n import t
from balade.transcripts import format_timestamp, video_url

# The model answers with this marker when the excerpts do not contain the answer.
NOT_FOUND = "NOT_IN_VIDEOS"

RAG_PROMPT = """You are Balade.AI, an assistant for the science videos of the French YouTube channel Balade Mentale.
Answer the question using ONLY the video excerpts below.
- If the excerpts do not contain the answer, reply with exactly {marker} and nothing else.
- Answer in {language}, even if the excerpts are in another language.
- Be clear and educational: 2 to 6 sentences, or a short list when it helps.
- Talk about what "the video" explains; never mention "excerpts" or "transcripts".
- Use the conversation so far to understand follow-up questions.

VIDEO EXCERPTS
{context}

CONVERSATION SO FAR
{history}

QUESTION: {question}"""

GENERAL_PROMPT = """You are Balade.AI, a friendly science assistant.
The indexed Balade Mentale videos do not cover this question: answer it from your general knowledge,
in {language}, clearly and briefly (2 to 6 sentences).

CONVERSATION SO FAR
{history}

QUESTION: {question}"""

LANGUAGE_NAMES = {"fr": "French", "en": "English"}


# -- language and intent ----------------------------------------------------

_WORD_RE = re.compile(r"[a-zàâäçéèêëîïôöùûüÿœ]+")
_FR_WORDS = {"le", "la", "les", "des", "du", "de", "est", "une", "un", "que", "qui", "quoi", "pourquoi",
             "comment", "quel", "quelle", "quels", "quelles", "et", "dans", "sur", "ce", "sont", "il",
             "elle", "ils", "elles", "tu", "vous", "je", "au", "aux", "pas", "ne", "qu", "explique",
             "combien", "où", "peux", "parle", "avec", "pour", "leur", "leurs", "ces", "cette"}
_EN_WORDS = {"the", "what", "why", "how", "is", "are", "does", "do", "which", "who", "where", "when",
             "can", "you", "of", "and", "tell", "explain", "about", "an", "it", "there", "many", "much",
             "they", "their", "this", "that", "with", "for", "me", "from", "these", "was", "were"}


def detect_language(text: str, default: str = "fr") -> str:
    """'fr' or 'en' from common words and accents; `default` (the interface language) when unsure."""
    words = _WORD_RE.findall(text.lower())
    french = sum(w in _FR_WORDS for w in words) + (2 if re.search(r"[àâçéèêëîïôùûœ]", text.lower()) else 0)
    english = sum(w in _EN_WORDS for w in words)
    if french == english:
        return default
    return "fr" if french > english else "en"


_GREETINGS = {"bonjour", "bonsoir", "salut", "coucou", "hello", "hi", "hey", "yo", "hola", "olá", "ola"}
_THANKS = {"merci", "thanks", "thank", "thx"}
_TOPIC_NOUNS = {"sujets", "thèmes", "themes", "vidéos", "videos", "topics", "subjects"}
_TOPIC_CUES = {"quels", "quelles", "quel", "quelle", "which", "what", "liste", "list", "disponibles",
               "available", "connais", "know", "indexées", "indexed", "as", "have", "parles", "cover"}


def classify(text: str) -> str:
    """'greeting', 'thanks', 'topics' or 'question'. Whole words only, and only for short messages:
    'What is this?' is a question, not a greeting, even though it contains 'hi'."""
    words = _WORD_RE.findall(text.lower())
    if not words:
        return "question"
    if len(words) <= 4 and words[0] in _GREETINGS:
        return "greeting"
    if len(words) <= 5 and words[0] in _THANKS:
        return "thanks"
    if len(words) <= 10 and _TOPIC_NOUNS & set(words) and _TOPIC_CUES & set(words):
        return "topics"
    return "question"


# -- answer -----------------------------------------------------------------

@dataclass
class Source:
    video_id: str
    title: str
    title_en: str
    start: float
    excerpt: str

    @property
    def url(self) -> str:
        return video_url(self.video_id, self.start)

    @property
    def timestamp(self) -> str:
        return format_timestamp(self.start)

    def display_title(self, lang: str) -> str:
        return self.title_en if lang == "en" and self.title_en else self.title

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_document(cls, doc: Document) -> "Source":
        meta = doc.metadata
        return cls(meta.get("video_id", ""), meta.get("title", ""), meta.get("title_en", ""),
                   float(meta.get("start", 0) or 0), doc.page_content)


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{doc.metadata.get('title', '')} — {format_timestamp(float(doc.metadata.get('start', 0) or 0))}]\n"
        f"{doc.page_content}"
        for doc in docs
    )


def fit_history(history: list[dict], budget_chars: int, max_messages: int = config.HISTORY_MESSAGES) -> str:
    """The most recent messages that fit in the space left in the context window.
    Older messages are dropped first, so a long conversation never overflows the model."""
    lines: list[str] = []
    used = 0
    for message in reversed(history[-max_messages:]):
        role = "User" if message["role"] == "user" else "Assistant"
        line = f"{role}: {message['content'][:2000]}"
        if used + len(line) > budget_chars:
            break
        lines.insert(0, line)
        used += len(line) + 1
    return "\n".join(lines) or "(none)"


def retrieval_query(question: str, history: list[dict]) -> str:
    """Short follow-ups ("and why?") are searched together with the previous question."""
    previous = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    if previous and len(question.split()) <= 6:
        return f"{previous}\n{question}"
    return question


_NOT_FOUND_PHRASES = ("pas dans la vidéo", "pas dans les vidéos", "pas abordé", "not in the video",
                      "not covered", "not mentioned in the video")


def looks_like_not_found(text: str) -> bool:
    """Small models sometimes write their own sentence instead of the marker."""
    lowered = text.lower()
    return len(lowered) < 250 and any(phrase in lowered for phrase in _NOT_FOUND_PHRASES)


class RagEngine:
    def __init__(self, store, llm, k: int = config.TOP_K, fetch_k: int = config.FETCH_K,
                 lambda_mult: float = config.MMR_LAMBDA, context_tokens: int = config.LLM_CONTEXT):
        self.store = store
        self.llm = llm
        self.k = k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        # ~3 characters per token is a safe estimate for French and English with Llama.
        self.budget_chars = (context_tokens - config.ANSWER_TOKENS) * 3

    def retrieve(self, query: str) -> list[Document]:
        """MMR search: relevant excerpts that do not repeat each other."""
        if self.store._collection.count() == 0:
            return []
        return self.store.max_marginal_relevance_search(
            query, k=self.k, fetch_k=self.fetch_k, lambda_mult=self.lambda_mult)

    def stream(self, prompt: str) -> Iterator[str]:
        for chunk in self.llm.stream(prompt):
            if isinstance(chunk.content, str) and chunk.content:
                yield chunk.content

    def answer(self, question: str, history: list[dict], lang: str) -> "AnswerStream":
        return AnswerStream(self, question, history, lang)


class AnswerStream:
    """Iterate over it to get the answer token by token (for st.write_stream).
    Once the iteration is over, `text`, `from_videos` and `sources` describe the answer."""

    def __init__(self, engine: RagEngine, question: str, history: list[dict], lang: str):
        self.engine = engine
        self.question = question
        self.history = history
        self.lang = lang
        self.text = ""
        self.from_videos = False
        self.sources: list[Source] = []

    def _emit(self, token: str) -> Iterator[str]:
        self.text += token
        yield token

    def __iter__(self) -> Iterator[str]:
        language = LANGUAGE_NAMES.get(self.lang, "French")
        docs = self.engine.retrieve(retrieval_query(self.question, self.history))

        if docs:
            context = format_context(docs)
            room = self.engine.budget_chars - len(RAG_PROMPT) - len(context) - len(self.question)
            prompt = RAG_PROMPT.format(marker=NOT_FOUND, language=language, context=context,
                                       history=fit_history(self.history, room), question=self.question)
            found = yield from self._stream_unless_not_found(self.engine.stream(prompt))
            if found and not looks_like_not_found(self.text):
                self.from_videos = True
                self.sources = [Source.from_document(doc) for doc in docs]
                return

        # Not covered by the videos: say so, then answer from general knowledge.
        # (If the model already wrote its own "not in the video" sentence, we just continue after it.)
        if not self.text:
            yield from self._emit(t("answer.not_in_videos", self.lang))
        yield from self._emit("\n\n")
        room = self.engine.budget_chars - len(GENERAL_PROMPT) - len(self.question)
        prompt = GENERAL_PROMPT.format(language=language, history=fit_history(self.history, room),
                                       question=self.question)
        for token in self.engine.stream(prompt):
            yield from self._emit(token)

    def _stream_unless_not_found(self, tokens: Iterator[str]):
        """Streams the answer, but holds back the first characters until we know they are not
        the NOT_IN_VIDEOS marker, so the user never sees it. Returns True if an answer was found."""
        buffer = ""
        for token in tokens:
            buffer += token
            probe = buffer.lstrip()
            if len(probe) < len(NOT_FOUND) and NOT_FOUND.startswith(probe):
                continue                      # still possibly the marker: keep waiting
            if probe.startswith(NOT_FOUND):
                return False
            yield from self._emit(probe)
            for rest in tokens:
                yield from self._emit(rest)
            return True
        return False                          # empty answer or only the marker
