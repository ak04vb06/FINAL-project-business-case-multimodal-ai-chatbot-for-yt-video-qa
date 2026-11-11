"""Balade.AI: Streamlit interface of the YouTube video Q&A chatbot.

Run with:  streamlit run app.py
The index must exist first (python build_index.py), or videos can be added from the sidebar.
"""
import logging
import time
import uuid
from datetime import datetime

import streamlit as st
from langchain_ollama import ChatOllama

from balade import config, voice
from balade.export import conversation_markdown
from balade.feedback import save_feedback
from balade.i18n import LANGUAGES, t
from balade.index import (TOPICS, VideoInfo, add_video, chunk_count, indexed_videos, is_indexed, make_embeddings,
                          open_store, remove_video, video_timeline)
from balade.models import installed_chat_models
from balade.rag import LANGUAGE_NAMES, RagEngine, Source, classify, detect_language
from balade.summary import SUMMARY_PROMPT, build_transcript, link_timestamps
from balade.transcripts import TranscriptError, fetch_video_details, get_transcript, parse_video_id
from balade.ui import (apply_style, brand_html, hero_html, label_html, meta_html, sources_html,
                       stats_html, videos_html)

log = logging.getLogger("balade")

st.set_page_config(page_title="Balade.AI", page_icon=":material/auto_awesome:", layout="centered", initial_sidebar_state="auto")
apply_style()


# -- resources (loaded once per server, shared by every visitor) ------------

@st.cache_resource(show_spinner="Loading the embedding model…")
def load_engine() -> RagEngine:
    config.enable_tracing()
    return RagEngine(open_store(make_embeddings()), get_llm(config.OLLAMA_MODEL))


@st.cache_resource
def get_llm(model: str) -> ChatOllama:
    return ChatOllama(model=model, base_url=config.OLLAMA_BASE_URL,
                      temperature=config.LLM_TEMPERATURE, num_ctx=config.LLM_CONTEXT)


@st.cache_data(ttl=60, show_spinner=False)
def chat_models() -> list[str]:
    models = installed_chat_models(config.OLLAMA_BASE_URL)
    return models if config.OLLAMA_MODEL in models else [config.OLLAMA_MODEL, *models]


@st.cache_resource(show_spinner="Loading the speech recognition model…")
def get_transcriber() -> voice.Transcriber:
    return voice.Transcriber()


# -- session ----------------------------------------------------------------

state = st.session_state
state.setdefault("messages", [])   # see new_message() for the fields
state.setdefault("lang", "fr")
state.setdefault("scope", "all")   # "all" or the video_id the search is limited to
state.setdefault("model", config.OLLAMA_MODEL)


def tr(key: str, **values) -> str:
    """Interface text in the language picked in the sidebar."""
    return t(key, state.lang, **values)


def new_message(role: str, content: str, **fields) -> dict:
    """role/content plus, for questions: kind ("question" or "summary") and video_id;
    for answers: kind, from_videos, sources, seconds, video_id."""
    return {"id": uuid.uuid4().hex[:10], "role": role, "content": content, **fields}


try:
    engine = load_engine()
except Exception as error:
    log.exception("Start-up failed")
    st.error(f"**{tr('setup.title')}**\n\n`{error}`")
    st.info(tr("setup.help", model=config.OLLAMA_MODEL))
    st.stop()

store = engine.store
videos = indexed_videos(store)
video_by_id = {video.video_id: video for video in videos}

# A video added or removed in the previous run changes the scope before the widget is drawn.
if "next_scope" in state:
    state.scope = state.pop("next_scope")
if state.scope != "all" and state.scope not in video_by_id:
    state.scope = "all"


# -- sidebar actions --------------------------------------------------------

def keep_one_language() -> None:
    """Clicking the selected language again would leave none selected: keep the current one."""
    if state.lang_choice:
        state.lang = state.lang_choice
    else:
        state.lang_choice = state.lang


def ask_summary(video_id: str) -> None:
    state.pending = {"type": "summary", "video_id": video_id}


def remove_scoped_video(video_id: str) -> None:
    title = video_by_id[video_id].display_title(state.lang)
    remove_video(store, video_id)
    state.next_scope = "all"
    state.flash = ("success", tr("remove.done", title=title))


def add_video_from_url(url: str, topic: str) -> None:
    """Fetches the captions, chunks and indexes them. The result is kept in state.flash so it is
    still shown after the page reloads, and the new video becomes the search scope."""
    video_id = parse_video_id(url)
    if not video_id:
        state.flash = ("error", tr("add.invalid"))
        return
    if is_indexed(store, video_id):
        state.flash = ("info", tr("add.exists"))
        return

    with st.status(tr("add.running"), expanded=True) as status:
        try:
            st.write(tr("add.step_transcript"))
            transcript = get_transcript(video_id, config.TRANSCRIPTS_DIR, config.TRANSCRIPT_LANGUAGES)
            details = fetch_video_details(video_id)
            video = VideoInfo(video_id=video_id, title=details.get("title") or tr("add.untitled", id=video_id),
                              topic=topic, channel=details.get("channel", ""), language=transcript.language)
            st.write(tr("add.step_index"))
            count = add_video(store, video, transcript.segments)
        except TranscriptError as error:
            status.update(state="error")
            state.flash = ("error", tr(f"add.{error.reason}"))
            return
        except Exception as error:
            log.exception("Adding a video failed")
            status.update(state="error")
            state.flash = ("error", tr("add.error", detail=error))
            return
    state.flash = ("success", tr("add.done", title=video.title, count=count))
    state.next_scope = video_id


def clear_conversation() -> None:
    state.messages = []


# -- sidebar ----------------------------------------------------------------

lang = state.lang
with st.sidebar:
    st.html(brand_html(lang))
    st.segmented_control("Language", options=list(LANGUAGES), format_func=str.upper, key="lang_choice",
                         default=lang, on_change=keep_one_language, label_visibility="collapsed")
    lang = state.lang

    st.html(stats_html(len(videos), chunk_count(store), config.LLM_CONTEXT, lang))

    # Search scope: all videos, or a single one (which can then be summarised or removed).
    st.selectbox(tr("scope.label"), ["all", *video_by_id], key="scope",
                 format_func=lambda key: tr("scope.all") if key == "all" else video_by_id[key].display_title(lang))
    if state.scope != "all":
        scoped = video_by_id[state.scope]
        st.button(tr("summary.button"), icon=":material/summarize:", width="stretch",
                  on_click=ask_summary, args=(scoped.video_id,), type="primary")
        with st.popover(tr("remove.button"), icon=":material/delete:", type="tertiary"):
            st.write(tr("remove.confirm", title=scoped.display_title(lang)))
            st.button(tr("remove.yes"), on_click=remove_scoped_video, args=(scoped.video_id,), type="primary")

    if "flash" in state:
        kind, text = state.pop("flash")
        getattr(st, kind)(text)

    with st.expander(tr("add.title"), icon=":material/add:"):
        with st.form("add_video", clear_on_submit=True, border=False):
            url = st.text_input(tr("add.url"), placeholder="https://youtu.be/…", help=tr("add.url_help"))
            topic = st.selectbox(tr("add.topic"), TOPICS, format_func=lambda key: t(f"topic.{key}", lang))
            if st.form_submit_button(tr("add.submit"), type="primary", width="stretch") and url:
                add_video_from_url(url, topic)
                st.rerun()

    st.html(label_html(f"{tr('sidebar.videos')} · {len(videos)}"))
    st.html(videos_html(videos, lang) if videos else f"<p class='ba-sub'>{tr('sidebar.empty')}</p>")

    with st.expander(tr("options.title"), icon=":material/settings:"):
        models = chat_models()
        if state.model not in models:
            state.model = models[0]
        st.selectbox(tr("options.model"), models, key="model", help=tr("options.model_help"))
        st.download_button(
            tr("options.export"), data=conversation_markdown(state.messages, lang),
            file_name=f"balade-ai-{datetime.now():%Y%m%d-%H%M}.md", mime="text/markdown",
            icon=":material/download:", width="stretch", on_click="ignore", disabled=not state.messages)
        st.button(tr("sidebar.clear"), icon=":material/delete_sweep:", width="stretch",
                  on_click=clear_conversation, disabled=not state.messages)


# -- conversation: display --------------------------------------------------

ASSISTANT_AVATAR = ":material/auto_awesome:"


def show_message(index: int, message: dict, is_last: bool) -> None:
    role = message["role"]
    with st.container(key=f"msg-{role}-{index}"):
        with st.chat_message(role, avatar=ASSISTANT_AVATAR if role == "assistant" else None):
            st.markdown(message["content"])
            if role == "assistant" and "from_videos" in message:
                show_details(message)
                show_actions(index, message, is_last)


def show_details(message: dict) -> None:
    """Badge, "watch at 3:42" link and answer time, then the excerpts used."""
    sources = [Source(**s) for s in message.get("sources", [])]
    video = video_by_id.get(message.get("video_id") or "")
    st.html(meta_html(message.get("kind", "answer"), message["from_videos"], sources, lang,
                      seconds=message.get("seconds"),
                      video=(video.video_id, video.display_title(lang)) if video else None))
    if sources:
        with st.expander(tr("chat.sources", count=len(sources))):
            st.html(sources_html(sources, lang))


def show_actions(index: int, message: dict, is_last: bool) -> None:
    """Thumbs up / down on every answer, and "regenerate" under the last one."""
    with st.container(horizontal=True, vertical_alignment="center", gap="medium"):
        st.feedback("thumbs", key=f"fb-{message['id']}", on_change=record_feedback, args=(index,))
        if is_last:
            st.button(tr("chat.regenerate"), icon=":material/refresh:", type="tertiary",
                      key=f"regen-{message['id']}", on_click=regenerate)


def record_feedback(index: int) -> None:
    answer = state.messages[index]
    rating = state.get(f"fb-{answer['id']}")
    if rating is None:
        return
    question = state.messages[index - 1]["content"] if index else ""
    try:
        save_feedback(config.FEEDBACK_FILE, question, answer["content"], "up" if rating == 1 else "down",
                      answer.get("sources", []), state.model)
        st.toast(tr("feedback.thanks"), icon=":material/favorite:")
    except Exception:
        log.exception("Saving feedback failed")


def regenerate() -> None:
    """Removes the last answer; the run that follows answers the same question again."""
    if state.messages and state.messages[-1]["role"] == "assistant":
        state.messages.pop()
        state.pending = {"type": "replay"}


# -- conversation: answering ------------------------------------------------

def topics_answer() -> str:
    lines = [f"- **{v.display_title(lang)}** · {t('topic.' + v.topic, lang) if v.topic in TOPICS else ''}"
             for v in videos]
    return tr("answer.topics", count=len(videos)) + "\n\n" + "\n".join(lines)


def answer_question(question: dict) -> dict | None:
    """Streams the answer to a question and returns the message to store (None if it failed)."""
    history = state.messages[:-1]
    intent = classify(question["content"])
    if intent != "question":
        text = {"greeting": tr("answer.greeting"), "thanks": tr("answer.thanks"), "topics": topics_answer()}[intent]
        st.markdown(text)
        return new_message("assistant", text, kind="chat")

    # The answer follows the language of the question; the interface language when unsure.
    answer = engine.answer(question["content"], history, detect_language(question["content"], default=lang),
                           video_id=question.get("video_id"), llm=get_llm(state.model))
    started = time.perf_counter()
    st.write_stream(answer)
    return new_message("assistant", answer.text, kind="answer", from_videos=answer.from_videos,
                       sources=[s.to_dict() for s in answer.sources], video_id=question.get("video_id"),
                       seconds=round(time.perf_counter() - started, 1))


def summarise(question: dict) -> dict:
    """Streams a summary of a whole video with key moments, then turns the [3:42] times into links."""
    video_id = question["video_id"]
    video = video_by_id[video_id]
    room = engine.budget_chars - len(SUMMARY_PROMPT) - 500
    prompt = SUMMARY_PROMPT.format(title=video.title, language=LANGUAGE_NAMES[lang],
                                   moments_title=tr("summary.moments"),
                                   transcript=build_transcript(video_timeline(store, video_id), room))
    started = time.perf_counter()
    placeholder = st.empty()
    text = placeholder.write_stream(engine.stream(prompt, get_llm(state.model)))
    text = link_timestamps(text if isinstance(text, str) else "".join(text), video_id)
    placeholder.markdown(text)
    return new_message("assistant", text, kind="summary", from_videos=True, sources=[], video_id=video_id,
                       seconds=round(time.perf_counter() - started, 1))


def reply(question: dict) -> None:
    index = len(state.messages)
    with st.container(key=f"msg-assistant-{index}"):
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            try:
                message = summarise(question) if question.get("kind") == "summary" else answer_question(question)
            except Exception:
                log.exception("LLM call failed")
                st.error(tr("chat.error", model=state.model))
                return
    state.messages.append(message)
    st.rerun()   # redraw the conversation with the thumbs and "regenerate" buttons


# -- page -------------------------------------------------------------------

scoped_title = video_by_id[state.scope].display_title(lang) if state.scope != "all" else None
placeholder = tr("scope.placeholder", title=scoped_title) if scoped_title else tr("chat.placeholder")
voice_on = voice.is_available()
submitted = st.chat_input(placeholder, accept_audio=voice_on, audio_sample_rate=16000)

# What to answer in this run: a typed or spoken question, a suggestion, a summary or a replay.
text = submitted if isinstance(submitted, str) else (submitted.text if submitted else "")
audio = getattr(submitted, "audio", None) if submitted and not isinstance(submitted, str) else None
if audio is not None and not (text or "").strip():
    try:
        with st.spinner(tr("voice.loading")):
            text = get_transcriber().transcribe(audio.getvalue())
        if not text:
            st.toast(tr("voice.empty"))
    except Exception as error:
        log.exception("Speech recognition failed")
        st.toast(tr("voice.error", detail=error))

pending = state.pop("pending", None)
question = None
if text and text.strip():
    if len(text) > config.MAX_QUESTION_CHARS:
        st.toast(tr("chat.too_long", max=config.MAX_QUESTION_CHARS))
    else:
        question = new_message("user", text.strip(), kind="question",
                               video_id=None if state.scope == "all" else state.scope)
elif pending and pending["type"] == "question":
    question = new_message("user", pending["text"], kind="question",
                           video_id=None if state.scope == "all" else state.scope)
elif pending and pending["type"] == "summary" and pending["video_id"] in video_by_id:
    title = video_by_id[pending["video_id"]].display_title(lang)
    question = new_message("user", tr("summary.request", title=title), kind="summary",
                           video_id=pending["video_id"])
elif pending and pending["type"] == "replay" and state.messages and state.messages[-1]["role"] == "user":
    question = state.messages[-1]                 # already in the conversation
if question and question is not (state.messages[-1] if state.messages else None):
    state.messages.append(question)

main = st.container()
if not state.messages:
    main.html(hero_html(lang))
    if not videos:
        main.info(tr("setup.empty"))
    with main.container(key="suggestions"):
        suggestions = [tr(f"welcome.s{i}") for i in range(1, 5)]
        for row in (suggestions[:2], suggestions[2:]):
            for column, suggestion in zip(st.columns(2), row):
                if column.button(suggestion, key=f"suggest-{suggestion}", width="stretch"):
                    state.pending = {"type": "question", "text": suggestion}
                    st.rerun()

with main:
    last = len(state.messages) - 1
    for i, message in enumerate(state.messages):
        show_message(i, message, is_last=(i == last and not question))
    if question:
        reply(question)
