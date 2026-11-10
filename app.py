"""Balade.AI: Streamlit interface of the YouTube video Q&A chatbot.

Run with:  streamlit run app.py
The index must exist first (python build_index.py), or videos can be added from the sidebar.
"""
import logging

import streamlit as st
from langchain_ollama import ChatOllama

from balade import config
from balade.i18n import LANGUAGES, t
from balade.index import TOPICS, VideoInfo, add_video, chunk_count, indexed_videos, is_indexed, make_embeddings, open_store
from balade.rag import RagEngine, Source, classify, detect_language
from balade.transcripts import TranscriptError, fetch_video_details, get_transcript, parse_video_id
from balade.ui import (apply_style, brand_html, hero_html, label_html, meta_html, sources_html,
                       stats_html, videos_html)

log = logging.getLogger("balade")

st.set_page_config(page_title="Balade.AI", page_icon="🌌", layout="centered", initial_sidebar_state="auto")
apply_style()


# -- pipeline (loaded once per server, shared by every visitor) -------------

@st.cache_resource(show_spinner="Loading the embedding model…")
def load_engine() -> RagEngine:
    config.enable_tracing()
    store = open_store(make_embeddings())
    llm = ChatOllama(model=config.OLLAMA_MODEL, base_url=config.OLLAMA_BASE_URL,
                     temperature=config.LLM_TEMPERATURE, num_ctx=config.LLM_CONTEXT)
    return RagEngine(store, llm)


# -- session ----------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []     # {"role", "content", "from_videos", "sources"}
if "lang" not in st.session_state:
    st.session_state.lang = "fr"


def tr(key: str, **values) -> str:
    """Interface text in the language picked in the sidebar."""
    return t(key, st.session_state.lang, **values)


try:
    engine = load_engine()
except Exception as error:
    log.exception("Start-up failed")
    st.error(f"**{tr('setup.title')}**\n\n`{error}`")
    st.info(tr("setup.help", model=config.OLLAMA_MODEL))
    st.stop()

store = engine.store


# -- add a video ------------------------------------------------------------

def add_video_from_url(url: str, topic: str) -> None:
    """Fetches the captions, chunks and indexes them. The result is kept in session_state.flash
    so it is still shown after the page reloads."""
    video_id = parse_video_id(url)
    if not video_id:
        st.session_state.flash = ("error", tr("add.invalid"))
        return
    if is_indexed(store, video_id):
        st.session_state.flash = ("info", tr("add.exists"))
        return

    with st.status(tr("add.running"), expanded=True) as status:
        try:
            st.write(tr("add.step_transcript"))
            transcript = get_transcript(video_id, config.TRANSCRIPTS_DIR, config.TRANSCRIPT_LANGUAGES)
            details = fetch_video_details(video_id)
            video = VideoInfo(
                video_id=video_id,
                title=details.get("title") or tr("add.untitled", id=video_id),
                topic=topic,
                channel=details.get("channel", ""),
                language=transcript.language,
            )
            st.write(tr("add.step_index"))
            count = add_video(store, video, transcript.segments)
        except TranscriptError as error:
            status.update(state="error")
            st.session_state.flash = ("error", tr(f"add.{error.reason}"))
            return
        except Exception as error:
            log.exception("Adding a video failed")
            status.update(state="error")
            st.session_state.flash = ("error", tr("add.error", detail=error))
            return
    st.session_state.flash = ("success", tr("add.done", title=video.title, count=count))


# -- sidebar ----------------------------------------------------------------

def keep_one_language() -> None:
    """Clicking the selected language again would leave none selected: keep the current one."""
    choice = st.session_state.lang_choice
    if choice:
        st.session_state.lang = choice
    else:
        st.session_state.lang_choice = st.session_state.lang


videos = indexed_videos(store)
lang = st.session_state.lang

with st.sidebar:
    st.html(brand_html(lang))
    st.segmented_control("Language", options=list(LANGUAGES), format_func=str.upper, key="lang_choice",
                         default=lang, on_change=keep_one_language, label_visibility="collapsed")
    lang = st.session_state.lang

    st.html(stats_html(len(videos), chunk_count(store), config.LLM_CONTEXT, lang))

    with st.expander(f"＋ {tr('add.title')}"):
        with st.form("add_video", clear_on_submit=True, border=False):
            url = st.text_input(tr("add.url"), placeholder="https://youtu.be/…", help=tr("add.url_help"))
            topic = st.selectbox(tr("add.topic"), TOPICS, format_func=lambda key: t(f"topic.{key}", lang))
            if st.form_submit_button(tr("add.submit"), type="primary", width="stretch") and url:
                add_video_from_url(url, topic)
                st.rerun()
    if "flash" in st.session_state:
        kind, text = st.session_state.pop("flash")
        getattr(st, kind)(text)

    st.html(label_html(f"{tr('sidebar.videos')} · {len(videos)}"))
    st.html(videos_html(videos, lang) if videos else f"<p class='ba-sub'>{tr('sidebar.empty')}</p>")

    st.write("")
    if st.button(tr("sidebar.clear"), icon=":material/delete_sweep:", width="stretch",
                 disabled=not st.session_state.messages):
        st.session_state.messages = []
        st.rerun()


# -- conversation -----------------------------------------------------------

ASSISTANT_AVATAR = ":material/auto_awesome:"


def show_message(index: int, message: dict) -> None:
    role = message["role"]
    with st.container(key=f"msg-{role}-{index}"):
        with st.chat_message(role, avatar=ASSISTANT_AVATAR if role == "assistant" else None):
            st.markdown(message["content"])
            if role == "assistant" and "from_videos" in message:
                show_answer_details(message["from_videos"], [Source(**s) for s in message["sources"]])


def show_answer_details(from_videos: bool, sources: list[Source]) -> None:
    """Badge, "watch at 3:42" link and the excerpts used, under an answer."""
    st.html(meta_html(from_videos, sources, lang))
    if sources:
        with st.expander(tr("chat.sources", count=len(sources))):
            st.html(sources_html(sources, lang))


def topics_answer() -> str:
    lines = [f"- **{v.display_title(lang)}** · {t('topic.' + v.topic, lang) if v.topic in TOPICS else ''}"
             for v in videos]
    return tr("answer.topics", count=len(videos)) + "\n\n" + "\n".join(lines)


def reply(question: str) -> None:
    """Writes the answer to the last question, streamed token by token, and stores it."""
    history = st.session_state.messages[:-1]
    intent = classify(question)
    index = len(st.session_state.messages)

    with st.container(key=f"msg-assistant-{index}"):
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            if intent != "question":
                text = {"greeting": tr("answer.greeting"), "thanks": tr("answer.thanks"),
                        "topics": topics_answer()}[intent]
                st.markdown(text)
                st.session_state.messages.append({"role": "assistant", "content": text})
                return

            # The answer follows the language of the question; the interface language when unsure.
            answer = engine.answer(question, history, detect_language(question, default=lang))
            try:
                st.write_stream(answer)
            except Exception:
                log.exception("LLM call failed")
                st.session_state.messages.pop()        # the question can be asked again
                st.error(tr("chat.error", model=config.OLLAMA_MODEL))
                return
            show_answer_details(answer.from_videos, answer.sources)

    st.session_state.messages.append({
        "role": "assistant", "content": answer.text, "from_videos": answer.from_videos,
        "sources": [source.to_dict() for source in answer.sources],
    })


# -- page -------------------------------------------------------------------

typed = st.chat_input(tr("chat.placeholder"))
question = typed or st.session_state.pop("pending", None)
if question and len(question) > config.MAX_QUESTION_CHARS:
    st.toast(tr("chat.too_long", max=config.MAX_QUESTION_CHARS))
    question = None
if question:
    st.session_state.messages.append({"role": "user", "content": question})

main = st.container()

if not st.session_state.messages:
    main.html(hero_html(lang))
    if not videos:
        main.info(tr("setup.empty"))
    with main.container(key="suggestions"):
        suggestions = [tr(f"welcome.s{i}") for i in range(1, 5)]
        for row in (suggestions[:2], suggestions[2:]):
            for column, suggestion in zip(st.columns(2), row):
                if column.button(suggestion, key=f"suggest-{suggestion}", width="stretch"):
                    st.session_state.pending = suggestion
                    st.rerun()

with main:
    for i, message in enumerate(st.session_state.messages):
        show_message(i, message)
    if question:
        reply(question)
