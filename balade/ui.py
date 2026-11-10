"""Design: CSS and the small HTML pieces of the interface. Every text from the user, a video
or a transcript goes through html.escape before being put in HTML."""
from __future__ import annotations

from html import escape

import streamlit as st

from balade.i18n import t
from balade.index import TOPICS, VideoInfo
from balade.rag import Source
from balade.transcripts import thumbnail_url, video_url

CSS = """
<style>
:root {
  --bg: #0c0816; --panel: #120b20; --surface: #1a1130; --surface-2: #22173d;
  --border: #2f2352; --text: #efeaf9; --muted: #a197c2;
  --violet: #8b5cf6; --violet-2: #a78bfa; --magenta: #d946ef;
  --teal: #2dd4bf; --amber: #fbbf24;
}

/* Background: deep purple with two soft glows */
.stApp {
  background:
    radial-gradient(900px 520px at 88% -12%, rgba(139, 92, 246, .22), transparent 60%),
    radial-gradient(700px 420px at -12% 112%, rgba(217, 70, 239, .10), transparent 60%),
    var(--bg);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { max-width: 880px; padding-top: 2.5rem; padding-bottom: 7rem; }
[data-testid="stBottomBlockContainer"] { max-width: 880px; background: transparent; }
[data-testid="stBottom"] > div { background: linear-gradient(to top, var(--bg) 70%, transparent); }

/* Chat input */
[data-testid="stChatInput"] {
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: 18px !important; box-shadow: 0 10px 40px rgba(10, 4, 25, .55);
}
[data-testid="stChatInput"]:focus-within { border-color: var(--violet) !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--border); }
.ba-brand { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 1.6rem; letter-spacing: -.02em; line-height: 1; }
.ba-brand span { color: var(--violet-2); }
.ba-tagline { color: var(--muted); font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; margin: .45rem 0 .2rem; }
.ba-label { color: var(--muted); font-size: .68rem; letter-spacing: .14em; text-transform: uppercase; margin: 1.1rem 0 .5rem; }

.ba-videos { display: flex; flex-direction: column; gap: .4rem; }
a.ba-video {
  display: flex; gap: .65rem; align-items: center; padding: .45rem; border-radius: 12px;
  text-decoration: none !important; color: var(--text) !important; border: 1px solid transparent;
  transition: background .15s, border-color .15s;
}
a.ba-video:hover { background: var(--surface); border-color: var(--border); }
.ba-thumb {
  width: 76px; aspect-ratio: 16 / 9; border-radius: 8px; flex-shrink: 0; overflow: hidden;
  background-color: #2a1b4d; background-size: cover; background-position: center;
  box-shadow: inset 0 0 0 1px rgba(167, 139, 250, .18);
}
.ba-video-title { font-size: .8rem; line-height: 1.3; margin-bottom: .3rem;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }

.ba-tag { display: inline-block; font-size: .64rem; padding: .1rem .5rem; border-radius: 99px;
  border: 1px solid currentColor; opacity: .95; }
.ba-tag--marine_biology { color: var(--teal); }
.ba-tag--astronomy { color: #f472b6; }
.ba-tag--oceanography { color: #60a5fa; }
.ba-tag--physics { color: var(--violet-2); }
.ba-tag--other { color: var(--muted); }

.ba-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: .45rem; margin: 1rem 0 .6rem; }
.ba-stat { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: .55rem .3rem; text-align: center; }
.ba-stat b { display: block; font-family: 'Syne', sans-serif; font-size: 1.05rem; color: var(--violet-2); }
.ba-stat span { font-size: .62rem; color: var(--muted); letter-spacing: .08em; text-transform: uppercase; }

/* Welcome */
.ba-hero { text-align: center; padding: 3.2rem 0 1.6rem; }
.ba-kicker { display: inline-block; font-size: .72rem; letter-spacing: .12em; text-transform: uppercase;
  color: var(--violet-2); background: rgba(139, 92, 246, .12); border: 1px solid rgba(139, 92, 246, .35);
  padding: .3rem .8rem; border-radius: 99px; margin-bottom: 1.2rem; }
.ba-title { font-family: 'Syne', sans-serif; font-weight: 800; font-size: clamp(2.2rem, 6vw, 3.4rem);
  letter-spacing: -.03em; line-height: 1.05; margin: 0 0 1rem;
  background: linear-gradient(120deg, #f5f0ff 10%, var(--violet-2) 55%, var(--magenta));
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.ba-sub { color: var(--muted); max-width: 560px; margin: 0 auto; line-height: 1.65; font-size: 1rem; }

.st-key-suggestions [data-testid="stButton"] button {
  width: 100%; min-height: 4.2rem; justify-content: flex-start; text-align: left;
  background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
  color: var(--text); padding: .8rem 1rem; transition: border-color .15s, transform .15s, background .15s;
}
.st-key-suggestions [data-testid="stButton"] button:hover {
  border-color: var(--violet); background: var(--surface-2); transform: translateY(-1px); color: var(--text);
}

/* While the first answer streams, Streamlit greys out the old welcome buttons: hide them instead. */
[data-stale="true"][class*="st-key-suggest-"] { display: none; }

/* Messages */
[data-testid="stChatMessage"] { background: transparent; padding: .35rem 0; gap: .75rem; }
[class*="st-key-msg-user"] [data-testid="stChatMessage"] { flex-direction: row-reverse; justify-content: flex-start; }
[class*="st-key-msg-user"] [data-testid="stChatMessageAvatarUser"] { display: none; }
[class*="st-key-msg-user"] [data-testid="stChatMessageContent"] {
  flex: 0 1 auto; max-width: 78%; margin: 0 0 0 auto; padding: .65rem 1rem;
  background: linear-gradient(135deg, #6d28d9, #9333ea); color: #fff;
  border-radius: 18px 4px 18px 18px; box-shadow: 0 6px 24px rgba(124, 58, 237, .30);
}
[class*="st-key-msg-assistant"] [data-testid="stChatMessageContent"] {
  background: var(--surface); border: 1px solid var(--border); padding: .8rem 1.05rem;
  border-radius: 4px 18px 18px 18px; max-width: 92%;
}
[data-testid="stChatMessageContent"] p:last-child { margin-bottom: 0; }
[class*="st-key-msg-user"] [data-testid="stChatMessageContent"] p { margin: 0; color: #fff; }
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] { margin-bottom: 0; }
[data-testid="stChatMessageAvatarCustom"], [data-testid="stChatMessageAvatarAssistant"] {
  background: linear-gradient(135deg, var(--violet), var(--magenta)); color: #fff; border-radius: 10px;
}

.ba-meta { display: flex; flex-wrap: wrap; gap: .45rem; align-items: center; margin-top: .7rem; }
.ba-badge { font-size: .72rem; padding: .18rem .6rem; border-radius: 99px; border: 1px solid; }
.ba-badge--videos { color: var(--teal); border-color: rgba(45, 212, 191, .35); background: rgba(45, 212, 191, .08); }
.ba-badge--general { color: var(--amber); border-color: rgba(251, 191, 36, .35); background: rgba(251, 191, 36, .08); }
a.ba-watch { font-size: .74rem; padding: .18rem .65rem; border-radius: 99px; text-decoration: none !important;
  color: #fff !important; background: linear-gradient(135deg, var(--violet), var(--magenta)); }
a.ba-watch:hover { filter: brightness(1.12); }

[class*="st-key-msg-assistant"] [data-testid="stExpander"] details {
  border: 1px solid var(--border); border-radius: 12px; background: rgba(12, 8, 22, .35); margin-top: .6rem; }
[class*="st-key-msg-assistant"] [data-testid="stExpander"] summary { font-size: .8rem; color: var(--muted); }
.ba-source { padding: .55rem 0; border-top: 1px solid var(--border); }
.ba-source:first-child { border-top: none; padding-top: 0; }
.ba-source a { font-size: .78rem; color: var(--violet-2) !important; text-decoration: none; font-weight: 600; }
.ba-source p { font-size: .8rem; color: var(--muted); margin: .25rem 0 0; line-height: 1.5; }

@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"] { padding-top: 1.2rem; }
  .ba-hero { padding-top: 1.6rem; }
  [class*="st-key-msg-user"] [data-testid="stChatMessageContent"] { max-width: 88%; }
}
</style>
"""


def apply_style() -> None:
    st.html(CSS)


def brand_html(lang: str) -> str:
    return f"<div class='ba-brand'>Balade<span>.</span>AI</div><div class='ba-tagline'>{escape(t('brand.tagline', lang))}</div>"


def label_html(text: str) -> str:
    return f"<div class='ba-label'>{escape(text)}</div>"


def videos_html(videos: list[VideoInfo], lang: str) -> str:
    cards = []
    for video in videos:
        topic = video.topic if video.topic in TOPICS else "other"
        cards.append(
            f"<a class='ba-video' href='{escape(video_url(video.video_id))}' target='_blank' rel='noopener'>"
            # Background image rather than <img>: offline, the purple gradient shows instead of a broken icon.
            f"<div class='ba-thumb' style=\"background-image: url('{escape(thumbnail_url(video.video_id))}')\"></div>"
            f"<div><div class='ba-video-title'>{escape(video.display_title(lang))}</div>"
            f"<span class='ba-tag ba-tag--{topic}'>{escape(t('topic.' + topic, lang))}</span></div></a>"
        )
    return f"<div class='ba-videos'>{''.join(cards)}</div>"


def stats_html(videos: int, chunks: int, context_tokens: int, lang: str) -> str:
    items = [(videos, t("stats.videos", lang)), (chunks, t("stats.chunks", lang)),
             (f"{context_tokens // 1000}k", t("stats.context", lang))]
    tiles = "".join(f"<div class='ba-stat'><b>{escape(str(v))}</b><span>{escape(label)}</span></div>"
                    for v, label in items)
    return f"<div class='ba-stats'>{tiles}</div>"


def hero_html(lang: str) -> str:
    return (f"<div class='ba-hero'><div class='ba-kicker'>{escape(t('welcome.kicker', lang))}</div>"
            f"<h1 class='ba-title'>{escape(t('welcome.title', lang))}</h1>"
            f"<p class='ba-sub'>{escape(t('welcome.sub', lang))}</p></div>")


def meta_html(from_videos: bool, sources: list[Source], lang: str) -> str:
    if not from_videos:
        return f"<div class='ba-meta'><span class='ba-badge ba-badge--general'>{escape(t('badge.general', lang))}</span></div>"
    parts = [f"<span class='ba-badge ba-badge--videos'>{escape(t('badge.videos', lang))}</span>"]
    if sources:
        best = sources[0]   # the most relevant excerpt
        label = f"▶ {t('chat.watch', lang, time=best.timestamp)} · {best.display_title(lang)}"
        parts.append(f"<a class='ba-watch' href='{escape(best.url)}' target='_blank' rel='noopener'>{escape(label)}</a>")
    return f"<div class='ba-meta'>{''.join(parts)}</div>"


def sources_html(sources: list[Source], lang: str, max_chars: int = 320) -> str:
    items = []
    for source in sources:
        excerpt = source.excerpt if len(source.excerpt) <= max_chars else source.excerpt[:max_chars].rsplit(" ", 1)[0] + "…"
        items.append(
            f"<div class='ba-source'><a href='{escape(source.url)}' target='_blank' rel='noopener'>"
            f"▶ {escape(source.timestamp)} · {escape(source.display_title(lang))}</a>"
            f"<p>{escape(excerpt)}</p></div>"
        )
    return "".join(items)
