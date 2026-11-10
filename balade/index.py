"""Vector store: embedding model, ChromaDB collection, adding a video and listing the indexed videos."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from balade import config
from balade.chunking import chunk_segments
from balade.transcripts import Segment, video_url

TOPICS = ("marine_biology", "astronomy", "oceanography", "physics", "other")


@dataclass
class VideoInfo:
    video_id: str
    title: str
    title_en: str = ""
    topic: str = "other"
    channel: str = ""
    language: str = ""

    def display_title(self, lang: str) -> str:
        return self.title_en if lang == "en" and self.title_en else self.title


# -- embeddings -------------------------------------------------------------

class E5Embeddings(Embeddings):
    """multilingual-e5 models were trained with "query: " and "passage: " prefixes.
    Without them, retrieval quality drops, so this wrapper adds them."""

    def __init__(self, base: Embeddings):
        self.base = base

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.base.embed_documents([f"passage: {text}" for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.base.embed_query(f"query: {text}")


def make_embeddings(provider: str = config.EMBEDDING_PROVIDER, model: str = config.EMBEDDING_MODEL) -> Embeddings:
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=model, base_url=config.OLLAMA_BASE_URL)

    # Imported here so the Ollama option works without PyTorch installed.
    import torch
    from langchain_huggingface import HuggingFaceEmbeddings

    base = HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )
    return E5Embeddings(base) if "e5" in model.lower() else base


def open_store(embeddings: Embeddings, persist_dir: Path = config.CHROMA_DIR,
               collection: str = config.COLLECTION_NAME) -> Chroma:
    """Opens (or creates) the persistent ChromaDB collection, using cosine distance."""
    return Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


# -- videos -----------------------------------------------------------------

def load_catalog(path: Path = config.VIDEOS_FILE) -> list[VideoInfo]:
    """The list of videos build_index.py indexes (data/videos.json)."""
    return [VideoInfo(**item) for item in json.loads(Path(path).read_text(encoding="utf-8"))]


def indexed_videos(store: Chroma) -> list[VideoInfo]:
    """Every video in the collection, in the order they were added. Read from the chunk metadata,
    so videos added from the app show up without any other file."""
    metadatas = store.get(include=["metadatas"])["metadatas"]
    videos: dict[str, tuple[float, VideoInfo]] = {}
    for meta in metadatas:
        video_id = meta.get("video_id")
        if video_id and video_id not in videos:
            videos[video_id] = (meta.get("added_at", 0), VideoInfo(
                video_id=video_id,
                title=meta.get("title") or video_id,
                title_en=meta.get("title_en", ""),
                topic=meta.get("topic", "other"),
                channel=meta.get("channel", ""),
                language=meta.get("language", ""),
            ))
    return [video for _, video in sorted(videos.values(), key=lambda item: item[0])]


def is_indexed(store: Chroma, video_id: str) -> bool:
    return bool(store.get(where={"video_id": video_id}, limit=1)["ids"])


def add_video(store: Chroma, video: VideoInfo, segments: list[Segment],
              size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> int:
    """Chunks a transcript, embeds the chunks and stores them. Returns the number of chunks
    added (0 if the video was already indexed)."""
    if is_indexed(store, video.video_id):
        return 0
    chunks = chunk_segments(segments, size, overlap)
    if not chunks:
        raise ValueError("The transcript is empty.")

    added_at = time.time()
    documents = [
        Document(
            page_content=chunk.text,
            metadata={
                "video_id": video.video_id,
                "title": video.title,
                "title_en": video.title_en,
                "topic": video.topic,
                "channel": video.channel,
                "language": video.language,
                "start": chunk.start,              # used for the "watch at 3:42" links
                "source": video_url(video.video_id, chunk.start),
                "added_at": added_at,
            },
        )
        for chunk in chunks
    ]
    # Stable ids: re-running the indexing never creates duplicates.
    ids = [f"{video.video_id}-{i:04d}" for i in range(len(documents))]
    store.add_documents(documents, ids=ids)
    return len(documents)


def chunk_count(store: Chroma) -> int:
    return store._collection.count()
