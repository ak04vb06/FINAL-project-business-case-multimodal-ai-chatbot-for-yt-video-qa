"""Settings. Every value can be changed in a `.env` file at the project root (see .env.example)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name, "").strip() or default


# -- data -------------------------------------------------------------------
CHROMA_DIR = Path(_env("CHROMA_DIR", str(ROOT / "chroma_db")))
COLLECTION_NAME = "balade_mentale_videos"
VIDEOS_FILE = ROOT / "data" / "videos.json"          # the videos indexed by build_index.py
TRANSCRIPTS_DIR = ROOT / "data" / "transcripts"      # local cache, so YouTube is asked only once
TRANSCRIPT_LANGUAGES = ("fr", "en")

# -- embeddings -------------------------------------------------------------
# "huggingface" runs the model locally with sentence-transformers (GPU if available);
# "ollama" asks Ollama for the embeddings (lighter install, no PyTorch needed).
EMBEDDING_PROVIDER = _env("EMBEDDING_PROVIDER", "huggingface")
EMBEDDING_MODEL = _env(
    "EMBEDDING_MODEL",
    "intfloat/multilingual-e5-large" if EMBEDDING_PROVIDER == "huggingface" else "bge-m3",
)

# -- language model (Ollama) ------------------------------------------------
OLLAMA_MODEL = _env("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = _env("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_TEMPERATURE = 0.1
# Context window in tokens. llama3.1 supports up to 128k, but every token kept in the
# window uses GPU memory: 32k fits on a 12 GB card next to the 8B model.
LLM_CONTEXT = int(_env("LLM_CONTEXT", "32768"))
ANSWER_TOKENS = 1024                                  # room kept free for the answer itself

# -- retrieval --------------------------------------------------------------
CHUNK_SIZE = 800          # characters per chunk
CHUNK_OVERLAP = 150       # characters shared by two consecutive chunks
TOP_K = int(_env("TOP_K", "6"))            # excerpts given to the model
FETCH_K = int(_env("FETCH_K", "30"))       # candidates MMR picks from
MMR_LAMBDA = 0.7          # 1 = only relevance, 0 = only diversity

# -- conversation -----------------------------------------------------------
HISTORY_MESSAGES = int(_env("HISTORY_MESSAGES", "20"))   # messages of memory sent with each question
MAX_QUESTION_CHARS = 2000

# -- voice questions (optional: pip install faster-whisper) -------------------
WHISPER_MODEL = _env("WHISPER_MODEL", "small")      # tiny, base, small, medium, large-v3
WHISPER_DEVICE = _env("WHISPER_DEVICE", "cpu")      # "cuda" needs the CUDA libraries of CTranslate2
# "auto" tries faster-whisper, then Whisper through transformers/PyTorch (see balade/voice.py).
WHISPER_BACKEND = _env("WHISPER_BACKEND", "auto")   # auto, faster-whisper, transformers

# -- feedback ---------------------------------------------------------------
FEEDBACK_FILE = ROOT / "data" / "feedback.jsonl"

# -- LangSmith (optional) ---------------------------------------------------
LANGSMITH_PROJECT = _env("LANGCHAIN_PROJECT", "youtube-rag-balade-mentale")


def enable_tracing() -> bool:
    """Turns LangSmith tracing on only when a key is configured (no key: no tracing, no warnings)."""
    if os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", LANGSMITH_PROJECT)
        return True
    return False
