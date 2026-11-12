# Balade.AI · Chat with YouTube science videos (RAG)

Ask a question about the videos of the French science channel
[Balade Mentale](https://www.youtube.com/@BaladeMentale), in **French or English**, and get an answer
built from what the videos actually say, with a link that opens the video **at the right moment**.

Runs **100% locally**: a Llama 3.1 model through [Ollama](https://ollama.com), multilingual embeddings,
and a ChromaDB vector store. No API key and no cost per question.

> Started as my final project at the Ironhack AI Engineering bootcamp (2026), then reworked:
> new interface, hybrid search, streamed answers with sources and timestamps, video summaries,
> voice questions, adding videos from the app, a larger context window, and tests.

## What it does

- **Answers from the videos.** Each answer is generated from the most relevant transcript excerpts
  and marked **From the videos**.
- **Takes you to the right moment.** Every answer has a **Watch at 3:42** link, and a **Sources**
  panel lists the excerpts used, each with its own timestamp link.
- **Streams the answer** word by word, like a chat assistant.
- **Says when the videos don't cover a question.** It answers from general knowledge instead, clearly
  labelled **General knowledge**, rather than inventing what a video says.
- **French or English.** The videos are in French; questions and answers can be in either language,
  and the interface switches between FR and EN.
- **Remembers the conversation.** Follow-up questions such as "and why?" work: the recent messages
  are sent with each question, within a 32k-token context window.
- **Summarises a whole video** with its key moments, each one a link to that moment.
- **Search in one video.** Pick a video in the sidebar to limit the answers to it.
- **Voice questions.** Click the microphone in the message box and ask out loud (optional, with
  faster-whisper, running locally).
- **Add any YouTube video from the sidebar.** Paste a link: the captions are downloaded, chunked,
  embedded and indexed in a few seconds, without restarting the app. Videos can also be removed.
- **Choose the model** among those installed in Ollama, **export the conversation** as Markdown,
  **regenerate** an answer, and rate answers with thumbs up / down (saved in `data/feedback.jsonl` to spot the
  questions the bot answers badly).

## How it works

```
Indexing (build_index.py, or "Add a video" in the app)
  YouTube captions with timestamps (youtube-transcript-api, cached locally)
    → chunks of ~800 characters, 150 overlap, never cutting a caption line, each keeping its start time
    → multilingual-e5-large embeddings ("passage: " prefix)
    → ChromaDB (cosine distance, persisted in ./chroma_db)

Answering (app.py → balade/rag.py)
  question → small talk / "which videos?" handled directly, without the model
           → hybrid search, optionally limited to one video:
               MMR vector search ("query: " prefix) + BM25 keyword search, merged with Reciprocal Rank Fusion
               → 6 excerpts
           → prompt = excerpts with title + timestamp, recent conversation, question, answer language
           → Llama 3.1 8B via Ollama, streamed token by token
           → model replies NOT_IN_VIDEOS? → the marker is hidden and a general-knowledge answer follows
```

Design choices:

- **Chunks keep their timestamp.** Chunks are built from whole caption lines instead of splitting a
  flat text, so each chunk knows its exact start time in the video. That is what makes the
  "watch at 3:42" links possible.
- **Multilingual embeddings with the right prefixes.** `multilingual-e5-large` maps French and English
  to the same vector space, so an English question finds French passages without translation.
  E5 models are trained with `query: ` / `passage: ` prefixes; the app adds them (see `E5Embeddings`).
- **MMR instead of plain similarity.** Plain similarity often returns near-duplicate passages from the
  same paragraph; MMR balances relevance and diversity, so the model gets richer context.
- **Hybrid search.** Vector search understands paraphrases and works across languages, but can miss
  exact terms such as names or numbers ("273.15", "Kelvin"). BM25 finds those exact words.
  Reciprocal Rank Fusion merges both rankings: excerpts found by both searches come first.
- **A hidden "not found" marker.** The model must answer `NOT_IN_VIDEOS` when the excerpts don't
  contain the answer. The stream holds back the first characters until it knows they are not the
  marker, so the user never sees it, and switches to a clearly labelled general answer.
- **Context budget.** The window is 32k tokens by default (`LLM_CONTEXT`). The prompt keeps the most
  recent messages that fit and drops the oldest first, so a long conversation never overflows it.
  The same window lets the model read a whole video transcript at once to summarise it (longer
  videos are shortened evenly to fit).
- **Voice without ffmpeg.** The first version tried openai-whisper, which needs ffmpeg and failed on
  Windows. faster-whisper decodes audio with PyAV instead, and Streamlit's chat input records the
  microphone directly.
- **Direct RAG in the app, agent in the notebook.** The notebook builds a LangGraph ReAct agent with
  three tools (search, add a video, list topics). In the app, a direct pipeline is faster and more
  reliable with an 8B model: it calls the model once per answer and can stream it.

## Tech stack

Python · LangChain · ChromaDB · sentence-transformers (`intfloat/multilingual-e5-large`) · rank-bm25 ·
Ollama (`llama3.1:8b`) · faster-whisper · Streamlit · youtube-transcript-api ·
LangGraph and LangSmith (notebook) · pytest

## Run it locally

You need Python 3.11+, [Ollama](https://ollama.com) and, ideally, an NVIDIA GPU (it also runs on CPU, more slowly).

```bash
git clone https://github.com/ak04vb06/balade-ai-youtube-rag.git
cd balade-ai-youtube-rag

python -m venv .venv
.venv\Scripts\activate                 # macOS / Linux: source .venv/bin/activate

# NVIDIA GPU: install PyTorch with CUDA first (otherwise skip this line)
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

ollama pull llama3.1:8b
```

Build the index. It downloads the captions of the 7 videos listed in `data/videos.json` and, the
first time, the embedding model (about 2 GB):

```bash
python build_index.py
```

Start the app:

```bash
streamlit run app.py
```

Open http://localhost:8501. Stop the app before running `build_index.py` again: two programs
writing to the same ChromaDB folder at once can corrupt the running app's view of it.

**Voice questions (optional):** `pip install faster-whisper`. The microphone button then appears in
the message box. The first question downloads the Whisper model (`small`, about 500 MB).
On Windows 11, Smart App Control can refuse to load faster-whisper's libraries (`WinError 4551`);
the app then switches by itself to Whisper through `transformers` and PyTorch, which are already
installed for the embeddings.

**Lighter option, without PyTorch:** set `EMBEDDING_PROVIDER=ollama` in `.env`, run
`ollama pull bge-m3`, and skip the torch / sentence-transformers lines. Rebuild the index after
changing the embedding model (`python build_index.py --rebuild`).

### Settings

Copy `.env.example` to `.env` to change them. Every setting is optional.

| Setting | Default | What it does |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.1:8b` | Model used to answer |
| `LLM_CONTEXT` | `32768` | Context window in tokens (llama3.1 supports up to 131072) |
| `EMBEDDING_PROVIDER` | `huggingface` | `huggingface` (sentence-transformers) or `ollama` |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | Embedding model (`bge-m3` with Ollama) |
| `TOP_K` | `6` | Excerpts given to the model for each question |
| `HISTORY_MESSAGES` | `20` | Messages of conversation memory |
| `WHISPER_MODEL` | `small` | Speech recognition model (`base` is faster, `medium` more accurate) |
| `WHISPER_DEVICE` | `cpu` | `cuda` to run Whisper on the GPU |
| `WHISPER_BACKEND` | `auto` | `auto` (faster-whisper, then transformers), `faster-whisper` or `transformers` |
| `LANGCHAIN_API_KEY` | | Turns on LangSmith tracing when set |

**About the context window.** Every token in the window uses GPU memory (about 128 KB per token
for Llama 3.1 8B, so 4 GB at 32k, on top of the 5 GB model). If `ollama ps` shows the model is not
100% on the GPU, answers get slower: let Ollama compress its cache by setting the environment
variables `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0` and restarting Ollama (this
halves that memory), or lower `LLM_CONTEXT` to `16384`. With the compressed cache and enough GPU
memory, `LLM_CONTEXT=65536` works too.

### Tests

```bash
pytest -q
```

53 tests cover link parsing, timestamps, the transcript cache, chunking, language detection,
small-talk detection, the conversation budget, indexing and removal, hybrid search, the video
filter, summaries, export, feedback, the model list, speech recognition and its fallback (with a fake Whisper), and
the full answer flow (streaming, sources, hidden marker, fallback) with a fake model and fake
embeddings. No GPU, model or internet connection needed.

## Project structure

```
app.py               Streamlit interface
build_index.py       Builds the ChromaDB index from data/videos.json
balade/
  config.py          Settings (read from .env)
  transcripts.py     YouTube links, captions with timestamps, local cache, video titles
  chunking.py        Caption lines → overlapping chunks that keep their start time
  index.py           Embeddings, ChromaDB, add a video, list indexed videos
  search.py          BM25 keyword index and Reciprocal Rank Fusion
  rag.py             Language, small talk, hybrid retrieval, prompts, streamed answer, fallback
  summary.py         Whole-video summary with linked key moments
  voice.py           Speech to text (faster-whisper, with a transformers fallback)
  models.py          Chat models installed in Ollama
  export.py          Conversation → Markdown
  feedback.py        Thumbs up / down saved as JSON Lines
  i18n.py            Interface texts (FR / EN)
  ui.py              Dark purple theme and HTML pieces (all text escaped)
data/videos.json     The 7 indexed videos
main.ipynb           Step-by-step notebook: pipeline, LangGraph agent, LangSmith evaluation
tests/               pytest
docs/                Bootcamp presentation and LangSmith evaluation
```

The transcripts and the vector store are created on your machine (`data/transcripts/`, `chroma_db/`)
and are not part of the repository: the video content belongs to Balade Mentale.

## Evaluation

The notebook evaluates the RAG pipeline with LangSmith on a ground-truth set of questions
(correctness and faithfulness, judged by an LLM), including an out-of-scope question to check
the fallback. See `docs/langsmith-evaluation.jpg`.

## Limits

- Answers depend on YouTube's captions; auto-generated captions can contain transcription errors.
- YouTube sometimes blocks caption downloads after many requests; transcripts are cached so each
  video is downloaded once.
- An 8B local model is fast and free but less precise than large hosted models.
- Voice input (Whisper) was explored in the notebook but is not part of the app.

## Author

**Benny Michel Vako**, AI Engineer · Ironhack AI Engineering bootcamp, 2026
