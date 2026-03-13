![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Project III — Multimodal AI ChatBot for YouTube Video Q&A

**Balade.AI** — A bilingual RAG-powered chatbot for science video Q&A, built on top of the French YouTube channel [Balade Mentale](https://www.youtube.com/@BaladeMentale).

---

##  Project Overview

Balade.AI allows users to ask questions — in French or English — about the content of science documentary videos and receive accurate, sourced answers drawn directly from the video transcripts. When a question falls outside the indexed content, the system transparently falls back to the LLM's general knowledge, clearly labelling the source of the answer.

The project covers the full RAG pipeline: from transcript extraction and chunking, to embedding and vector search, to LangGraph agent orchestration with conversational memory, to a production-ready Streamlit web interface.

---

##  Use Case

The dataset is 7 videos from **Balade Mentale**, a French science popularisation YouTube channel, covering the following topics:

| Video | Topic |
|---|---|
| Les pieuvres vampires des Abysses | Marine Biology |
| Des milliards d'autres planètes ! | Astronomy |
| LE GIGANTISME des Abysses | Marine Biology |
| Créatures géantes et transparentes | Marine Biology |
| La folle Histoire du système solaire | Astronomy |
| Ces lacs sous-marins qui ne devraient pas exister | Oceanography |
| Le froid maximal — Voyage vers le zéro absolu | Physics |

---

##  Architecture

```
YouTube URL
    │
    ▼
Transcript Extraction (youtube-transcript-api)
    │
    ▼
Text Cleaning + Chunking (LangChain RecursiveCharacterTextSplitter)
    │   chunk_size=800, chunk_overlap=150
    ▼
Multilingual Embeddings (intfloat/multilingual-e5-large, GPU/CUDA)
    │
    ▼
ChromaDB Vector Store (280 chunks, 7 videos, persistent local storage)
    │
    ▼
MMR Retriever (k=4, fetch_k=20, lambda_mult=0.7)
    │
    ▼
LangGraph ReAct Agent + 3 Tools + Conversational Memory
    │
    ▼
Local LLM (llama3.1:8b via Ollama, num_ctx=8192)
    │
    ▼
Streamlit Web Interface (bilingual FR/EN, dark theme)
```

---

##  Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangChain + LangGraph |
| Embeddings | `intfloat/multilingual-e5-large` (HuggingFace) |
| Vector Store | ChromaDB (local persistent) |
| LLM | `llama3.1:8b` via Ollama (local, no API key) |
| Agent | LangGraph `create_react_agent` (ReAct) |
| Observability | LangSmith |
| Web Interface | Streamlit |
| Language Detection | `langdetect` |
| Hardware | NVIDIA RTX 5070 (CUDA), 16GB DDR5 RAM |

---

##  Requirements Checklist

| Requirement | Status |
|---|---|
| RAG system with vector database | ✅ ChromaDB, 280 chunks |
| LangChain agents + tools | ✅ LangGraph ReAct agent, 3 tools |
| Conversational memory | ✅ 6-turn history injected into every prompt |
| Several tools | ✅ `rag_search`, `add_youtube_video`, `list_available_topics` |
| LangSmith integration | ✅ Full tracing on project `youtube-rag-balade-mentale` |
| Web app | ✅ Streamlit, bilingual FR/EN |
| Speech recognition | ⚠️ Attempted (Whisper) — blocked by `ffmpeg` DLL on Windows |

---

##  Tools Defined

**`rag_search(question: str)`** — The core RAG tool. It retrieves the 4 most relevant transcript chunks from ChromaDB using MMR search, injects them into a bilingual prompt, and asks the LLM to answer. If the answer is not found in the transcripts, the system automatically falls back to the LLM's general knowledge and clearly labels the response as such.

**`add_youtube_video(url: str)`** — Dynamically ingests a new YouTube video into ChromaDB at runtime: extracts the transcript, cleans it, splits it into chunks, embeds them, and adds them to the collection — all without restarting the app.

**`list_available_topics()`** — Returns a structured list of all indexed videos with their YouTube URLs and topics, so the agent can tell the user what content is available.

---

## Web Interface Features

The Streamlit interface (`app.py`) was built with a focus on both usability and visual quality.

Every response includes a source badge (`📹 From Balade Mentale transcripts` or `⚠️ General knowledge`), a confidence score showing the cosine similarity between the question and the best matching chunk, and a direct YouTube link to the source video. The interface supports full language switching between French and English — both the UI and the LLM responses adapt to the selected language. A sidebar displays all 7 indexed videos with colour-coded topic tags, live chunk count, and a conversation history cleared with one click.

---

##  Getting Started

**Prerequisites:** Conda, CUDA-compatible GPU (optional), [Ollama](https://ollama.com) installed and running.

```bash
# 1. Clone the repository
git clone <repo-url>
cd <repo-folder>

# 2. Create and activate the conda environment
conda create -n youtube-rag python=3.11
conda activate youtube-rag

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the LLM
ollama pull llama3.1:8b

# 5. Configure environment
cp .env.example .env
# Add your LANGCHAIN_API_KEY to .env

# 6. Run the notebook to build the vector store (Step 9 already done — 280 chunks)
jupyter notebook main.ipynb

# 7. Launch the web app
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

> **Note:** The `./chroma_db` folder contains the pre-built vector store. You do not need to re-run the ingestion step (Step 9) unless you want to add new videos.

---

##  Project Structure

```
├── main.ipynb          # Full RAG pipeline — step-by-step notebook
├── app.py              # Streamlit web interface
├── chroma_db/          # Persistent ChromaDB vector store (280 chunks)
├── .env                # API keys (not committed)
├── .gitignore
└── README.md
```

---

##  Key Design Decisions

**Why `multilingual-e5-large` for embeddings?** The videos are in French but users can ask in English. A monolingual French model would fail on English queries. `multilingual-e5-large` maps both languages to the same vector space, so a question like "What is bioluminescence?" retrieves the right French transcript chunks without any translation step.

**Why MMR instead of plain similarity search?** With 4 chunks retrieved per query, pure similarity search often returns near-duplicate passages from the same paragraph. MMR (Maximal Marginal Relevance) balances relevance and diversity, ensuring the LLM receives 4 genuinely different context windows — which leads to richer, more complete answers.

**Why `llama3.1:8b` locally instead of OpenAI?** Full data privacy, zero API cost, no rate limits, and better tool-calling performance compared to `mistral:7b` which was evaluated first. The RTX 5070 runs inference at a comfortable speed for an interactive chat interface.

**Why a hybrid RAG + fallback instead of pure RAG?** A pure RAG system that returns "not found" for out-of-scope questions creates a frustrating user experience. The hybrid approach keeps the chatbot useful for general science questions while being transparent about when it's drawing from the videos vs. its own knowledge.

---

##  Evaluation

Tracing and evaluation are handled via LangSmith under the project `youtube-rag-balade-mentale`. Every chain invocation, tool call, and retrieval step is logged, which allows post-hoc analysis of retrieval quality, answer faithfulness, and latency.

---

##  Author

**Benny Michel** — Ironhack AI Bootcamp, Final Project (2026)