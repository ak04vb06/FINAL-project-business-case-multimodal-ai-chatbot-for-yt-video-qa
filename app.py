import streamlit as st
import os
import torch
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langdetect import detect as langdetect_detect

# ── TRANSLATIONS ──────────────────────────────────────────────────────────────
T = {
    "fr": {
        "assistant_sub": "Assistant Scientifique",
        "videos_label": "Vidéos disponibles",
        "stats_videos": "Vidéos",
        "stats_chunks": "Chunks",
        "stats_lang": "Langues",
        "clear": "🗑️ Effacer la conversation",
        "welcome_title": "Explorez la science",
        "welcome_sub": "Posez vos questions sur les vidéos Balade Mentale.<br>Je réponds depuis les transcripts réels en FR ou EN.",
        "suggestions": [
            ("🦑", "Qu'est-ce que la bioluminescence ?"),
            ("🪐", "Comment s'est formé le système solaire ?"),
            ("🌊", "Pourquoi ces lacs sous-marins existent ?"),
            ("❄️", "C'est quoi le zéro absolu ?"),
        ],
        "placeholder": "Posez votre question sur les vidéos Balade Mentale...",
        "searching": "🔍 Recherche...",
        "loading": "⏳ Chargement...",
        "source_badge": "📹 Depuis les transcripts Balade Mentale",
        "warning_badge": "⚠️ Connaissances générales",
        "error_msg": "Assure-toi que ChromaDB existe et qu'Ollama tourne en arrière-plan.",
        "greeting": "Bonjour ! Je suis votre assistant spécialisé dans les vidéos Balade Mentale. Posez-moi des questions sur la science ! 🎬",
        "topics_answer": "📚 Je connais 7 vidéos Balade Mentale : biologie marine (pieuvres, gigantisme, créatures transparentes), astronomie (planètes, système solaire), océanographie (lacs sous-marins) et physique (zéro absolu). Posez vos questions !",
    },
    "en": {
        "assistant_sub": "Scientific Assistant",
        "videos_label": "Available videos",
        "stats_videos": "Videos",
        "stats_chunks": "Chunks",
        "stats_lang": "Languages",
        "clear": "🗑️ Clear conversation",
        "welcome_title": "Explore science",
        "welcome_sub": "Ask questions about Balade Mentale videos.<br>I answer from real transcripts in FR or EN.",
        "suggestions": [
            ("🦑", "What is bioluminescence?"),
            ("🪐", "How was the solar system formed?"),
            ("🌊", "Why do underwater lakes exist?"),
            ("❄️", "What is absolute zero?"),
        ],
        "placeholder": "Ask your question about Balade Mentale videos...",
        "searching": "🔍 Searching...",
        "loading": "⏳ Loading...",
        "source_badge": "📹 From Balade Mentale transcripts",
        "warning_badge": "⚠️ General knowledge",
        "error_msg": "Make sure ChromaDB exists and Ollama is running in the background.",
        "greeting": "Hello! I'm your assistant specialized in Balade Mentale videos. Ask me anything about science! 🎬",
        "topics_answer": "📚 I know 7 Balade Mentale videos: marine biology (vampire squid, gigantism, transparent creatures), astronomy (planets, solar system), oceanography (underwater lakes) and physics (absolute zero). Ask away!",
    }
}

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Balade Mentale AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400&display=swap');

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

.stApp {
    background: #0a0a0f !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #e8e8f0 !important;
}

.stApp *, .stApp p, .stApp div, .stApp span, .stApp label,
.stApp h1, .stApp h2, .stApp h3, .stMarkdown, .stMarkdown p {
    color: #e8e8f0 !important;
}

[data-testid="stMain"], [data-testid="block-container"] {
    background: #0a0a0f !important;
}

[data-testid="stChatInput"] textarea {
    color: #e8e8f0 !important;
    background: #1a1a28 !important;
}

.stTextInput input {
    color: #e8e8f0 !important;
    background: #1a1a28 !important;
    border: 1px solid #2a2a3d !important;
}

.stButton > button {
    background: #1a1a28 !important;
    border: 1px solid #2a2a3d !important;
    color: #e8e8f0 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stButton > button:hover {
    border-color: #6c63ff !important;
    color: #6c63ff !important;
}

[data-testid="stSidebar"] {
    background: #12121a !important;
    border-right: 1px solid #2a2a3d !important;
}
[data-testid="stSidebar"] > div { padding: 0 !important; }
[data-testid="stSidebar"] * { color: #e8e8f0 !important; }

.stChatInput > div {
    background: #1a1a28 !important;
    border: 1px solid #2a2a3d !important;
    border-radius: 16px !important;
}
.stChatInput textarea {
    color: #e8e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Lang toggle */
.lang-toggle {
    display: flex;
    gap: 8px;
    padding: 12px 20px;
    border-bottom: 1px solid #2a2a3d;
}

/* Chat messages */
.user-message {
    display: flex;
    justify-content: flex-end;
    margin: 12px 0;
    animation: slideUp 0.3s ease;
}
.user-bubble {
    background: #6c63ff;
    color: white !important;
    padding: 12px 18px;
    border-radius: 18px;
    border-top-right-radius: 4px;
    max-width: 70%;
    font-size: 14px;
    line-height: 1.6;
}
.bot-message {
    display: flex;
    gap: 12px;
    margin: 12px 0;
    align-items: flex-start;
    animation: slideUp 0.3s ease;
}
.bot-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6c63ff, #00d4aa);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
}
.bot-bubble {
    background: #1a1a28;
    border: 1px solid #2a2a3d;
    color: #e8e8f0 !important;
    padding: 12px 18px;
    border-radius: 18px;
    border-top-left-radius: 4px;
    max-width: 75%;
    font-size: 14px;
    line-height: 1.6;
}
.source-badge {
    display: inline-block;
    margin-top: 8px;
    font-size: 11px;
    color: #00d4aa !important;
    background: rgba(0,212,170,0.1);
    padding: 3px 10px;
    border-radius: 4px;
    border: 1px solid rgba(0,212,170,0.2);
}
.warning-badge {
    display: inline-block;
    margin-top: 8px;
    font-size: 11px;
    color: #ff9f43 !important;
    background: rgba(255,159,67,0.1);
    padding: 3px 10px;
    border-radius: 4px;
    border: 1px solid rgba(255,159,67,0.2);
}

/* Welcome */
.welcome-container { text-align: center; padding: 60px 20px; }
.welcome-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 36px;
    background: linear-gradient(135deg, #e8e8f0 0%, #00d4aa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 12px;
}
.welcome-sub {
    color: #6b6b8a !important;
    font-size: 15px;
    max-width: 500px;
    margin: 0 auto 32px;
    line-height: 1.6;
}

/* Video cards */
.video-card {
    padding: 10px 16px;
    margin: 2px 8px;
    border-radius: 8px;
    border-left: 2px solid transparent;
    transition: all 0.15s;
}
.video-card:hover { background: #1a1a28; border-left-color: #6c63ff; }
.video-card-title { font-size: 12px; color: #e8e8f0 !important; line-height: 1.4; margin-bottom: 4px; }
.tag { display: inline-block; font-size: 10px; padding: 2px 8px; border-radius: 20px; }
.tag-bio   { color: #00d4aa !important; background: rgba(0,212,170,0.1); border: 1px solid rgba(0,212,170,0.2); }
.tag-astro { color: #ff9f43 !important; background: rgba(255,159,67,0.1); border: 1px solid rgba(255,159,67,0.2); }
.tag-phys  { color: #6c63ff !important; background: rgba(108,99,255,0.1); border: 1px solid rgba(108,99,255,0.2); }
.tag-ocean { color: #54a0ff !important; background: rgba(84,160,255,0.1); border: 1px solid rgba(84,160,255,0.2); }

/* Stats */
.confidence-badge {
    display: inline-block;
    margin-top: 8px;
    margin-left: 6px;
    font-size: 11px;
    color: #6c63ff !important;
    background: rgba(108,99,255,0.1);
    padding: 3px 10px;
    border-radius: 4px;
    border: 1px solid rgba(108,99,255,0.2);
}
.source-link {
    display: inline-block;
    margin-top: 8px;
    margin-left: 6px;
    font-size: 11px;
    color: #54a0ff !important;
    text-decoration: none;
    padding: 3px 10px;
    border-radius: 4px;
    border: 1px solid rgba(84,160,255,0.2);
    background: rgba(84,160,255,0.05);
}
.source-link:hover { border-color: #54a0ff !important; }
.stats-row {
    display: flex;
    justify-content: space-around;
    padding: 16px;
    border-top: 1px solid #2a2a3d;
    margin-top: 8px;
}
.stat-item { text-align: center; }
.stat-value { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 20px; color: #00d4aa !important; }
.stat-label { font-size: 10px; color: #6b6b8a !important; text-transform: uppercase; letter-spacing: 0.5px; }

@keyframes slideUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Prevent sidebar from collapsing ────────────────────────────────────────── */
[data-testid="collapsedControl"] {
    display: none !important;
}
section[data-testid="stSidebar"] {
    min-width: 260px !important;
    max-width: 320px !important;
    transform: translateX(0) !important;
    visibility: visible !important;
    pointer-events: auto !important;
}
</style>
""", unsafe_allow_html=True)

# ── CONFIG ────────────────────────────────────────────────────────────────────
CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "balade_mentale_sciences"
EMBED_MODEL     = "intfloat/multilingual-e5-large"
LLM_MODEL       = "llama3.1:8b"
LLM_CTX         = 8192
LANGSMITH_PROJECT = "youtube-rag-balade-mentale"

# ── LOAD ENV ──────────────────────────────────────────────────────────────────
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]    = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]    = LANGSMITH_PROJECT

# ── VIDEOS METADATA ───────────────────────────────────────────────────────────
VIDEOS_META = [
    {"title": "Les pieuvres vampires des Abysses",    "title_en": "Vampire Squid from the Abyss",          "tag": "bio",   "tag_label": "biologie marine", "tag_label_en": "marine biology",  "id": "300eyzsqang", "url": "https://youtu.be/300eyzsqang"},
    {"title": "Des milliards d'autres planetes !",   "title_en": "Billions of other planets!",            "tag": "astro", "tag_label": "astronomie",      "tag_label_en": "astronomy",        "id": "BXebijm_Mao", "url": "https://youtu.be/BXebijm_Mao"},
    {"title": "LE GIGANTISME des Abysses",            "title_en": "The GIGANTISM of the Abyss",            "tag": "bio",   "tag_label": "biologie marine", "tag_label_en": "marine biology",  "id": "2KQYrXJ_0N4", "url": "https://youtu.be/2KQYrXJ_0N4"},
    {"title": "Creatures geantes et transparentes",   "title_en": "Giant transparent creatures",           "tag": "bio",   "tag_label": "biologie marine", "tag_label_en": "marine biology",  "id": "NDeyeTQ-ZAM", "url": "https://youtu.be/NDeyeTQ-ZAM"},
    {"title": "La folle Histoire du systeme solaire", "title_en": "The crazy History of our solar system", "tag": "astro", "tag_label": "astronomie",      "tag_label_en": "astronomy",        "id": "2--n3nStqTk", "url": "https://youtu.be/2--n3nStqTk"},
    {"title": "Ces lacs sous-marins impossibles",     "title_en": "Underwater lakes that shouldn't exist","tag": "ocean", "tag_label": "oceanographie",   "tag_label_en": "oceanography",     "id": "86-T834tOU8", "url": "https://youtu.be/86-T834tOU8"},
    {"title": "Le froid maximal - Zero absolu",       "title_en": "Maximum cold - Absolute zero",          "tag": "phys",  "tag_label": "physique",        "tag_label_en": "physics",          "id": "NFJk2fmIidU", "url": "https://youtu.be/NFJk2fmIidU"},
]
VIDEO_MAP = {v["id"]: v for v in VIDEOS_META}

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "lang" not in st.session_state:
    st.session_state.lang = "fr"

# ── INIT PIPELINE ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ...")
def load_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings_model = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32}
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings_model,
        collection_name=COLLECTION_NAME
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.7}
    )

    llm = ChatOllama(model=LLM_MODEL, temperature=0.1, num_ctx=LLM_CTX)

    RAG_PROMPT_FR = PromptTemplate(
        input_variables=["context", "question"],
        template="""Tu es un assistant expert spécialisé dans les vidéos Balade Mentale.
Réponds UNIQUEMENT en te basant sur les extraits de transcript fournis.
Si la réponse n'est pas dans le contexte, dis clairement "Cette information n'est pas dans la vidéo."
Réponds de façon claire et pédagogique.

CONTEXTE :
{context}

QUESTION : {question}

RÉPONSE :"""
    )

    RAG_PROMPT_EN = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are an expert assistant specialized in Balade Mentale science videos.
Answer ONLY based on the transcript excerpts provided.
If the answer is not in the context, clearly say "This information is not in the video."
You MUST respond in English.

CONTEXT:
{context}

QUESTION: {question}

ANSWER IN ENGLISH:"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def detect_language(text):
        english_words = ["what", "how", "why", "when", "where", "who", "is", "are", "the", "does", "do", "can", "tell", "explain"]
        words = text.lower().split()
        if any(w in english_words for w in words):
            return "en"
        try:
            return langdetect_detect(text)
        except Exception:
            return "fr"

    def rag_search_fn(question, history=[]):
        lang = detect_language(question)
        prompt = RAG_PROMPT_FR if lang == "fr" else RAG_PROMPT_EN

        # Recherche avec score de confiance
        try:
            results = vectorstore.similarity_search_with_relevance_scores(question, k=4)
            docs = [d for d, _ in results]
            confidence = int(results[0][1] * 100) if results else 70
        except Exception:
            docs = vectorstore.similarity_search(question, k=4)
            confidence = 70

        # Détection de la vidéo source depuis les métadonnées du premier chunk
        source_video = None
        if docs:
            vid_id = docs[0].metadata.get("video_id", "")
            source_video = VIDEO_MAP.get(vid_id)

        # Historique des 6 derniers échanges pour le contexte
        history_text = ""
        for h in history[-6:]:
            role = ("Utilisateur" if lang == "fr" else "User") if h["role"] == "user" else "Assistant"
            history_text += f"{role}: {h['content'][:200]}\n"
        full_q = f"{history_text}Question: {question}" if history_text else question

        context = format_docs(docs)
        chain = ({"context": lambda _: context, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
        answer = chain.invoke(full_q)

        # Fallback hybride si pas trouvé dans les vidéos
        not_found = "pas dans la vidéo" in answer.lower() or "not in the video" in answer.lower()
        if not_found:
            if lang == "fr":
                fallback_p = f"Réponds de façon claire et pédagogique : {question}"
                prefix = "⚠️ Cette information n'est pas dans les vidéos Balade Mentale, mais voici une réponse générale :\n\n"
            else:
                fallback_p = f"Answer clearly and educationally: {question}"
                prefix = "⚠️ This information is not in the Balade Mentale videos, but here is a general answer:\n\n"
            answer = prefix + llm.invoke(fallback_p).content
            confidence = 0
            source_video = None

        return answer, not_found, confidence, source_video

    return {
        "rag_search": rag_search_fn,
        "chunk_count": vectorstore._collection.count(),
    }

# ── LOAD PIPELINE ─────────────────────────────────────────────────────────────
try:
    pipeline       = load_pipeline()
    rag_search     = pipeline["rag_search"]
    chunk_count    = pipeline["chunk_count"]
    pipeline_ready = True
except Exception as e:
    pipeline_ready = False
    pipeline_error = str(e)

# ── LANG SHORTCUT ─────────────────────────────────────────────────────────────
lang = st.session_state.lang
tx   = T[lang]

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown(f"""
    <div style='padding:24px 20px 12px;border-bottom:1px solid #2a2a3d;'>
        <div style='font-family:Syne,sans-serif;font-weight:800;font-size:22px;color:#e8e8f0;'>
            Balade<span style='color:#00d4aa;'>.</span>AI
        </div>
        <div style='font-size:11px;color:#6b6b8a;margin-top:4px;letter-spacing:1px;text-transform:uppercase;'>
            {tx["assistant_sub"]}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Lang toggle
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🇫🇷 Français", use_container_width=True):
            st.session_state.lang = "fr"
            st.rerun()
    with col2:
        if st.button("🇬🇧 English", use_container_width=True):
            st.session_state.lang = "en"
            st.rerun()

    # Videos list
    st.markdown(f"<div style='padding:12px 20px 6px;font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#6b6b8a;'>{tx['videos_label']}</div>", unsafe_allow_html=True)

    for video in VIDEOS_META:
        title     = video["title"]    if lang == "fr" else video["title_en"]
        tag_label = video["tag_label"] if lang == "fr" else video["tag_label_en"]
        st.markdown(f"""
        <div class='video-card'>
            <div class='video-card-title'>{title}</div>
            <span class='tag tag-{video["tag"]}'>{tag_label}</span>
        </div>
        """, unsafe_allow_html=True)

    # Stats
    if pipeline_ready:
        st.markdown(f"""
        <div class='stats-row'>
            <div class='stat-item'>
                <div class='stat-value'>7</div>
                <div class='stat-label'>{tx["stats_videos"]}</div>
            </div>
            <div class='stat-item'>
                <div class='stat-value'>{chunk_count}</div>
                <div class='stat-label'>{tx["stats_chunks"]}</div>
            </div>
            <div class='stat-item'>
                <div class='stat-value'>FR/EN</div>
                <div class='stat-label'>{tx["stats_lang"]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button(tx["clear"], use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── MAIN AREA ─────────────────────────────────────────────────────────────────
if not pipeline_ready:
    st.error(f"⚠️ {pipeline_error}")
    st.info(tx["error_msg"])
    st.stop()

# Welcome screen
if not st.session_state.messages:
    st.markdown(f"""
    <div class='welcome-container'>
        <div style='font-size:52px;margin-bottom:16px;'></div>
        <div class='welcome-title'>{tx["welcome_title"]}</div>
        <div class='welcome-sub'>{tx["welcome_sub"]}</div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    for col, (emoji, text) in zip(cols, tx["suggestions"]):
        with col:
            if st.button(f"{emoji} {text}", use_container_width=True, key=f"sug_{text}"):
                st.session_state.messages.append({"role": "user", "content": text})
                with st.spinner(tx["searching"]):
                    response, not_found, confidence, source = rag_search(text, [])
                st.session_state.messages.append({"role": "assistant", "content": response, "from_rag": not not_found, "confidence": confidence, "source": source})
                st.rerun()

# Display messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class='user-message'>
            <div class='user-bubble'>{msg['content']}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        from_rag   = msg.get("from_rag", True)
        confidence = msg.get("confidence", 0)
        source     = msg.get("source")
        badge = f"<span class='source-badge'>{tx['source_badge']}</span>" if from_rag else f"<span class='warning-badge'>{tx['warning_badge']}</span>"
        conf_html = f"<span class='confidence-badge'>🎯 {confidence}% match</span>" if confidence > 0 else ""
        src_html = ""
        if source and from_rag:
            title = source["title"] if lang == "fr" else source["title_en"]
            watch = "▶️ Voir sur YouTube" if lang == "fr" else "▶️ Watch on YouTube"
            src_html = f"<a class='source-link' href='{source['url']}' target='_blank'>{watch} — {title}</a>"
        st.markdown(f"""
        <div class='bot-message'>
            <div class='bot-avatar'>🤖</div>
            <div>
                <div class='bot-bubble'>{msg['content']}</div>
                <div style='margin-top:6px'>{badge}{conf_html}{src_html}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── RESPONSE LOGIC ────────────────────────────────────────────────────────────
TOPIC_KEYWORDS  = ["sujets", "topics", "disponibles", "vidéos", "videos", "available", "what can"]
CASUAL_KEYWORDS = ["bonjour", "hello", "salut", "hi", "merci", "thanks", "au revoir", "bye", "qui es-tu", "comment tu t'appelles"]

def get_response(question):
    q = question.lower()
    if any(kw in q for kw in TOPIC_KEYWORDS):
        return tx["topics_answer"], False, 0, None
    if any(kw in q for kw in CASUAL_KEYWORDS):
        return tx["greeting"], False, 0, None
    answer, not_found, confidence, source = rag_search(question, st.session_state.messages[-12:])
    return answer, not not_found, confidence, source

# ── CHAT INPUT ────────────────────────────────────────────────────────────────
if prompt := st.chat_input(tx["placeholder"]):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner(tx["searching"]):
        answer, from_rag, confidence, source = get_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": answer, "from_rag": from_rag, "confidence": confidence, "source": source})
    st.rerun()