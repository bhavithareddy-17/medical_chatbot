#!/usr/bin/env python3
# ─────────────────────────────────────────────
#  app.py — MediAssist Streamlit UI (v2)
#  • Hybrid retrieval (BM25 + MMR)
#  • Multimodal image analysis (Gemini Vision)
#  • Multilingual auto-detection
#  • Guaranteed full structured output
# ─────────────────────────────────────────────
import sys
import time
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import GOOGLE_API_KEY, VECTORSTORE_PATH
from src.utils.logger import get_logger
from src.utils.language import detect_language, get_ui_labels

logger = get_logger("app")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MediAssist — AI Medical Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%); }

.med-header { text-align:center; padding:1.5rem 0 0.5rem 0; }
.med-header h1 {
    font-family:'Playfair Display',serif; font-size:2.6rem;
    background:linear-gradient(90deg,#38bdf8,#818cf8,#34d399);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.med-header p { color:#94a3b8; font-size:0.95rem; }

.user-bubble {
    background:linear-gradient(135deg,#1d4ed8,#2563eb);
    color:#fff; padding:0.9rem 1.2rem;
    border-radius:18px 18px 4px 18px;
    margin:0.4rem 0 0.4rem 20%;
    box-shadow:0 4px 15px rgba(37,99,235,0.3);
    font-size:0.93rem; line-height:1.6;
}
.lang-badge {
    display:inline-block; background:#1e3a5f; color:#7dd3fc;
    border:1px solid #2563eb; border-radius:20px;
    padding:1px 8px; font-size:0.7rem; margin-left:6px;
}
.source-badge {
    display:inline-block; background:#1e3a5f; color:#7dd3fc;
    border:1px solid #2563eb44; border-radius:20px;
    padding:2px 10px; font-size:0.72rem; margin:2px;
}
.img-badge {
    display:inline-block; background:#1a3a2a; color:#6ee7b7;
    border:1px solid #10b981; border-radius:20px;
    padding:2px 10px; font-size:0.72rem; margin:2px;
}
.severity-mild   { color:#34d399; font-weight:600; }
.severity-moderate { color:#fbbf24; font-weight:600; }
.severity-severe  { color:#f87171; font-weight:600; }

.status-ok   { background:#064e3b; border:1px solid #10b981; border-radius:10px; padding:0.5rem 1rem; color:#6ee7b7!important; font-size:0.83rem; margin:3px 0; }
.status-warn { background:#78350f; border:1px solid #f59e0b; border-radius:10px; padding:0.5rem 1rem; color:#fcd34d!important; font-size:0.83rem; margin:3px 0; }
.status-err  { background:#7f1d1d; border:1px solid #ef4444; border-radius:10px; padding:0.5rem 1rem; color:#fca5a5!important; font-size:0.83rem; margin:3px 0; }

.stButton>button {
    background:linear-gradient(135deg,#0ea5e9,#6366f1);
    color:white; border:none; border-radius:10px;
    font-weight:600; transition:all 0.2s;
}
.stButton>button:hover { transform:translateY(-1px); box-shadow:0 6px 20px rgba(14,165,233,0.4); }

section[data-testid="stSidebar"] { background:#0f172a!important; border-right:1px solid #1e293b; }
section[data-testid="stSidebar"] * { color:#94a3b8!important; }

.stTextInput>div>div>input {
    background:#1e293b!important; border:1px solid #334155!important;
    border-radius:12px!important; color:#f1f5f9!important; font-size:0.93rem!important;
}
.stTextInput>div>div>input:focus { border-color:#38bdf8!important; box-shadow:0 0 0 3px rgba(56,189,248,0.15)!important; }

.stFileUploader { background:#1e293b; border:1px dashed #334155; border-radius:12px; padding:0.5rem; }
hr { border-color:#1e293b!important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
defaults = {
    "chat_history" : [],
    "pipeline"     : None,
    "analyzer"     : None,
    "store_obj"    : None,
    "all_docs"     : None,
    "store_ready"  : False,
    "init_error"   : None,
    "show_context" : False,
    "pending_input": None,
    "input_key"    : 0,
    "current_lang" : "en",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE LOADER (cached)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_pipeline_and_analyzer():
    from src.rag.data_loader      import load_medqa_documents
    from src.rag.vector_store     import MedicalVectorStore
    from src.rag.hybrid_retriever import build_hybrid_retriever, HybridRetrieverWrapper
    from src.rag.pipeline         import MedicalRAGPipeline
    from src.rag.image_analyzer   import MedicalImageAnalyzer

    # Load docs & build vector store
    docs  = load_medqa_documents()
    store = MedicalVectorStore().build(docs)

    # Build hybrid retriever
    ensemble = build_hybrid_retriever(store, docs)
    retriever = HybridRetrieverWrapper(ensemble=ensemble)

    pipeline = MedicalRAGPipeline(retriever)
    analyzer = MedicalImageAnalyzer()

    return pipeline, analyzer, store, docs

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏥 MediAssist v2")
    st.markdown("---")
    st.markdown("### ⚙️ System Status")

    if GOOGLE_API_KEY and GOOGLE_API_KEY != "your_gemini_api_key_here":
        st.markdown('<div class="status-ok">✅ Gemini API Key Found</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-err">❌ Gemini API Key Missing</div>', unsafe_allow_html=True)

    vs_path = Path(VECTORSTORE_PATH)
    if vs_path.exists() and any(vs_path.iterdir()):
        st.markdown('<div class="status-ok">✅ Vector Store Ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-warn">⚠️ Run initialize.py first</div>', unsafe_allow_html=True)

    if st.session_state.pipeline:
        st.markdown('<div class="status-ok">✅ Hybrid RAG Pipeline Active</div>', unsafe_allow_html=True)
        st.markdown('<div class="status-ok">✅ Vision Analyzer Active</div>', unsafe_allow_html=True)
        st.markdown('<div class="status-ok">✅ Multilingual Support ON</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-warn">⏳ Pipeline Not Loaded</div>', unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🚀 Load / Reload Pipeline", use_container_width=True):
        with st.spinner("Loading hybrid RAG pipeline…"):
            try:
                pipeline, analyzer, store, docs = load_pipeline_and_analyzer()
                st.session_state.pipeline    = pipeline
                st.session_state.analyzer    = analyzer
                st.session_state.store_obj   = store
                st.session_state.all_docs    = docs
                st.session_state.store_ready = True
                st.session_state.init_error  = None
                st.success("Pipeline loaded!")
            except Exception as e:
                st.session_state.init_error = str(e)
                st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("### 🔧 Options")
    st.session_state.show_context = st.toggle("Show retrieved context", value=st.session_state.show_context)

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("""
### ℹ️ What's New in v2
- 🔀 **Hybrid Retrieval** — BM25 + MMR semantic search
- 📸 **Image Analysis** — Upload skin rash, wounds, etc.
- 🌐 **Multilingual** — Hindi, Telugu, Tamil, English & more
- 💊 **Full Output** — Conditions, Causes, Medications, Precautions, Doctor advice
- 🤖 Gemini 2.5 Flash Vision

⚠️ *Not a substitute for professional medical advice.*
""")

# ══════════════════════════════════════════════════════════════════════════════
#  AUTO-LOAD PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.pipeline and not st.session_state.init_error:
    vs_path = Path(VECTORSTORE_PATH)
    if vs_path.exists() and any(vs_path.iterdir()):
        with st.spinner("🔄 Loading hybrid RAG pipeline…"):
            try:
                pipeline, analyzer, store, docs = load_pipeline_and_analyzer()
                st.session_state.pipeline    = pipeline
                st.session_state.analyzer    = analyzer
                st.session_state.store_obj   = store
                st.session_state.all_docs    = docs
                st.session_state.store_ready = True
            except Exception as e:
                st.session_state.init_error = str(e)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="med-header">
    <h1>🏥 MediAssist</h1>
    <p>AI Medical Assistant · Hybrid RAG · Multimodal · Multilingual</p>
</div>
""", unsafe_allow_html=True)

if st.session_state.init_error:
    st.error(f"⚠️ {st.session_state.init_error}")
    st.info("Run `python initialize.py` then reload.")

# ── Quick symptom chips ────────────────────────────────────────────────────────
st.markdown("#### 💡 Quick Examples")
examples = [
    "Fever, headache & body ache", "Chest pain & shortness of breath",
    "Persistent cough & fatigue",  "Nausea, vomiting & abdominal pain",
    "Skin rash & itching",         "Joint pain & swelling",
    "Dizziness & blurred vision",  "Frequent urination & excessive thirst",
]
cols = st.columns(4)
for i, ex in enumerate(examples):
    if cols[i % 4].button(ex, key=f"chip_{i}", use_container_width=True):
        st.session_state["prefill"] = ex

st.markdown("---")

# ── Chat history ───────────────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        lang_badge = ""
        if msg.get("language") and msg["language"] != "English":
            lang_badge = f'<span class="lang-badge">🌐 {msg["language"]}</span>'
        img_badge = '<span class="img-badge">📸 Image</span>' if msg.get("has_image") else ""
        st.markdown(
            f'<div class="user-bubble">👤 {msg["content"]}{img_badge}{lang_badge}</div>',
            unsafe_allow_html=True,
        )
        # Show uploaded image thumbnail
        if msg.get("image_bytes"):
            st.image(msg["image_bytes"], width=200, caption="Uploaded image")

    else:
        st.markdown("---")
        # Language indicator
        if msg.get("language") and msg["language"] != "English":
            st.markdown(f'<span class="lang-badge">🌐 Response in {msg["language"]}</span>', unsafe_allow_html=True)

        st.markdown(f"🤖 **MediAssist:**")
        st.markdown(msg["content"])

        # Sources & severity
        col1, col2 = st.columns([3, 1])
        with col1:
            if msg.get("sources"):
                badges = "".join(f'<span class="source-badge">📖 {s}</span>' for s in msg["sources"])
                st.markdown(f"<small>🗂️ Sources: {badges}</small>", unsafe_allow_html=True)
        with col2:
            if msg.get("severity") and msg["severity"] != "unknown":
                sev = msg["severity"]
                css = f"severity-{sev}"
                st.markdown(f'<small>Severity: <span class="{css}">{sev.upper()}</span></small>', unsafe_allow_html=True)

        if st.session_state.show_context and msg.get("retrieved_docs"):
            with st.expander("📄 Retrieved Context (Hybrid BM25 + MMR)"):
                for i, doc_text in enumerate(msg["retrieved_docs"], 1):
                    st.markdown(f"**Document {i}:**")
                    st.text(doc_text[:500] + ("…" if len(doc_text) > 500 else ""))
                    st.markdown("---")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
#  INPUT AREA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 🩺 Describe Your Symptoms")
st.markdown(
    "*Type in any language — Hindi (हिंदी), Telugu (తెలుగు), Tamil (தமிழ்), English, etc.*",
)

prefill = st.session_state.pop("prefill", "")

# Image upload
uploaded_image = st.file_uploader(
    "📸 Upload a medical image (optional — skin rash, wound, eye, etc.)",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    key=f"img_upload_{st.session_state.input_key}",
)

col_input, col_btn = st.columns([5, 1])
with col_input:
    user_input = st.text_input(
        label="symptoms",
        label_visibility="collapsed",
        placeholder="e.g. I have a high fever and headache... / मुझे बुखार है... / నాకు జ్వరం ఉంది...",
        value=prefill,
        key=f"user_input_{st.session_state.input_key}",
    )
with col_btn:
    send_clicked = st.button("Send 🔍", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SUBMISSION HANDLER
# ══════════════════════════════════════════════════════════════════════════════
has_text  = bool(user_input.strip())
has_image = uploaded_image is not None
submitted = send_clicked and (has_text or has_image)

if submitted:
    if not st.session_state.pipeline:
        st.warning("⏳ Pipeline not loaded. Click **Load / Reload Pipeline** in the sidebar.")
    else:
        question = user_input.strip() if has_text else "Please analyze this medical image."

        # Detect language
        lang_code, lang_name = detect_language(question)

        # Read image bytes if uploaded
        image_bytes = uploaded_image.read() if has_image else None
        image_name  = uploaded_image.name   if has_image else None

        # Save user message
        st.session_state.chat_history.append({
            "role"       : "user",
            "content"    : question,
            "has_image"  : has_image,
            "image_bytes": image_bytes,
            "language"   : lang_name,
        })

        st.session_state.pending_input = {
            "question"   : question,
            "image_bytes": image_bytes,
            "image_name" : image_name,
            "has_image"  : has_image,
            "lang_code"  : lang_code,
            "lang_name"  : lang_name,
        }
        st.session_state.input_key += 1
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  PROCESS PENDING (runs on rerun after user msg saved)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.pending_input and st.session_state.pipeline:
    pending   = st.session_state.pending_input
    st.session_state.pending_input = None

    question    = pending["question"]
    image_bytes = pending["image_bytes"]
    image_name  = pending["image_name"]
    has_image   = pending["has_image"]
    lang_name   = pending["lang_name"]

    if has_image and st.session_state.analyzer:
        # ── MULTIMODAL PATH ────────────────────────────────────────────────────
        with st.spinner("🔬 Analyzing medical image with Gemini Vision…"):
            try:
                t0 = time.time()

                # Step 1: Extract symptoms from image for RAG retrieval
                analyzer   = st.session_state.analyzer
                extraction = analyzer.extract_symptoms_from_image(image_bytes, image_name or "image.jpg")
                search_q   = extraction.get("search_query", question)

                # Step 2: Retrieve relevant context using extracted symptoms
                rag_context = ""
                try:
                    docs = st.session_state.pipeline.retriever.invoke(search_q)
                    if docs:
                        from src.rag.pipeline import _format_docs
                        rag_context = _format_docs(docs)
                except Exception:
                    docs = []

                # Step 3: Full vision analysis with RAG context
                result = analyzer.analyze(
                    image_bytes    = image_bytes,
                    filename       = image_name or "image.jpg",
                    additional_text= question if question != "Please analyze this medical image." else "",
                    language       = lang_name,
                    rag_context    = rag_context,
                )

                elapsed = time.time() - t0
                logger.info(f"Image analysis complete in {elapsed:.2f}s")

                st.session_state.chat_history.append({
                    "role"          : "assistant",
                    "content"       : result["answer"],
                    "sources"       : ["MedQA (RAG)", "Gemini Vision"],
                    "retrieved_docs": [d.page_content for d in docs] if docs else [],
                    "has_context"   : True,
                    "language"      : lang_name,
                    "severity"      : result.get("severity", "unknown"),
                    "is_image"      : True,
                })

            except Exception as e:
                logger.error(f"Image analysis error: {e}")
                st.session_state.chat_history.append({
                    "role"   : "assistant",
                    "content": f"⚠️ Image analysis error: {e}. Please try again.",
                    "sources": [],
                })

    else:
        # ── TEXT PATH (Hybrid RAG) ─────────────────────────────────────────────
        with st.spinner("🔬 Analyzing symptoms with Hybrid RAG…"):
            try:
                t0     = time.time()
                result = st.session_state.pipeline.ask(
                    question     = question,
                    chat_history = st.session_state.chat_history[:-1],
                )
                elapsed = time.time() - t0
                logger.info(f"Text query answered in {elapsed:.2f}s | Lang: {result.get('language')}")

                st.session_state.chat_history.append({
                    "role"          : "assistant",
                    "content"       : result["answer"],
                    "sources"       : result.get("sources", []),
                    "retrieved_docs": [d.page_content for d in result.get("retrieved_docs", [])],
                    "has_context"   : result.get("has_context", False),
                    "language"      : result.get("language", "English"),
                    "severity"      : None,
                    "is_image"      : False,
                })

            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                st.session_state.chat_history.append({
                    "role"   : "assistant",
                    "content": f"⚠️ Error: {e}. Please check your API key and try again.",
                    "sources": [],
                })

    st.rerun()

# ── Empty state ────────────────────────────────────────────────────────────────
if not st.session_state.chat_history:
    st.markdown("""
<div style="text-align:center; padding:2.5rem 1rem; color:#475569;">
    <div style="font-size:3rem; margin-bottom:1rem;">🩺</div>
    <div style="font-size:1rem; color:#64748b; line-height:1.8;">
        Describe your symptoms <b>or upload a medical image</b> (skin rash, wound, eye condition, etc.)<br>
        Type in <b>any language</b> — Hindi, Telugu, Tamil, English and more.<br><br>
        Get: Possible Conditions · Causes · Medications · Precautions · Doctor Advice
    </div>
    <div style="margin-top:1.5rem; font-size:0.82rem; color:#475569;">
        ⚠️ Educational purposes only — not a substitute for professional medical advice.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#334155; font-size:0.78rem; padding:0.8rem 0;">
    MediAssist v2 · Gemini 2.5 Flash · Hybrid RAG (BM25 + MMR) · MedQA 5K · Multilingual<br>
    ⚠️ Educational information only. Always consult a qualified healthcare professional.
</div>
""", unsafe_allow_html=True)