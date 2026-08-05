
# medical_chatbot

# 🏥 MediAssist — AI Medical Chatbot

> A production-grade, **RAG-powered** medical chatbot built with LangChain, Gemini 1.5 Flash, ChromaDB, LangSmith, and Streamlit. Grounded in the **MedQA** dataset (5,000 records) with aggressive anti-hallucination measures.

---

## 📁 Project Structure

```
medical_chatbot/
├── app.py                  # Streamlit UI (entry point)
├── initialize.py           # One-time setup: download data + build vector store
├── evaluate.py             # LangSmith evaluation script
├── config.py               # Central configuration (paths, models, prompts)
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
│
├── src/
│   ├── rag/
│   │   ├── data_loader.py  # MedQA dataset loader + local cache
│   │   ├── vector_store.py # ChromaDB vector store builder + retriever
│   │   └── pipeline.py     # Full RAG chain (retriever → prompt → LLM → parser)
│   └── utils/
│       └── logger.py       # Centralized logging (file + console)
│
├── data/
│   ├── vectorstore/        # Persisted ChromaDB index (auto-created)
│   └── medqa_cache.json    # Cached dataset (auto-created)
│
└── logs/
    └── medical_chatbot.log # Application logs (auto-created)
```

---

## 🚀 Quick Start

### 1. Clone & Create Environment

```bash
git clone <your-repo>
cd medical_chatbot

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional (for LangSmith tracing)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=medical-chatbot-rag
```

- **Gemini API Key (Free)**: https://aistudio.google.com/app/apikey
- **LangSmith API Key (Free)**: https://smith.langchain.com/

### 3. Initialize (Download Data + Build Vector Store)

```bash
python initialize.py
```

This will:
1. Download 5,000 MedQA records from HuggingFace
2. Split into chunks (500 tokens, 50 overlap)
3. Embed with `all-MiniLM-L6-v2` (runs locally, free)
4. Persist ChromaDB vector store to `data/vectorstore/`

> ⏱️ Takes ~5–10 minutes on first run. Subsequent runs use the cache (< 30 seconds).

### 4. Run the App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 🧠 Architecture

```
User Input (Symptoms)
        │
        ▼
┌───────────────────┐
│  Streamlit UI     │  ← Session state, chat history, quick chips
└────────┬──────────┘
         │
         ▼
┌────────────────────────────────────┐
│  RAG Pipeline (LangChain LCEL)     │
│                                    │
│  1. ChromaDB Retriever             │  ← Top-5 similar MedQA docs
│     (similarity + score threshold) │
│                                    │
│  2. Context Formatter              │  ← Numbered source blocks
│                                    │
│  3. Anti-Hallucination Prompt      │  ← System + context + question
│                                    │
│  4. Gemini 1.5 Flash (temp=0.1)    │  ← Low temperature = conservative
│                                    │
│  5. StrOutputParser                │  ← Clean string output
└────────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│  LangSmith       │  ← Full trace: retrieval + LLM call + latency
└──────────────────┘
         │
         ▼
  Structured Response:
  🔍 Possible Conditions
  📋 Likely Causes
  🛡️ Precautions & Self-Care
  🏥 Should You See a Doctor?
  ⚠️ Disclaimer
```

---

## 🛡️ Anti-Hallucination Strategies

| Strategy | Implementation |
|---|---|
| **Low temperature** | `temperature=0.1` on Gemini — deterministic, factual responses |
| **Grounded prompting** | System prompt explicitly forbids claims not in retrieved context |
| **Score threshold** | Only retrieves docs with similarity ≥ 0.3 |
| **Fallback message** | If no relevant context found, returns "consult a doctor" instead of guessing |
| **Source attribution** | Every response shows which MedQA records were used |
| **Context injection** | Full retrieved documents included in every LLM call |

---

## ⚙️ Configuration (config.py)

| Parameter | Default | Description |
|---|---|---|
| `MAX_RECORDS` | 5000 | MedQA records to load |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local HuggingFace embeddings |
| `CHUNK_SIZE` | 500 | Token chunk size |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `RETRIEVER_K` | 5 | Top-K docs to retrieve |
| `RETRIEVER_SCORE_THRESHOLD` | 0.3 | Min similarity score |
| `LLM_MODEL` | `gemini-1.5-flash` | Gemini model (free tier) |
| `LLM_TEMPERATURE` | 0.1 | Response randomness |

---

## 📊 Evaluation (LangSmith)

```bash
python evaluate.py
```

Runs 5 test queries and reports:
- Retrieval success rate
- Average context docs per query
- Average response length
- All traces visible in LangSmith dashboard

---

## 🔧 Useful Commands

```bash
# Force rebuild vector store (if data changes)
python initialize.py --force

# Run evaluation
python evaluate.py

# View logs
tail -f logs/medical_chatbot.log
```

---

## ⚠️ Disclaimer

MediAssist provides **educational information only** and is **not a substitute** for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for medical decisions.

---

## 📦 Tech Stack

| Component | Technology |
|---|---|
| LLM | Google Gemini 1.5 Flash (Free Tier) |
| RAG Framework | LangChain 0.3 (LCEL) |
| Tracing | LangSmith |
| Vector Store | ChromaDB (persistent) |
| Embeddings | `all-MiniLM-L6-v2` (local, free) |
| Dataset | MedQA via HuggingFace `bigbio/med_qa` |
| UI | Streamlit |
| Python | 3.10+ |

