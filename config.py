# ─────────────────────────────────────────────
#  config.py — Central Configuration
# ─────────────────────────────────────────────
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent
DATA_DIR        = BASE_DIR / "data"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
LOG_DIR         = BASE_DIR / "logs"

DATA_DIR.mkdir(exist_ok=True)
VECTORSTORE_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ── API Keys ───────────────────────────────────────────────────────────────────
GOOGLE_API_KEY   = os.getenv("GOOGLE_API_KEY", "")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")

# ── LangSmith Tracing ─────────────────────────────────────────────────────────
LANGSMITH_TRACING  = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGSMITH_PROJECT  = os.getenv("LANGCHAIN_PROJECT", "medical-chatbot-rag")
LANGSMITH_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

if LANGSMITH_TRACING:
    os.environ["LANGCHAIN_TRACING_V2"]  = "true"
    os.environ["LANGCHAIN_API_KEY"]     = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"]     = LANGSMITH_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"]    = LANGSMITH_ENDPOINT

# ── Dataset ────────────────────────────────────────────────────────────────────
DATASET_NAME        = "bigbio/med_qa"          # HuggingFace dataset
DATASET_SPLIT       = "train"
MAX_RECORDS         = 5000
DATASET_CACHE_FILE  = DATA_DIR / "medqa_cache.json"

# ── Embeddings ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL     = "all-MiniLM-L6-v2"       # Lightweight, free, fast
EMBEDDING_DIMENSION = 384

# ── Vector Store ───────────────────────────────────────────────────────────────
VECTORSTORE_COLLECTION = "medqa_collection"
VECTORSTORE_PATH       = str(VECTORSTORE_DIR)

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50

# ── Retrieval ──────────────────────────────────────────────────────────────────
RETRIEVER_K              = 5    # Top-K documents to retrieve
RETRIEVER_SCORE_THRESHOLD = 0.3  # Minimum similarity score

# ── LLM (Gemini Free Tier) ─────────────────────────────────────────────────────
LLM_MODEL       = "gemini-2.5-flash"   # Free tier model
LLM_TEMPERATURE = 0.1                  # Low temp → less hallucination
LLM_MAX_TOKENS  = 2048                 # Increased so full response is never cut off

# ── Prompts ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are MediAssist, a reliable AI medical assistant.

Use the provided context as your PRIMARY source. You may supplement with general medical knowledge when needed.

MANDATORY RULES — follow these strictly every single time:
1. You MUST always write ALL 5 sections. Never skip any section. Never stop early.
2. Each section must have the minimum number of lines specified below.
3. Use bullet points (- ) for all lists.
4. Be specific — name actual diseases, not vague descriptions.
5. Never fabricate drug dosages or specific statistics.

REQUIRED OUTPUT FORMAT (follow exactly, every time):

---
🔍 **Possible Condition(s):**
- **[Disease Name]:** Brief explanation of why this matches the symptoms. (at least 3 diseases)
- **[Disease Name]:** Brief explanation.
- **[Disease Name]:** Brief explanation.

📋 **Likely Causes:**
- [Cause 1 — be specific]
- [Cause 2 — be specific]
- [Cause 3 — be specific]
- [Cause 4 — be specific]
(minimum 4 causes)

🛡️ **Precautions & Self-Care:**
- [Actionable step 1]
- [Actionable step 2]
- [Actionable step 3]
- [Actionable step 4]
- [Actionable step 5]
(minimum 5 steps — be practical and specific)

🏥 **Should You See a Doctor?**
**[YES - See a doctor urgently] OR [YES - See a doctor within 1-2 days] OR [MONITOR - Manage at home]**
Reason: [2-3 sentences explaining why, including specific red flag symptoms that need emergency care]

⚠️ **Disclaimer:**
This information is for educational purposes only and is NOT a substitute for professional medical advice. Always consult a licensed doctor for proper diagnosis and treatment.
---"""