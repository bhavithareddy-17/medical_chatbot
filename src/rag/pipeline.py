# ─────────────────────────────────────────────
#  src/rag/pipeline.py — Fixed RAG Pipeline
# ─────────────────────────────────────────────
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, GOOGLE_API_KEY
from src.utils.logger import get_logger
from src.utils.language import detect_language, build_language_instruction

logger = get_logger(__name__)


def _build_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not set in .env")
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=4096,   # must be high enough for full report
        convert_system_message_to_human=True,
    )
    logger.info(f"LLM: {LLM_MODEL} (temp={LLM_TEMPERATURE})")
    return llm


def _format_docs(docs: List[Document]) -> str:
    if not docs:
        return "No specific context retrieved — use your general medical knowledge."
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "MedQA")
        parts.append(f"[Source {i} — {source}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


def _build_full_prompt(question: str, context: str, history_text: str, lang_instruction: str) -> str:
    """
    Build the complete prompt as a single string.
    Using a plain string avoids any ChatPromptTemplate truncation issues.
    """
    return f"""You are MediAssist, a reliable AI medical assistant.

{lang_instruction}

MEDICAL KNOWLEDGE BASE CONTEXT:
════════════════════════════════════════
{context}
════════════════════════════════════════

{history_text}

Patient Query: {question}

YOUR TASK: Write a COMPLETE medical response. You MUST include ALL 6 sections below.
Do NOT stop after the first section. Do NOT summarize. Write every section in full.

MANDATORY OUTPUT — ALL 6 SECTIONS REQUIRED:

---
🔍 **Possible Condition(s):**
- **[Disease Name]:** Explanation of why it matches the symptoms. (write AT LEAST 3 diseases)
- **[Disease Name]:** Explanation.
- **[Disease Name]:** Explanation.

📋 **Likely Causes:**
- [Specific cause 1]
- [Specific cause 2]
- [Specific cause 3]
- [Specific cause 4]
- [Specific cause 5]

💊 **Medications & Treatments:**
- [Drug category 1 and what it does — e.g. bronchodilators to open airways]
- [Drug category 2 — e.g. antihistamines for allergic response]
- [Drug category 3 — e.g. corticosteroids to reduce inflammation]
- [OTC option — e.g. saline nasal spray, steam inhalation]
- [Lifestyle-based treatment]

🛡️ **Precautions & Self-Care:**
- [Action 1]
- [Action 2]
- [Action 3]
- [Action 4]
- [Action 5]

🏥 **Should You See a Doctor?**
**[YES - See a doctor urgently] OR [YES - See a doctor within 1-2 days] OR [MONITOR - Manage at home]**
Reason: [2-3 sentences. Include specific red flag symptoms that need emergency care.]

⚠️ **Disclaimer:**
This is educational information only, NOT a substitute for professional medical advice. Always consult a licensed healthcare professional for proper diagnosis and treatment.
---

Remember: Write ALL 6 sections completely. Do not stop early."""


class MedicalRAGPipeline:
    def __init__(self, retriever: BaseRetriever) -> None:
        self.retriever = retriever
        self.llm       = _build_llm()
        self.parser    = StrOutputParser()
        logger.info("MedicalRAGPipeline ready.")

    def ask(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        chat_history = chat_history or []

        # Language detection
        lang_code, lang_name = detect_language(question)
        lang_instruction     = build_language_instruction(lang_code, lang_name)
        logger.info(f"Language: {lang_name} ({lang_code})")

        # Build conversation history as plain text
        history_text = ""
        if chat_history:
            lines = []
            for msg in chat_history[-4:]:   # last 2 turns only to save tokens
                role = "Patient" if msg["role"] == "user" else "MediAssist"
                lines.append(f"{role}: {msg['content'][:300]}")
            if lines:
                history_text = "PREVIOUS CONVERSATION:\n" + "\n".join(lines) + "\n"

        # Hybrid retrieval
        retrieved_docs: List[Document] = []
        try:
            retrieved_docs = self.retriever.invoke(question)
            logger.info(f"Retrieved {len(retrieved_docs)} docs.")
        except Exception as e:
            logger.warning(f"Retrieval warning: {e}")

        context = _format_docs(retrieved_docs)

        # Build full prompt as plain string — no template truncation risk
        full_prompt = _build_full_prompt(question, context, history_text, lang_instruction)

        # Send as a single HumanMessage
        try:
            response = self.llm.invoke([HumanMessage(content=full_prompt)])
            answer   = self.parser.invoke(response)
            logger.info(f"Response length: {len(answer)} chars")
        except Exception as e:
            logger.error(f"LLM error: {e}")
            raise

        return {
            "answer"        : answer,
            "sources"       : list({d.metadata.get("source", "MedQA") for d in retrieved_docs}),
            "retrieved_docs": retrieved_docs,
            "has_context"   : bool(retrieved_docs),
            "language"      : lang_name,
        }
