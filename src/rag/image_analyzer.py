# ─────────────────────────────────────────────
#  src/rag/image_analyzer.py
#  Multimodal RAG — Gemini Vision for disease image analysis
# ─────────────────────────────────────────────
"""
Accepts an uploaded image (skin rash, wound, eye condition, etc.)
and uses Gemini Vision to:
  1. Identify visible symptoms / findings
  2. Extract a text description for RAG retrieval
  3. Generate structured medical analysis
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Vision Prompt ──────────────────────────────────────────────────────────────
VISION_EXTRACTION_PROMPT = """You are a medical image analysis assistant.

Carefully examine this medical image and extract ALL visible symptoms and findings.

Respond ONLY with a JSON object in this exact format (no markdown, no backticks):
{
  "visible_symptoms": ["symptom1", "symptom2", "symptom3"],
  "affected_area": "describe the body part / area visible",
  "visual_characteristics": "describe color, texture, shape, size, pattern of the condition",
  "severity_estimate": "mild / moderate / severe",
  "search_query": "concise medical search query to find relevant conditions (e.g. red scaly skin rash with silvery patches)"
}"""

IMAGE_ANALYSIS_PROMPT = """You are MediAssist, a reliable AI medical image analysis assistant.

A patient has uploaded a medical image. Based on the image and any additional context provided,
give a thorough medical analysis.

IMPORTANT RULES:
1. Clearly state what you observe visually — be specific about appearance.
2. List all plausible conditions — do NOT narrow to just one prematurely.
3. Complete ALL sections below without exception.
4. Recommend professional evaluation — visual diagnosis has limits.
5. {language_instruction}

REQUIRED OUTPUT FORMAT:

---
📸 **Visual Findings:**
Describe exactly what is visible in the image (color, texture, location, size, pattern).

🔍 **Possible Condition(s):**
- **[Condition Name]:** Why this matches the visual findings. (list at least 3)
- **[Condition Name]:** Why this matches.
- **[Condition Name]:** Why this matches.

📋 **Likely Causes:**
- [Cause 1]
- [Cause 2]
- [Cause 3]
- [Cause 4]
(minimum 4 causes)

💊 **Medications & Treatments:**
- [Treatment/medication type 1 — general category, not specific prescription]
- [Treatment/medication type 2]
- [Treatment/medication type 3]
- [Home remedy / OTC option]
(minimum 4 options — use general categories, never specific dosages)

🛡️ **Precautions & Self-Care:**
- [Step 1]
- [Step 2]
- [Step 3]
- [Step 4]
- [Step 5]
(minimum 5 actionable steps)

🏥 **Should You See a Doctor?**
**[YES - See a doctor urgently] OR [YES - See a doctor within 1-2 days] OR [MONITOR - Manage at home]**
Reason: Explain in 2-3 sentences. List red flag symptoms requiring emergency care.

⚠️ **Disclaimer:**
Visual AI analysis is NOT a medical diagnosis. This image analysis is for educational purposes only.
Always consult a licensed dermatologist or healthcare professional for proper diagnosis and treatment.
---"""


def _encode_image(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")


def _get_mime_type(filename: str) -> str:
    """Infer MIME type from filename extension."""
    ext = Path(filename).suffix.lower()
    return {
        ".jpg" : "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png" : "image/png",
        ".gif" : "image/gif",
        ".webp": "image/webp",
        ".bmp" : "image/bmp",
    }.get(ext, "image/jpeg")


class MedicalImageAnalyzer:
    """
    Analyzes medical images using Gemini Vision.

    Usage
    -----
    analyzer = MedicalImageAnalyzer()
    result = analyzer.analyze(image_bytes, filename="rash.jpg", language="English")
    """

    def __init__(self) -> None:
        self.llm = self._build_vision_llm()
        logger.info("MedicalImageAnalyzer ready.")

    def _build_vision_llm(self):
        """Build Gemini multimodal LLM."""
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=LLM_TEMPERATURE,
            max_output_tokens=2048,
            convert_system_message_to_human=True,
        )

    def extract_symptoms_from_image(
        self,
        image_bytes: bytes,
        filename: str = "image.jpg",
    ) -> Dict[str, Any]:
        """
        Step 1: Extract structured symptom data from image using Gemini Vision.
        Returns dict with visible_symptoms, search_query, etc.
        """
        import json
        from langchain_core.messages import HumanMessage

        mime_type  = _get_mime_type(filename)
        b64_image  = _encode_image(image_bytes)

        message = HumanMessage(content=[
            {"type": "text",       "text": VISION_EXTRACTION_PROMPT},
            {"type": "image_url",  "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}},
        ])

        try:
            response = self.llm.invoke([message])
            raw = response.content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw.strip())
            logger.info(f"Image extraction: {data.get('search_query', 'N/A')}")
            return data
        except Exception as e:
            logger.warning(f"Image extraction failed: {e}")
            return {
                "visible_symptoms"      : ["skin condition"],
                "affected_area"         : "visible in image",
                "visual_characteristics": "visible skin/body condition",
                "severity_estimate"     : "unknown",
                "search_query"          : "skin condition symptoms treatment",
            }

    def analyze(
        self,
        image_bytes: bytes,
        filename: str = "image.jpg",
        additional_text: str = "",
        language: str = "English",
        rag_context: str = "",
    ) -> Dict[str, Any]:
        """
        Full multimodal analysis: Vision → RAG context → Structured response.

        Parameters
        ----------
        image_bytes     : Raw image bytes
        filename        : Original filename (for MIME type detection)
        additional_text : Any text the user typed alongside the image
        language        : Output language
        rag_context     : Retrieved medical context to ground the response

        Returns
        -------
        dict with keys: answer, extracted_symptoms, search_query
        """
        from langchain_core.messages import HumanMessage

        mime_type = _get_mime_type(filename)
        b64_image = _encode_image(image_bytes)

        # Language instruction
        lang_instr = (
            f"Respond entirely in {language}."
            if language.lower() != "english"
            else "Respond in English."
        )

        # Build analysis prompt
        analysis_prompt = IMAGE_ANALYSIS_PROMPT.format(
            language_instruction=lang_instr
        )

        context_block = ""
        if rag_context:
            context_block = (
                f"\n\nRELEVANT MEDICAL CONTEXT FROM KNOWLEDGE BASE:\n"
                f"{'═'*40}\n{rag_context}\n{'═'*40}\n"
                "Use this context to support your analysis.\n"
            )

        user_text = ""
        if additional_text.strip():
            user_text = f"\nPatient also says: {additional_text.strip()}\n"

        full_prompt = (
            f"{analysis_prompt}\n"
            f"{context_block}"
            f"{user_text}"
            "\nAnalyze the image and provide the complete structured medical report."
        )

        message = HumanMessage(content=[
            {"type": "text",      "text": full_prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}},
        ])

        try:
            logger.info("Running Gemini Vision analysis…")
            response = self.llm.invoke([message])
            answer   = response.content.strip()
            logger.info("Image analysis complete.")
        except Exception as e:
            logger.error(f"Vision analysis error: {e}")
            raise

        # Also extract symptoms for RAG retrieval
        extracted = self.extract_symptoms_from_image(image_bytes, filename)

        return {
            "answer"            : answer,
            "extracted_symptoms": extracted.get("visible_symptoms", []),
            "search_query"      : extracted.get("search_query", ""),
            "severity"          : extracted.get("severity_estimate", "unknown"),
        }