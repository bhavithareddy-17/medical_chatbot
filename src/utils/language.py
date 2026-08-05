# ─────────────────────────────────────────────
#  src/utils/language.py
#  Auto language detection + multilingual support
# ─────────────────────────────────────────────
"""
Detects input language and instructs the LLM to respond in the same language.
Supports: English, Hindi, Telugu, Tamil, Kannada, Bengali, Marathi,
          Spanish, French, German, Arabic, Chinese, Japanese, and more.
Uses lightweight langdetect library (no API calls needed).
"""
from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Language code → full name mapping ─────────────────────────────────────────
LANGUAGE_MAP = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
    "ml": "Malayalam",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ur": "Urdu",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ar": "Arabic",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "it": "Italian",
    "tr": "Turkish",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
}

# Scripts that are visually identifiable (don't need langdetect)
SCRIPT_HINTS = {
    # Devanagari → Hindi/Marathi
    "\u0900": "hi", "\u0901": "hi", "\u0902": "hi",
    # Telugu script
    "\u0C00": "te", "\u0C15": "te", "\u0C05": "te",
    # Tamil script
    "\u0B85": "ta", "\u0B95": "ta",
    # Kannada
    "\u0C85": "kn", "\u0C95": "kn",
    # Arabic
    "\u0621": "ar", "\u0622": "ar",
    # Chinese CJK
    "\u4E00": "zh-cn",
    # Japanese Hiragana
    "\u3041": "ja", "\u3042": "ja",
    # Korean Hangul
    "\uAC00": "ko",
}


def detect_language(text: str) -> tuple[str, str]:
    """
    Detect the language of the input text.

    Returns
    -------
    (language_code, language_name)  e.g. ("hi", "Hindi")
    """
    if not text or len(text.strip()) < 3:
        return "en", "English"

    # Fast script-based detection for Indian/Asian scripts
    for char in text:
        for script_char, lang_code in SCRIPT_HINTS.items():
            if char >= script_char and ord(char) - ord(script_char) < 100:
                lang_name = LANGUAGE_MAP.get(lang_code, "Unknown")
                logger.info(f"Script detection: {lang_code} ({lang_name})")
                return lang_code, lang_name

    # Fallback: langdetect library
    try:
        from langdetect import detect, LangDetectException
        code = detect(text)
        # Normalize zh variants
        if code.startswith("zh"):
            code = "zh-cn"
        name = LANGUAGE_MAP.get(code, "English")
        logger.info(f"langdetect: {code} ({name})")
        return code, name
    except Exception as e:
        logger.warning(f"Language detection failed: {e} — defaulting to English")
        return "en", "English"


def build_language_instruction(lang_code: str, lang_name: str) -> str:
    """
    Build the language instruction to append to the LLM prompt.
    """
    if lang_code == "en":
        return "Respond in English."

    return (
        f"IMPORTANT: The user has written in {lang_name}. "
        f"You MUST respond entirely in {lang_name}. "
        f"All sections, headings, bullet points, and the disclaimer must be in {lang_name}. "
        f"Do not mix languages."
    )


def get_ui_labels(lang_code: str) -> dict:
    """
    Return translated UI labels for common elements.
    Currently returns English for unsupported languages.
    """
    labels = {
        "en": {
            "input_placeholder": "e.g. I have a high fever, severe headache, and body aches…",
            "send_btn"         : "Send 🔍",
            "upload_label"     : "📸 Upload a medical image (optional)",
            "analyzing"        : "🔬 Analyzing symptoms…",
            "image_analyzing"  : "🔬 Analyzing image…",
            "you"              : "👤 You",
            "bot"              : "🤖 MediAssist",
        },
        "hi": {
            "input_placeholder": "उदा. मुझे तेज बुखार, सिरदर्द और शरीर में दर्द है…",
            "send_btn"         : "भेजें 🔍",
            "upload_label"     : "📸 मेडिकल छवि अपलोड करें (वैकल्पिक)",
            "analyzing"        : "🔬 लक्षणों का विश्लेषण हो रहा है…",
            "image_analyzing"  : "🔬 छवि का विश्लेषण हो रहा है…",
            "you"              : "👤 आप",
            "bot"              : "🤖 MediAssist",
        },
        "te": {
            "input_placeholder": "ఉదా. నాకు జ్వరం, తలనొప్పి మరియు శరీర నొప్పులు ఉన్నాయి…",
            "send_btn"         : "పంపు 🔍",
            "upload_label"     : "📸 వైద్య చిత్రాన్ని అప్‌లోడ్ చేయండి (ఐచ్ఛికం)",
            "analyzing"        : "🔬 లక్షణాలను విశ్లేషిస్తున్నారు…",
            "image_analyzing"  : "🔬 చిత్రాన్ని విశ్లేషిస్తున్నారు…",
            "you"              : "👤 మీరు",
            "bot"              : "🤖 MediAssist",
        },
        "ta": {
            "input_placeholder": "எ.கா. எனக்கு காய்ச்சல், தலைவலி மற்றும் உடல் வலி உள்ளது…",
            "send_btn"         : "அனுப்பு 🔍",
            "upload_label"     : "📸 மருத்துவ படத்தை பதிவேற்றவும் (விரும்பினால்)",
            "analyzing"        : "🔬 அறிகுறிகளை பகுப்பாய்வு செய்கிறது…",
            "image_analyzing"  : "🔬 படத்தை பகுப்பாய்வு செய்கிறது…",
            "you"              : "👤 நீங்கள்",
            "bot"              : "🤖 MediAssist",
        },
    }
    return labels.get(lang_code, labels["en"])