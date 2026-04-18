"""
Translation service for converting English text to Indian languages.

Design:
- English is always the base language.
- If translation library (googletrans) is available, use it.
- Otherwise, gracefully fall back to returning the original English text.
"""

from typing import Dict

try:
    # googletrans is commonly used; if installed we will use it.
    from googletrans import Translator  # type: ignore
    _HAS_GOOGLETRANS = True
except Exception:
    Translator = None  # type: ignore
    _HAS_GOOGLETRANS = False


SUPPORTED_LANGUAGES: Dict[str, str] = {
    "en": "English",
    "mr": "Marathi",
    "hi": "Hindi",
    "gu": "Gujarati",
    "ta": "Tamil",
    "te": "Telugu",
}


class TranslationService:
    def __init__(self):
        self.enabled = _HAS_GOOGLETRANS
        self.translator = Translator() if self.enabled else None

    def normalize_lang(self, lang: str) -> str:
        """
        Normalize incoming language value from UI to a language code
        compatible with translation + TTS.
        """
        if not lang:
            return "en"

        lang = str(lang).strip().lower()

        # Accept both code and name inputs
        mapping = {
            "english": "en",
            "en": "en",
            "marathi": "mr",
            "mr": "mr",
            "hindi": "hi",
            "hi": "hi",
            "gujarati": "gu",
            "gujrati": "gu",  # common misspelling
            "gu": "gu",
            "tamil": "ta",
            "ta": "ta",
            "telugu": "te",
            "telgu": "te",  # common misspelling
            "te": "te",
        }

        return mapping.get(lang, "en")

    def translate(self, text: str, target_lang: str) -> str:
        """
        Translate from English to target_lang.
        If translation backend is unavailable, returns the original text.
        """
        if not text:
            return ""

        code = self.normalize_lang(target_lang)

        # English stays as is
        if code == "en":
            return text

        if not self.enabled or not self.translator:
            # Graceful fallback: no translation, but system still works
            return text

        try:
            result = self.translator.translate(text, src="en", dest=code)
            return result.text
        except Exception as e:
            # On any failure, fall back to English
            print(f"Translation error ({target_lang}): {e}")
            return text

