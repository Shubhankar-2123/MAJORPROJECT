"""
Text-to-Speech Service for Sign Language Translator
Supports multiple voices and languages including Indian languages.
"""

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

import io
import base64
import tempfile
import os
import platform
import threading
from typing import Optional, Dict
try:
    import pythoncom
except Exception:
    pythoncom = None

# gTTS language codes for Indian languages (and common variants)
INDIAN_LANG_GTTS: Dict[str, str] = {
    "en": "en",
    "hi": "hi",   # Hindi
    "mr": "mr",   # Marathi
    "gu": "gu",   # Gujarati
    "ta": "ta",   # Tamil
    "te": "te",   # Telugu
    "bn": "bn",   # Bengali
    "kn": "kn",   # Kannada
    "ml": "ml",   # Malayalam
    "pa": "pa",   # Punjabi (Gurmukhi)
    "ne": "ne",   # Nepali
    "ur": "ur",   # Urdu
}


def _normalize_lang_for_gtts(lang: str) -> str:
    """Normalize language code for gTTS (Indian and other supported codes)."""
    if not lang:
        return "en"
    code = str(lang).strip().lower()
    return INDIAN_LANG_GTTS.get(code, code if len(code) == 2 else "en")


class TTSService:
    """
    Text-to-Speech service with graceful fallbacks:
    - Prefers gTTS (web-friendly MP3)
    - Falls back to pyttsx3 (offline) if gTTS unavailable
    """

    def __init__(self):
        self.engine = None
        if PYTTSX3_AVAILABLE:
            try:
                # Prefer Windows SAPI5 explicitly
                if platform.system() == 'Windows':
                    self.engine = pyttsx3.init(driverName='sapi5')
                else:
                    self.engine = pyttsx3.init()
                self.setup_voice()
            except Exception as e:
                print(f"TTS initialization warning: {e}")
                self.engine = None

    def setup_voice(self):
        """Configure TTS voice settings"""
        if not self.engine:
            return

        try:
            voices = self.engine.getProperty('voices')
            # Prefer female voice for better accessibility
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break

            self.engine.setProperty('rate', 150)  # Speaking rate
            self.engine.setProperty('volume', 0.9)
        except Exception as e:
            print(f"Voice setup error: {e}")

    def speak_text(self, text: str, save_audio: bool = False) -> Optional[str]:
        """
        Convert text to speech
        Returns audio file path if save_audio=True
        """
        if not self.engine:
            return None

        try:
            if save_audio:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    self.engine.save_to_file(text, tmp_file.name)
                    self.engine.runAndWait()
                    return tmp_file.name
            else:
                self.engine.say(text)
                self.engine.runAndWait()
                return None
        except Exception as e:
            print(f"TTS speak error: {e}")
            return None

    def _pyttsx3_to_base64(self, text: str) -> str:
        """Generate base64 audio using pyttsx3 (wav) with safe timeout to prevent hangs."""
        return self._safe_pyttsx3_b64(text, timeout=8)

    def _sapi_worker(self, text: str, out_path: str, result: dict):
        """Worker to synthesize audio using SAPI5 in a separate thread, with COM initialization."""
        try:
            if pythoncom:
                try:
                    pythoncom.CoInitialize()
                except Exception:
                    pass
            try:
                eng = pyttsx3.init(driverName='sapi5') if platform.system() == 'Windows' else pyttsx3.init()
                try:
                    voices = eng.getProperty('voices')
                    for voice in voices:
                        if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                            eng.setProperty('voice', voice.id)
                            break
                    eng.setProperty('rate', 150)
                    eng.setProperty('volume', 0.9)
                except Exception:
                    pass
                eng.save_to_file(text, out_path)
                eng.runAndWait()
                result['ok'] = True
            except Exception as e:
                result['err'] = str(e)
        finally:
            try:
                if pythoncom:
                    pythoncom.CoUninitialize()
            except Exception:
                pass

    def _safe_pyttsx3_b64(self, text: str, timeout: int = 8) -> str:
        """Safely generate WAV base64 via pyttsx3 using a background thread with timeout."""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                out_path = tmp_file.name
            result = {}
            t = threading.Thread(target=self._sapi_worker, args=(text, out_path, result), daemon=True)
            t.start()
            t.join(timeout)
            if not result.get('ok'):
                # Cleanup partial file if any and return empty to avoid blocking
                try:
                    if os.path.exists(out_path):
                        os.unlink(out_path)
                except Exception:
                    pass
                return ""
            with open(out_path, 'rb') as f:
                audio_data = f.read()
                b64 = base64.b64encode(audio_data).decode('utf-8')
            try:
                os.unlink(out_path)
            except Exception:
                pass
            return b64
        except Exception as e:
            print(f"pyttsx3 safe base64 error: {e}")
            return ""

    def get_audio_with_meta(self, text: str, lang: str = 'en') -> tuple[str, str, str]:
        """
        Generate base64 audio and return (audio_base64, mime, ext).
        Supports Indian languages (Hindi, Marathi, Gujarati, Tamil, Telugu, etc.) via gTTS.
        - If gTTS available: returns MP3 with mime 'audio/mpeg' and ext 'mp3'.
        - Else if pyttsx3 available: returns WAV with mime 'audio/wav' and ext 'wav'.
        - Else: returns ("", "", "").
        """
        if not (text and str(text).strip()):
            return "", "", ""

        gtts_lang = _normalize_lang_for_gtts(lang)

        # Prefer gTTS (mp3) - supports Indian languages (hi, mr, gu, ta, te, bn, kn, ml, pa, etc.)
        if GTTS_AVAILABLE:
            try:
                # lang_check=False avoids validation/network issues; Indian codes are valid in gTTS
                tts = gTTS(text=text, lang=gtts_lang, slow=False, lang_check=False)
                # Use mkstemp and close fd immediately so gTTS can write on Windows (avoid WinError 32)
                fd, path = tempfile.mkstemp(suffix='.mp3')
                try:
                    os.close(fd)
                    tts.save(path)
                    with open(path, 'rb') as audio_file:
                        audio_data = audio_file.read()
                        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                    return audio_base64, 'audio/mpeg', 'mp3'
                finally:
                    try:
                        os.unlink(path)
                    except Exception:
                        pass
            except Exception as e:
                print(f"TTS Error (gTTS, lang={gtts_lang}): {e}")
        # Fallback to pyttsx3 (wav) - typically English only on Windows SAPI5
        if PYTTSX3_AVAILABLE:
            b64 = self._safe_pyttsx3_b64(text, timeout=8)
            if b64:
                return b64, 'audio/wav', 'wav'
        return "", "", ""
    def save_to_file(self, text: str, lang: str = 'en', out_dir: str = '.', filename: str = 'tts_output') -> tuple[str, str]:
        """
        Synthesize audio and save to out_dir/filename.<ext> using the correct format.
        Returns (saved_path, mime). If synthesis fails, returns ("", "").
        """
        try:
            os.makedirs(out_dir, exist_ok=True)
            b64, mime, ext = self.get_audio_with_meta(text, lang)
            if not b64:
                return "", ""
            out_path = os.path.join(out_dir, f"{filename}.{ext}")
            with open(out_path, 'wb') as wf:
                wf.write(base64.b64decode(b64))
            return out_path, mime
        except Exception as e:
            print(f"TTS save_to_file error: {e}")
            return "", ""

    def is_available(self) -> bool:
        """Check if TTS is available"""
        return GTTS_AVAILABLE or (self.engine is not None)

    def get_audio_base64(self, text: str, lang: str = 'en') -> str:
        """Return only the base64-encoded audio for the given text and language."""
        b64, mime, ext = self.get_audio_with_meta(text, lang)
        return b64

    def synthesize_to_base64(self, text: str, lang: str = 'en') -> str:
        """Alias of get_audio_base64 for compatibility with tests and APIs."""
        b64, mime, ext = self.get_audio_with_meta(text, lang)
        return b64
