import os
import torch
import secrets

# Project root
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def _resolve_path_env(env_name: str, default_path: str) -> str:
	value = os.getenv(env_name, "").strip()
	if value:
		return os.path.abspath(value)
	return os.path.abspath(default_path)

# Flask Configuration
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "false").lower() == "true"
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"

# Email Configuration
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD", "")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))

# Device and runtime knobs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "30"))

# Model directories
MODELS_MAIN_DIR = _resolve_path_env("MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))
# Normalized static directory; legacy path retained for fallback
STATIC_MAIN_DIR = os.path.join(MODELS_MAIN_DIR, "static")
STATIC_LEGACY_DIR = os.path.join(MODELS_MAIN_DIR, "static images")
WORDS_MAIN_DIR = os.path.join(MODELS_MAIN_DIR, "words")
SENTENCES_MAIN_DIR = os.path.join(MODELS_MAIN_DIR, "sentences")

# Model save configuration (edit here to change filenames/locations)
MODEL_SAVE_CONFIG = {
	"word": {
		"dir": WORDS_MAIN_DIR,
		"latest_model": "words_augmented_model_1.pth",
		"latest_encoder": "word_label_encoder_1.pkl",
		"version_model": "words_augmented_model_{ts}.pth",
		"version_encoder": "word_label_encoder_{ts}.pkl",
	},
	"sentence": {
		"dir": SENTENCES_MAIN_DIR,
		"latest_model": "dynamic_augmented_model_1.pth",
		"latest_encoder": "dynamic_label_encoder_1.pkl",
		"version_model": "dynamic_augmented_model_{ts}.pth",
		"version_encoder": "dynamic_label_encoder_{ts}.pkl",
	},
	"static": {
		"dir": STATIC_MAIN_DIR,
		"latest_model": "static_model.pth",
		"latest_encoder": "static_label_encoder.pkl",
		"latest_scaler": "static_scaler.pkl",
		"version_model": "static_model_{ts}.pth",
	},
}

# Confidence thresholds
DYNAMIC_DEFAULT_THRESHOLD = float(os.getenv("DYNAMIC_DEFAULT_THRESHOLD", "0.60"))
WORD_CONF_THRESHOLD = float(os.getenv("WORD_CONF_THRESHOLD", "0.55"))
SENT_CONF_THRESHOLD = float(os.getenv("SENT_CONF_THRESHOLD", "0.75"))

# Upload safety: 50 MB default
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", str(50 * 1024 * 1024)))

# Data directory
DATA_DIR = _resolve_path_env("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))

# Text processing / LLM (optional)
ENABLE_LLM = os.getenv("ENABLE_LLM", "false").lower() == "true"
LLM_API_URL = os.getenv("LLM_API_URL", "").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
LLM_TIMEOUT_SEC = int(os.getenv("LLM_TIMEOUT_SEC", "10"))
LLM_MIN_MATCH_RATIO = float(os.getenv("LLM_MIN_MATCH_RATIO", "0.5"))

# YouTube API configuration (optional, for fallback video embedding)
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()
YOUTUBE_EMBED_TIMEOUT_SEC = int(os.getenv("YOUTUBE_EMBED_TIMEOUT_SEC", "5"))

# Text simplification cache
TEXT_CACHE_TTL_SECONDS = int(os.getenv("TEXT_CACHE_TTL_SECONDS", str(24 * 3600)))

# User custom signs directory
USER_SIGNS_DIR = os.getenv(
	"USER_SIGNS_DIR",
	os.path.join(DATA_DIR, "custom_signs"),
)