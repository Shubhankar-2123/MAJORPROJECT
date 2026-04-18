

import os
import torch
import torch.nn.functional as F
import joblib
import json
import base64
import time
import tempfile
from werkzeug.utils import secure_filename
from rapidfuzz import process
from flask import Flask, request, jsonify, render_template, send_file, session, redirect, url_for
from urllib.parse import quote
import re
from datetime import datetime
from functools import wraps
import requests

# Load .env manually to prevent Flask's dotenv loading issues
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
except ImportError:
    # If dotenv is not installed, that's OK - use environment variables as-is
    pass

# Ensure imports work regardless of run directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../services")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from text_processing_service import TextProcessingService
from dictionary_service import build_dictionary_entries

from train_static_model import StaticModel
from train_dynamic_new import DynamicLSTM

# Use unified inference service for consistent preprocessing
from inference_service import get_inference_service
from preprocessing_service import PreprocessingService

inference_service = get_inference_service()

# Import new features
from model_agent import IntelligentModelAgent, ModelType
from tts_service import TTSService
from confidence_viz import ConfidenceVisualizer
from feedback_system import FeedbackSystem, FeedbackEntry
from services.performance_monitor import PerformanceMonitor, PerformanceMetric
from translation_service import TranslationService

# -----------------------------
# CONFIG (centralized)
# -----------------------------
from config import (
    DEVICE,
    MAX_FRAMES,
    PROJECT_ROOT,
    MODELS_MAIN_DIR,
    STATIC_MAIN_DIR,
    STATIC_LEGACY_DIR,
    WORDS_MAIN_DIR,
    SENTENCES_MAIN_DIR,
    DYNAMIC_DEFAULT_THRESHOLD,
    WORD_CONF_THRESHOLD,
    SENT_CONF_THRESHOLD,
    MAX_CONTENT_LENGTH,
    DATA_DIR,
    ENABLE_LLM,
    LLM_API_URL,
    LLM_API_KEY,
    LLM_MODEL,
    LLM_TIMEOUT_SEC,
    LLM_MIN_MATCH_RATIO,
    TEXT_CACHE_TTL_SECONDS,
    USER_SIGNS_DIR,
)

# -----------------------------
# LOAD MODELS (Static + Dynamic registries)
# -----------------------------
# Static (A–Z, 1–9)

def _static_artifact_path(filename):
    for base in [STATIC_MAIN_DIR, STATIC_LEGACY_DIR]:
        p = os.path.join(base, filename)
        if os.path.exists(p):
            return p
    return os.path.join(STATIC_MAIN_DIR, filename)

STATIC_MODEL_PATH = _static_artifact_path("static_model.pth")
STATIC_LABEL_ENCODER_PATH = _static_artifact_path("static_label_encoder.pkl")
STATIC_SCALER_PATH = _static_artifact_path("static_scaler.pkl")

with open(STATIC_LABEL_ENCODER_PATH, "rb") as f:
    static_label_encoder = joblib.load(f)

# Human-readable encoder fallback (optional, if compatible with training)
STATIC_HUMAN_LABEL_ENCODER_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "static_label_encoder.pkl")
STATIC_SCALER_FALLBACK_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "static_scaler.pkl")

static_label_encoder_human = None
try:
    if os.path.exists(STATIC_HUMAN_LABEL_ENCODER_PATH):
        with open(STATIC_HUMAN_LABEL_ENCODER_PATH, "rb") as f:
            static_label_encoder_human = joblib.load(f)
except Exception:
    static_label_encoder_human = None

input_dim_static = 126  # 2 hands * 21 landmarks * 3 coords
num_classes_static = len(static_label_encoder.classes_)
static_model = StaticModel(input_dim_static, num_classes_static).to(DEVICE)
static_model.load_state_dict(torch.load(STATIC_MODEL_PATH, map_location=DEVICE))
static_model.eval()

# Dynamic multi-model registry (words + sentences)
import re as _re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

@dataclass
class DynamicEntry:
    name: str
    kind: str  # "word" or "sentence"
    model: Any
    label_encoder: Any
    threshold: float
    model_path: str
    model_mtime: float

def _discover_dynamic_models() -> List[DynamicEntry]:
    entries: List[DynamicEntry] = []

    def _load_pair(kind: str, dir_path: str, pth_pattern: str, le_pattern: str):
        if not os.path.isdir(dir_path):
            return
        for fname in sorted(os.listdir(dir_path)):
            if not fname.lower().endswith('.pth'):
                continue
            m = _re.search(r"(\d+)", fname)
            idx = m.group(1) if m else None
            if idx is None:
                continue
            pth_path = os.path.join(dir_path, fname)
            # find matching label encoder file
            candidate_names = [le_pattern.format(idx=idx)]
            le_path = None
            for cand in candidate_names:
                p = os.path.join(dir_path, cand)
                if os.path.exists(p):
                    le_path = p
                    break
            if not le_path:
                # try any pkl with the same index in name
                for f2 in os.listdir(dir_path):
                    if f2.lower().endswith('.pkl') and (idx in f2):
                        le_path = os.path.join(dir_path, f2)
                        break
            if not le_path:
                continue
            try:
                le = joblib.load(le_path)
                num_classes = len(getattr(le, 'classes_', []))
                mdl = DynamicLSTM(input_size=99, hidden_size=128, num_layers=2, num_classes=num_classes).to(DEVICE)
                state = torch.load(pth_path, map_location=DEVICE)
                fc_weight = state.get("fc.weight") if isinstance(state, dict) else None
                if fc_weight is not None and int(fc_weight.shape[0]) != int(num_classes):
                    print(
                        f"[MODEL_SKIP] {os.path.basename(pth_path)} encoder_classes={num_classes} "
                        f"!= model_out={int(fc_weight.shape[0])}"
                    )
                    continue
                mdl.load_state_dict(state, strict=True)
                mdl.eval()
                name = f"{kind}#{idx}"
                print(f"✅ Loaded model: {os.path.basename(pth_path)} [{kind}#{idx}] with {num_classes} classes")
                entries.append(
                    DynamicEntry(
                        name=name,
                        kind=kind,
                        model=mdl,
                        label_encoder=le,
                        threshold=DYNAMIC_DEFAULT_THRESHOLD,
                        model_path=pth_path,
                        model_mtime=os.path.getmtime(pth_path),
                    )
                )
            except Exception as e:
                # Skip corrupted/incompatible pairs silently to avoid breaking app
                continue

    # Words
    _load_pair("word", WORDS_MAIN_DIR, pth_pattern="words_augmented_model_{idx}.pth", le_pattern="word_label_encoder_{idx}.pkl")
    # Override thresholds for word models
    for e in entries:
        if e.kind == "word":
            e.threshold = WORD_CONF_THRESHOLD
        elif e.kind == "sentence":
            e.threshold = SENT_CONF_THRESHOLD
    # Sentences
    _load_pair("sentence", SENTENCES_MAIN_DIR, pth_pattern="dynamic_augmented_model_{idx}.pth", le_pattern="dynamic_label_encoder_{idx}.pkl")
    # Ensure sentence thresholds
    for e in entries:
        if e.kind == "sentence":
            e.threshold = SENT_CONF_THRESHOLD
    return entries

dynamic_registry: List[DynamicEntry] = _discover_dynamic_models()


def _choose_preferred_dynamic_candidates(candidates: List[DynamicEntry]) -> List[DynamicEntry]:
    if len(candidates) <= 1:
        return candidates

    selected = sorted(candidates, key=lambda e: e.model_mtime, reverse=True)[0]
    print(
        f"[DYNAMIC_MODEL_SELECT] kind={selected.kind} selected={os.path.basename(selected.model_path)} "
        f"from {len(candidates)} candidate models"
    )
    return [selected]

# ===== Preprocessing Wrapper Functions (using unified PreprocessingService) =====
def preprocess_dynamic_video(file, max_frames=MAX_FRAMES, device=DEVICE):
    """
    Preprocess video file using unified PreprocessingService
    Returns PyTorch tensor ready for model input
    """
    import tempfile
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            file.save(tmp.name)
            temp_path = tmp.name
        
        # Use unified preprocessing service
        keypoints = PreprocessingService.preprocess_video_for_inference(temp_path, max_frames)
        
        # Convert to PyTorch tensor with batch dimension and move to device
        tensor = torch.tensor(keypoints, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Clean up temp file
        os.remove(temp_path)
        return tensor
    except Exception as e:
        print(f"❌ Error preprocessing video: {e}")
        raise

def preprocess_static_image(file, scaler_path=None, device=DEVICE):
    """
    Preprocess image file using unified PreprocessingService
    Returns PyTorch tensor ready for model input
    """
    import tempfile
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            file.save(tmp.name)
            temp_path = tmp.name
        
        # Use unified preprocessing service
        keypoints = PreprocessingService.preprocess_image_for_inference(temp_path)
        
        # Convert to PyTorch tensor with batch dimension and move to device
        tensor = torch.tensor(keypoints, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Clean up temp file
        os.remove(temp_path)
        return tensor
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        raise

# ===== End Preprocessing Wrappers =====

# Decide which static scaler to use based on expected feature count
def _choose_static_scaler_path():
    expected = static_model.fc1.in_features if hasattr(static_model, 'fc1') else 126
    def _n_features(path):
        try:
            if os.path.exists(path):
                obj = joblib.load(path)
                return getattr(obj, 'n_features_in_', None)
        except Exception:
            return None
        return None

    p1, p2 = STATIC_SCALER_PATH, STATIC_SCALER_FALLBACK_PATH
    n1, n2 = _n_features(p1), _n_features(p2)
    if n1 == expected:
        return p1, n1
    if n2 == expected:
        return p2, n2
    # If neither matches, prefer the one that exists and report its features
    if n1 is not None:
        return p1, n1
    if n2 is not None:
        return p2, n2
    return p1, None

CHOSEN_STATIC_SCALER_PATH, CHOSEN_STATIC_SCALER_NF = _choose_static_scaler_path()

GESTURE_NOT_CLEAR_MSG = "Gesture not clearly detected. Please try again with proper lighting"
STATIC_CONF_THRESHOLD = 0.60
MODE_TO_KIND = {
    "alphabet": "letters",
    "number": "numbers",
    "word": "word",
    "sentence": "sentence",
}
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')


def _resolve_requested_kind(selected_mode: str, kind_hint: Optional[str]) -> Optional[str]:
    if selected_mode in MODE_TO_KIND:
        return MODE_TO_KIND[selected_mode]
    if kind_hint in {"letters", "numbers", "word", "sentence"}:
        return kind_hint
    return None


def _detect_file_type(filename: str, mimetype: str) -> Tuple[bool, bool, str]:
    lower_name = (filename or "").lower()
    mime = (mimetype or "").lower()
    is_image = any(lower_name.endswith(ext) for ext in IMAGE_EXTS) or mime.startswith('image/')
    is_video = any(lower_name.endswith(ext) for ext in VIDEO_EXTS) or mime.startswith('video/')
    file_type = "image" if is_image else "video" if is_video else "unknown"
    return is_image, is_video, file_type


def _log_prediction_context(selected_mode: str, requested_kind: Optional[str], detected_model: Optional[str], file_type: str):
    print(
        f"[PREDICT] selected_mode={selected_mode}, requested_kind={requested_kind or 'auto'}, "
        f"detected_model={detected_model or 'n/a'}, file_type={file_type}"
    )


def _validate_static_label_for_mode(pred_label: str, requested_kind: Optional[str]) -> Tuple[bool, Optional[str]]:
    if requested_kind not in {"letters", "numbers"}:
        return True, None

    label = (pred_label or "").strip()
    is_letter = len(label) == 1 and label.isalpha()
    is_number = len(label) == 1 and label.isdigit()

    if requested_kind == "letters" and not is_letter:
        return False, "Alphabet mode expects A-Z gestures only. Please show a letter sign."
    if requested_kind == "numbers" and not is_number:
        return False, "Number mode expects 0-9 gestures only. Please show a number sign."
    return True, None


def _predict_static_from_file(file, requested_kind: Optional[str] = None, selected_mode: str = "auto") -> Tuple[Optional[str], float, Optional[str]]:
    tensor = preprocess_static_image(file, scaler_path=CHOSEN_STATIC_SCALER_PATH, device=DEVICE)
    expected_in_features = static_model.fc1.in_features if hasattr(static_model, 'fc1') else 126
    if tensor.dim() != 2 or tensor.shape[1] != expected_in_features:
        return None, 0.0, None

    with torch.no_grad():
        logits = static_model(tensor)
        probs = F.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        conf_v = float(conf.item())
        pred_label = static_label_encoder.inverse_transform([int(pred_idx.item())])[0]

    pred_label_str = str(pred_label)
    print(
        f"[STATIC_PREDICT] selected_mode={selected_mode}, requested_kind={requested_kind or 'auto'}, "
        f"predicted_label={pred_label_str}, confidence={conf_v:.4f}"
    )

    if conf_v < STATIC_CONF_THRESHOLD:
        return None, conf_v, None

    is_valid_label, mismatch_msg = _validate_static_label_for_mode(pred_label_str, requested_kind)
    if not is_valid_label:
        print(
            f"[STATIC_MODE_MISMATCH] selected_mode={selected_mode}, requested_kind={requested_kind}, "
            f"predicted_label={pred_label_str}"
        )
        return None, conf_v, mismatch_msg

    return pred_label_str, conf_v, None


def _predict_dynamic_from_file(file, candidates: List[DynamicEntry]) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None

    tensor = preprocess_dynamic_video(file, max_frames=MAX_FRAMES, device=DEVICE)
    input_mean = float(tensor.mean().item())
    input_std = float(tensor.std().item())
    non_zero_ratio = float((tensor.abs() > 1e-8).float().mean().item())
    print(
        f"[DYNAMIC_INPUT] shape={tuple(tensor.shape)} mean={input_mean:.6f} "
        f"std={input_std:.6f} non_zero_ratio={non_zero_ratio:.4f}"
    )

    candidates = _choose_preferred_dynamic_candidates(candidates)

    best = {"label": None, "confidence": 0.0, "threshold": None, "kind": None, "model": None}
    with torch.no_grad():
        for entry in candidates:
            expected_dim = int(getattr(getattr(entry.model, "encoder", None), "fc1", None).in_features) if hasattr(getattr(entry.model, "encoder", None), "fc1") else 99
            if tensor.dim() != 3 or int(tensor.shape[2]) != expected_dim:
                print(
                    f"[DYNAMIC_SHAPE_MISMATCH] model={entry.name} expected_last_dim={expected_dim} "
                    f"got_shape={tuple(tensor.shape)}"
                )
                continue

            out = entry.model(tensor)
            probs = F.softmax(out, dim=1).squeeze(0)
            conf, idx = torch.max(probs, dim=0)
            conf_v = float(conf.item())
            pred_idx = int(idx.item())

            top_k = min(3, int(probs.shape[0]))
            top_vals, top_idxs = torch.topk(probs, k=top_k)
            top_rows = []
            for tv, ti in zip(top_vals.tolist(), top_idxs.tolist()):
                try:
                    lbl = str(entry.label_encoder.inverse_transform([int(ti)])[0])
                except Exception:
                    lbl = str(int(ti))
                top_rows.append({"idx": int(ti), "label": lbl, "confidence": float(tv)})

            print(
                f"[DYNAMIC_PROBS] model={entry.name} pred_idx={pred_idx} "
                f"top_k={top_rows}"
            )

            if conf_v > best["confidence"]:
                try:
                    label = entry.label_encoder.inverse_transform([pred_idx])[0]
                except Exception:
                    label = pred_idx
                best = {
                    "label": str(label),
                    "confidence": conf_v,
                    "threshold": entry.threshold,
                    "kind": entry.kind,
                    "model": entry.name,
                    "pred_idx": pred_idx,
                    "top_k": top_rows,
                }
    return best if best.get("label") is not None else None

"""
Text-to-Sign helpers: index available media by scanning data/ folders directly.
"""

def _normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _auto_title_from_text(text: str, max_words: int = 7, max_len: int = 40) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", (text or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return "New Chat"
    words = cleaned.split(" ")[:max_words]
    title = " ".join(words)
    if len(title) > max_len:
        title = title[:max_len].rstrip()
    return title

def _iter_existing_dirs(paths):
    for p in paths:
        if os.path.isdir(p):
            yield p


WEB_VIDEO_EXTS = (".mp4", ".webm", ".m4v")
ALL_VIDEO_EXTS = WEB_VIDEO_EXTS + (".mov", ".mkv", ".avi")


def _is_video_file(name: str) -> bool:
    return name.lower().endswith(ALL_VIDEO_EXTS)


def _video_priority(name: str):
    ext = os.path.splitext(name)[1].lower()
    # Prefer browser-friendly formats first, then stable filename ordering.
    if ext in WEB_VIDEO_EXTS:
        return (0, WEB_VIDEO_EXTS.index(ext), name.lower())
    return (1, ext, name.lower())


def _iter_preferred_video_files(files):
    return sorted((f for f in files if _is_video_file(f)), key=_video_priority)


def _guess_video_mimetype(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".mp4", ".m4v"]:
        return "video/mp4"
    if ext == ".webm":
        return "video/webm"
    if ext == ".mov":
        return "video/quicktime"
    if ext == ".mkv":
        return "video/x-matroska"
    if ext == ".avi":
        return "video/x-msvideo"
    return "application/octet-stream"

def _build_phrase_index(base_dirs):
    phrase_to_path = {}
    for base_dir in _iter_existing_dirs(base_dirs):
        for root, _dirs, files in os.walk(base_dir):
            for name in _iter_preferred_video_files(files):
                abs_p = os.path.join(root, name)
                # Candidates from folder and stem
                folder_phrase = os.path.basename(root)
                stem = os.path.splitext(name)[0]
                for candidate in {folder_phrase, stem, stem.replace("_", " "), folder_phrase.replace("_", " ")}:
                    norm = _normalize_text(candidate)
                    if norm and norm not in phrase_to_path:
                        phrase_to_path[norm] = abs_p
    return phrase_to_path

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_STATIC_DIR = os.path.join(DATA_DIR, "static")
PHRASE_BASE_DIRS = [
    os.path.join(DATA_DIR, "dynamic"),
    *[os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR) if d.lower().startswith("dynamic") and os.path.isdir(os.path.join(DATA_DIR, d))],
]
FRAMES_WORD_BASE_DIRS = [
    os.path.join(DATA_DIR, d)
    for d in sorted(os.listdir(DATA_DIR))
    if d.startswith("Frames_Word_Level") and os.path.isdir(os.path.join(DATA_DIR, d))
]
# Optional flat data-based static directory for non-dictionary lookups.
STATIC_VIDEOS_DIR = None
APP_USER_SIGNS_DIR = USER_SIGNS_DIR
try:
    os.makedirs(APP_USER_SIGNS_DIR, exist_ok=True)
except Exception:
    pass
phrase_video_index = _build_phrase_index(PHRASE_BASE_DIRS)

def _find_letter_image(ch: str):
    ch_norm = ch.lower()
    exts = ('.jpg', '.jpeg', '.png', '.webp')
    base = DATA_STATIC_DIR
    if not os.path.isdir(base):
        return None
    # Prefer folder exactly matching the character
    for root, _dirs, files in os.walk(base):
        folder = os.path.basename(root).lower()
        if folder == ch_norm:
            for f in files:
                if f.lower().endswith(exts):
                    return os.path.join(root, f)
    # Fallback: any file named like the character within static
    for root, _dirs, files in os.walk(base):
        for f in files:
            stem = os.path.splitext(f)[0].lower()
            if (stem == ch_norm) and f.lower().endswith(exts):
                return os.path.join(root, f)
    return None


def generate_youtube_queries(text: str) -> Tuple[str, str]:
    text = (text or "").strip().lower()

    isl_query = f"how to sign {text} in Indian Sign Language ISL tutorial step by step"
    asl_query = f"how to sign {text} in American Sign Language ASL tutorial step by step"

    return isl_query, asl_query


def _search_youtube_video(query: str, api_key: str) -> dict:
    """
    Search YouTube for a video matching the query and return video details.
    
    Returns:
        {
            "video_id": str (videoId),
            "title": str,
            "thumbnail": str (URL),
            "embed_url": str (YouTube embed URL)
        }
        or empty dict if no video found or API call fails
    """
    if not api_key:
        return {}
    
    try:
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": 1,
            "key": api_key,
        }
        
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get("items") and len(data["items"]) > 0:
            item = data["items"][0]
            video_id = item.get("id", {}).get("videoId")
            
            if video_id:
                snippet = item.get("snippet", {})
                return {
                    "video_id": video_id,
                    "title": snippet.get("title", ""),
                    "thumbnail": snippet.get("thumbnails", {}).get("default", {}).get("url", ""),
                    "embed_url": f"https://www.youtube.com/embed/{video_id}",
                }
        
        return {}
    
    except (requests.RequestException, ValueError) as e:
        # Log error but don't crash
        print(f"[YOUTUBE_API_ERROR] Failed to search YouTube for '{query}': {str(e)}")
        return {}


def _build_reference_links(word: str) -> dict:
    from config import YOUTUBE_API_KEY
    
    isl_query, asl_query = generate_youtube_queries(word)
    safe_isl = quote(isl_query)
    safe_asl = quote(asl_query)
    safe_google = quote(f"sign language {(word or '').strip()}")
    
    response = {
        "youtube_query": isl_query,
        "youtube_fallback_query": asl_query,
        "youtube_link": f"https://www.youtube.com/results?search_query={safe_isl}",
        "youtube_fallback_link": f"https://www.youtube.com/results?search_query={safe_asl}",
        "google_link": f"https://www.google.com/search?q={safe_google}",
        "search_strategy": "isl_first_asl_fallback",
    }
    
    # Try to fetch actual YouTube videos via API if key is available
    if YOUTUBE_API_KEY:
        # Try primary ISL query first
        isl_result = _search_youtube_video(isl_query, YOUTUBE_API_KEY)
        if isl_result:
            response["youtube_embed"] = isl_result["embed_url"]
            response["youtube_video_id"] = isl_result["video_id"]
            response["youtube_title"] = isl_result["title"]
            response["youtube_thumbnail"] = isl_result["thumbnail"]
        else:
            # Fall back to ASL query
            asl_result = _search_youtube_video(asl_query, YOUTUBE_API_KEY)
            if asl_result:
                response["youtube_embed"] = asl_result["embed_url"]
                response["youtube_video_id"] = asl_result["video_id"]
                response["youtube_title"] = asl_result["title"]
                response["youtube_thumbnail"] = asl_result["thumbnail"]
                response["youtube_using_asl"] = True
        
        # Try fallback ASL query separately
        asl_result = _search_youtube_video(asl_query, YOUTUBE_API_KEY)
        if asl_result and not response.get("youtube_using_asl"):
            response["youtube_fallback_embed"] = asl_result["embed_url"]
            response["youtube_fallback_video_id"] = asl_result["video_id"]
            response["youtube_fallback_title"] = asl_result["title"]
            response["youtube_fallback_thumbnail"] = asl_result["thumbnail"]
    else:
        # Fallback: use direct YouTube search URLs if no API key
        response["youtube_embed"] = None
        response["youtube_embed_error"] = "API key not configured. Using search links instead."
    
    return response

# -----------------------------
# INITIALIZE NEW SERVICES
# -----------------------------
model_agent = IntelligentModelAgent()
tts_service = TTSService()
confidence_viz = ConfidenceVisualizer()
feedback_system = FeedbackSystem(
    db_path=os.path.join(PROJECT_ROOT, "data", "feedback", "feedback.db")
)
performance_monitor = PerformanceMonitor(
    db_path=os.path.join(PROJECT_ROOT, "data", "performance", "performance.db")
)
translation_service = TranslationService()
text_processing_service = TextProcessingService(
    enable_llm=ENABLE_LLM,
    api_url=LLM_API_URL,
    api_key=LLM_API_KEY,
    model=LLM_MODEL,
    timeout_sec=LLM_TIMEOUT_SEC,
    cache_ttl_sec=TEXT_CACHE_TTL_SECONDS,
    min_match_ratio=LLM_MIN_MATCH_RATIO,
)

# -----------------------------
# FLASK APP
# -----------------------------
app = Flask(__name__)

# Disable Flask's auto-dotenv loading (we load it manually above)
app.config['ENV_PREFIX'] = ''  # This prevents Flask from trying to load .env
os.environ['FLASK_SKIP_DOTENV'] = '1'  # Tell Flask to skip dotenv loading

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Session configuration
from config import SECRET_KEY, SESSION_COOKIE_SECURE, SESSION_COOKIE_HTTPONLY, SESSION_COOKIE_SAMESITE
app.config['SECRET_KEY'] = SECRET_KEY
app.config['SESSION_COOKIE_SECURE'] = SESSION_COOKIE_SECURE
app.config['SESSION_COOKIE_HTTPONLY'] = SESSION_COOKIE_HTTPONLY
app.config['SESSION_COOKIE_SAMESITE'] = SESSION_COOKIE_SAMESITE
app.config['PERMANENT_SESSION_LIFETIME'] = 24 * 3600  # 24 hours

# Initialize database
from database.sqlite import init_db
init_db()

# Register blueprints
from flask_app.routes.auth import auth_bp
from flask_app.routes.profile_routes import profile_bp
from flask_app.routes.dashboard import dashboard_bp
from flask_app.routes.predictions import predictions_bp
from flask_app.routes.conversation_routes import conversation_bp
from flask_app.routes.feedback import feedback_bp
from flask_app.routes.custom_signs import custom_signs_bp

app.register_blueprint(auth_bp)
app.register_blueprint(profile_bp)
app.register_blueprint(dashboard_bp)
app.register_blueprint(feedback_bp)
app.register_blueprint(predictions_bp)
app.register_blueprint(conversation_bp)
app.register_blueprint(custom_signs_bp)

# Login required decorator for protected routes
def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        print(f"[LOGIN_REQUIRED] Path: {request.path}, Method: {request.method}, User: {session.get('user_id')}")
        if not session.get("user_id"):
            print(f"[LOGIN_REQUIRED] No user_id, checking conditions...")
            # Always return JSON for API-like routes
            if request.path.startswith('/api/') or request.path.startswith('/predict') or request.method == 'POST':
                print(f"[LOGIN_REQUIRED] Returning JSON 401")
                return jsonify({"error": "Authentication required"}), 401
            # For regular page routes, redirect to login
            if request.accept_mimetypes.accept_html and not request.is_json:
                print(f"[LOGIN_REQUIRED] Redirecting to index")
                return redirect(url_for("index"))
            return jsonify({"error": "Authentication required"}), 401
        return fn(*args, **kwargs)
    return wrapper


@app.route("/customize")
@login_required
def customize_redirect():
    return redirect(url_for("custom_signs.customize_page"))


@app.after_request
def add_no_cache_headers(response):
    if request.path.startswith("/static"):
        return response

    protected_paths = (
        "/app",
        "/preview",
        "/profile",
        "/profile/view",
        "/dashboard",
        "/dictionary",
        "/preview_dict",
        "/chat",
    )

    if session.get("user_id") or request.path.startswith(protected_paths):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

    return response

## Preprocessing is imported from utils/preprocessing.py


# ============================================================================
# MAIN ROUTES
# ============================================================================
@app.route('/')
def index():
    """Landing page for public users."""
    if session.get("user_id"):
        return redirect(url_for("preview_page"))
    return render_template("landing.html")

@app.route('/auth/login')
def login_page():
    """Login page."""
    if session.get("user_id"):
        return redirect(url_for("preview_page"))
    return render_template("login.html")

@app.route('/auth/signup')
def signup_page():
    """Signup page."""
    if session.get("user_id"):
        return redirect(url_for("preview_page"))
    return render_template("signup.html")

@app.route('/auth/password-reset')
def password_reset_page():
    """Password reset page."""
    return render_template("password_reset.html")

@app.route('/preview')
@login_required
def preview_page():
    """Preview page shown after every login."""
    return render_template("preview.html")

@app.route('/app')
@login_required
def app_home():
    """Main application dashboard."""
    return render_template("index.html")


@app.route('/welcome')
def welcome_page():
    if not session.get("user_id"):
        return redirect(url_for("login_page"))
    entries = build_dictionary_entries(PHRASE_BASE_DIRS, FRAMES_WORD_BASE_DIRS, APP_USER_SIGNS_DIR)
    vocab_count = len(entries)
    preview_items = entries[:6]
    return render_template("welcome.html", vocab_count=vocab_count, preview_items=preview_items)


@app.route('/welcome/seen', methods=['POST'])
def welcome_seen():
    if not session.get("user_id"):
        return jsonify({"error": "Authentication required"}), 401
    try:
        from database.models import mark_user_welcome_seen
        mark_user_welcome_seen(session.get("user_id"))
    except Exception:
        pass
    return jsonify({"success": True})


@app.route('/dictionary')
def dictionary_page():
    if not session.get("user_id"):
        return redirect(url_for("login_page"))
    return render_template("dictionary.html")


@app.route('/api/dictionary', methods=['GET'])
def dictionary_api():
    if not session.get("user_id"):
        return jsonify({"error": "Authentication required"}), 401
    entries = build_dictionary_entries(PHRASE_BASE_DIRS, FRAMES_WORD_BASE_DIRS, APP_USER_SIGNS_DIR)
    return jsonify({"count": len(entries), "entries": entries})

@app.route('/tts_test', methods=['GET', 'POST'])
def tts_test():
    """Return base64 TTS audio for verification; also writes an MP3 to static/tts."""
    # Default sample
    text = "Hello from TTS test"
    lang = "en"
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        if isinstance(data, dict):
            text = (data.get('text') or text)
            lang = (data.get('lang') or lang)
    available = tts_service.is_available()
    audio_b64, mime, ext = tts_service.get_audio_with_meta(text, lang)
    resp = {"success": bool(audio_b64), "available": available, "text": text}
    if audio_b64:
        resp["tts_audio"] = audio_b64
        resp["tts_mime"] = mime
        # Save audio to static for manual testing
        try:
            static_dir = os.path.join(os.path.dirname(__file__), "static", "tts")
            os.makedirs(static_dir, exist_ok=True)
            out_path = os.path.join(static_dir, f"tts_test.{ext}")
            with open(out_path, "wb") as wf:
                wf.write(base64.b64decode(audio_b64))
            resp["tts_url"] = f"/static/tts/tts_test.{ext}"
        except Exception:
            pass
    return jsonify(resp)

# -----------------------------
# TEXT-TO-SIGN FUNCTIONALITY
# Multimodal chat persistence (explicit conversation selection)
# -----------------------------

UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)


def _get_conversation_for_user(conversation_id: int, user_id: int):
    from database.conversation_models import get_conversation
    convo = get_conversation(conversation_id)
    if not convo:
        return None
    if convo.get("user_id") != user_id:
        return None
    return convo

@app.route('/text_to_sign', methods=['POST'])
def text_to_sign():
    """Convert input text to sign media by scanning data/ directly.
    Priority:
      1) Exact sentence/phrase video match in data/dynamic*
      2) Per-word videos (best-effort) from data/dynamic*
      3) If a token is a single alphanumeric char, try to serve an image for it
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    text = (data['text'] or '').strip()
    if not text:
        return jsonify({"error": "Empty text provided"}), 400

    conversation_id = data.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "conversation_id is required"}), 400

    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Login required"}), 401

    convo = _get_conversation_for_user(int(conversation_id), user_id)
    if not convo:
        return jsonify({"error": "Invalid conversation"}), 403

    norm_input = _normalize_text(text)

    # 1) Direct phrase
    phrase_abs = phrase_video_index.get(norm_input)
    if phrase_abs and os.path.exists(phrase_abs):
        base = None
        for b in _iter_existing_dirs(PHRASE_BASE_DIRS):
            try:
                if os.path.commonpath([b, phrase_abs]) == b:
                    base = b
                    break
            except Exception:
                pass
        if not base:
            base = PHRASE_BASE_DIRS[0]
        rel = os.path.relpath(phrase_abs, base)
        enc = quote(rel.replace("\\", "/"))
        response = {
            "success": True, 
            "phrase": {"url": f"/dyn_video/{enc}", "text": text}, 
            "words": [], 
            "missing": []
        }
        try:
            from database.conversation_models import create_message, touch_conversation, apply_auto_title
            create_message(
                conversation_id=int(conversation_id),
                sender="user",
                message_type="text",
                text_content=text,
            )
            create_message(
                conversation_id=int(conversation_id),
                sender="system",
                message_type="video",
                text_content=text,
                video_path=response["phrase"]["url"],
            )
            apply_auto_title(int(conversation_id), _auto_title_from_text(text))
            touch_conversation(int(conversation_id))
        except Exception:
            pass
        return jsonify(response)

    # 2) Per-word lookup with hybrid simplification
    results = []
    missing = []

    def _build_word_index():
        idx = {}
        for base in _iter_existing_dirs(PHRASE_BASE_DIRS):
            for root, _dirs, files in os.walk(base):
                word = os.path.basename(root)
                if ' ' in word:
                    continue
                for name in _iter_preferred_video_files(files):
                    key = _normalize_text(word)
                    idx.setdefault(key, os.path.join(root, name))
                    break
        return idx

    def _build_user_sign_index():
        idx = {}
        if not os.path.isdir(APP_USER_SIGNS_DIR):
            return idx
        for root, _dirs, files in os.walk(APP_USER_SIGNS_DIR):
            for name in _iter_preferred_video_files(files):
                stem = os.path.splitext(name)[0]
                key = _normalize_text(stem)
                idx.setdefault(key, os.path.join(root, name))
        return idx

    def _build_static_video_index():
        idx = {}
        if not STATIC_VIDEOS_DIR or not os.path.isdir(STATIC_VIDEOS_DIR):
            return idx
        for name in _iter_preferred_video_files(os.listdir(STATIC_VIDEOS_DIR)):
            stem = os.path.splitext(name)[0].replace("_", " ")
            key = _normalize_text(stem)
            idx.setdefault(key, os.path.join(STATIC_VIDEOS_DIR, name))
        return idx

    word_video_index = _build_word_index()
    user_sign_index = _build_user_sign_index()
    static_video_index = _build_static_video_index()
    available_words = set(word_video_index.keys()) | set(user_sign_index.keys()) | set(static_video_index.keys())

    simplify_result = text_processing_service.hybrid_simplify(text, available_words)
    tokens = simplify_result.get("tokens") or []
    if not tokens:
        tokens = [t for t in re.split(r"\s+", text) if t]

    matched_words = []

    for tok in tokens:
        norm_tok = _normalize_text(tok)
        if not norm_tok:
            continue
        # Single character image case first
        if len(norm_tok) == 1 and norm_tok.isalnum():
            img_abs = _find_letter_image(norm_tok)
            if img_abs and os.path.exists(img_abs):
                rel = os.path.relpath(img_abs, DATA_STATIC_DIR)
                enc = quote(rel.replace("\\", "/"))
                url = f"/static_image/{enc}"
                results.append({"word": tok, "url": url})
                matched_words.append(tok)
                continue

        # Custom user sign lookup
        user_abs = user_sign_index.get(norm_tok)
        if user_abs and os.path.exists(user_abs):
            rel = os.path.relpath(user_abs, APP_USER_SIGNS_DIR)
            enc = quote(rel.replace("\\", "/"))
            results.append({"word": tok, "url": f"/user_signs/{enc}", "source": "user"})
            matched_words.append(tok)
            continue

        # Optional flat static videos (data-based path only, if configured)
        static_abs = static_video_index.get(norm_tok)
        if static_abs and os.path.exists(static_abs):
            filename = os.path.basename(static_abs)
            results.append({"word": tok, "url": f"/video/{quote(filename)}", "source": "system"})
            matched_words.append(tok)
            continue

        # System word video lookup (data/dynamic*)
        vid_abs = word_video_index.get(norm_tok)
        if vid_abs and os.path.exists(vid_abs):
            base = None
            for b in _iter_existing_dirs(PHRASE_BASE_DIRS):
                try:
                    common = os.path.commonpath([b, vid_abs])
                    if common == b:
                        base = b
                        break
                except Exception:
                    pass
            if not base:
                base = PHRASE_BASE_DIRS[0]
            rel = os.path.relpath(vid_abs, base)
            enc = quote(rel.replace("\\", "/"))
            results.append({"word": tok, "url": f"/dyn_video/{enc}", "source": "system"})
            matched_words.append(tok)
        else:
            links = _build_reference_links(norm_tok)
            missing.append({"word": tok, "message": "media not found", **links})

    response = {
        "success": True,
        "phrase": None,
        "words": results,
        "missing": missing,
        "available": matched_words,
        "original_text": text,
        "simplify_source": simplify_result.get("source"),
    }
    
    try:
        from database.conversation_models import create_message, touch_conversation, apply_auto_title
        create_message(
            conversation_id=int(conversation_id),
            sender="user",
            message_type="text",
            text_content=text,
        )
        for word_item in results:
            if word_item.get("url"):
                create_message(
                    conversation_id=int(conversation_id),
                    sender="system",
                    message_type="video",
                    text_content=word_item.get("word") or "Sign",
                    video_path=word_item.get("url"),
                )
        if missing:
            missing_words = ", ".join([m.get("word") for m in missing if m.get("word")])
            create_message(
                conversation_id=int(conversation_id),
                sender="system",
                message_type="text",
                text_content=f"Missing words: {missing_words}" if missing_words else "Missing words found",
            )
        apply_auto_title(int(conversation_id), _auto_title_from_text(text))
        touch_conversation(int(conversation_id))
    except Exception:
        pass
    
    return jsonify(response)

@app.route('/video/<path:filename>')
def serve_video(filename):
    """Serve videos from data-based sources by filename (legacy /video URL compatibility)."""
    safe_name = os.path.basename(filename)
    if not safe_name or safe_name != filename:
        return jsonify({"error": "Invalid filename"}), 400

    # Optional flat data-based static dir support.
    if STATIC_VIDEOS_DIR and os.path.isdir(STATIC_VIDEOS_DIR):
        static_abs = os.path.join(STATIC_VIDEOS_DIR, safe_name)
        if os.path.exists(static_abs):
            return send_file(static_abs, mimetype=_guess_video_mimetype(static_abs))

    # Fallback: search data/dynamic* roots for backward compatibility.
    for base in _iter_existing_dirs(PHRASE_BASE_DIRS):
        try:
            for root, _dirs, files in os.walk(base):
                if safe_name in files:
                    abs_path = os.path.join(root, safe_name)
                    if _is_video_file(abs_path):
                        return send_file(abs_path, mimetype=_guess_video_mimetype(abs_path))
        except Exception:
            continue

    return jsonify({"error": "Video not found", "filename": filename}), 404


@app.route('/user_signs/<path:relpath>')
def serve_user_signs(relpath):
    """Serve user custom sign videos safely from uploads/custom_signs."""
    relpath = relpath.replace("\\", "/")
    candidate = os.path.normpath(os.path.join(APP_USER_SIGNS_DIR, relpath))
    try:
        common = os.path.commonpath([APP_USER_SIGNS_DIR, candidate])
    except Exception:
        common = ''
    if common != APP_USER_SIGNS_DIR:
        return jsonify({"error": "Invalid path"}), 400
    if not os.path.exists(candidate):
        return jsonify({"error": "Video not found", "relpath": relpath}), 404
    return send_file(candidate, mimetype=_guess_video_mimetype(candidate))


@app.route('/uploads/<path:relpath>')
@login_required
def serve_user_uploads(relpath):
    """Serve user uploads from uploads/user_<id>/ only for the owner."""
    user_id = session.get("user_id")
    relpath = relpath.replace("\\", "/")
    expected_prefix = f"user_{user_id}/"
    if not relpath.startswith(expected_prefix):
        return jsonify({"error": "Access denied"}), 403
    candidate = os.path.normpath(os.path.join(UPLOADS_DIR, relpath))
    try:
        common = os.path.commonpath([UPLOADS_DIR, candidate])
    except Exception:
        common = ''
    if common != UPLOADS_DIR:
        return jsonify({"error": "Invalid path"}), 400
    if not os.path.exists(candidate):
        return jsonify({"error": "File not found", "relpath": relpath}), 404
    return send_file(candidate)


@app.route('/frames_video/<path:relpath>')
def serve_frames_video(relpath):
    """Serve videos from data/Frames_Word_Level* directories by relative path, ignoring images.

    Security: ensures request stays within base directory.
    """
    # Normalize separators in relpath
    relpath = relpath.replace("\\", "/")
    
    # Find all Frames_Word_Level* directories
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    frames_dirs = []
    if os.path.isdir(data_dir):
        for item in sorted(os.listdir(data_dir)):
            if item.startswith("Frames_Word_Level"):
                full_path = os.path.join(data_dir, item)
                if os.path.isdir(full_path):
                    frames_dirs.append(full_path)
    
    # Try each Frames_Word_Level_* directory
    for base_dir in frames_dirs:
        # Build absolute candidate
        candidate = os.path.normpath(os.path.join(base_dir, relpath))
        # Ensure path traversal protection
        try:
            common = os.path.commonpath([base_dir, candidate])
        except Exception:
            common = ''
        if common != base_dir:
            continue
        if not os.path.exists(candidate):
            continue
        
        # File found, serve it
        break
    else:
        # File not found in any directory
        return jsonify({"error": "Video not found", "relpath": relpath}), 404
    
    # Reject images explicitly
    ext = os.path.splitext(candidate)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
        return jsonify({"error": "Not a video file"}), 400
    mimetype = 'application/octet-stream'
    if ext in ['.mp4', '.m4v']:
        mimetype = 'video/mp4'
    elif ext in ['.webm']:
        mimetype = 'video/webm'
    elif ext in ['.mov']:
        mimetype = 'video/quicktime'
    elif ext in ['.mkv']:
        mimetype = 'video/x-matroska'
    elif ext in ['.avi']:
        mimetype = 'video/x-msvideo'
    return send_file(candidate, mimetype=mimetype)

@app.route('/dyn_video/<path:relpath>')
def serve_dynamic_phrase_video(relpath):
    """Safely serve videos from data/dynamic* by relative path.
    Tries to resolve against any dynamic* base.
    """
    relpath = relpath.replace("\\", "/")
    for base in _iter_existing_dirs(PHRASE_BASE_DIRS):
        candidate = os.path.normpath(os.path.join(base, relpath))
        try:
            common = os.path.commonpath([base, candidate])
        except Exception:
            common = ''
        if common == base and os.path.exists(candidate):
            return send_file(candidate, mimetype=_guess_video_mimetype(candidate))
    return jsonify({"error": "Phrase video not found", "relpath": relpath}), 404

@app.route('/available_words', methods=['GET'])
def get_available_words():
    """List unique single-word folder names found under data/dynamic* that contain at least one video."""
    exts = (".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm")
    words = set()
    for base in _iter_existing_dirs(PHRASE_BASE_DIRS):
        for root, _dirs, files in os.walk(base):
            name = os.path.basename(root)
            if ' ' in name:
                continue
            if any(f.lower().endswith(exts) for f in files):
                words.add(name)
    return jsonify({"count": len(words), "words": sorted(words)})

@app.route('/predict_static', methods=['POST'])
@login_required
def predict_static():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided (field 'image')"}), 400

    file = request.files['image']
    selected_mode = (request.form.get('mode') or request.args.get('mode') or 'alphabet').strip().lower()
    kind_hint = request.form.get('kind') or request.args.get('kind')
    requested_kind = _resolve_requested_kind(selected_mode, kind_hint)
    
    # AUTO mode: default to letters for static images
    if selected_mode == "auto" and not requested_kind:
        requested_kind = "letters"
        print(f"[AUTO_MODE_STATIC] Auto mode detected for image, defaulting to letters model")
    
    _log_prediction_context(selected_mode, requested_kind, None, "image")

    if requested_kind in ["word", "sentence"]:
        return jsonify({"error": f"Mode '{selected_mode}' accepts videos only. Please upload a video file."}), 400

    try:
        pred_label, conf_v, mismatch_msg = _predict_static_from_file(
            file,
            requested_kind=requested_kind,
            selected_mode=selected_mode,
        )
        if mismatch_msg:
            return jsonify({"error": mismatch_msg, "confidence": conf_v, "mode_mismatch": True}), 422
        if not pred_label:
            return jsonify({"prediction": GESTURE_NOT_CLEAR_MSG, "confidence": conf_v})

        tts_audio = ""
        tts_mime = ""
        tts_url = ""
        if tts_service.is_available():
            try:
                tts_audio, tts_mime, ext = tts_service.get_audio_with_meta(pred_label, lang="en")
                if tts_audio:
                    try:
                        static_dir = os.path.join(os.path.dirname(__file__), "static", "tts")
                        os.makedirs(static_dir, exist_ok=True)
                        stamp = int(time.time() * 1000)
                        out_path = os.path.join(static_dir, f"static_{stamp}.{ext}")
                        with open(out_path, "wb") as wf:
                            wf.write(base64.b64decode(tts_audio))
                        tts_url = f"/static/tts/static_{stamp}.{ext}"
                    except Exception:
                        pass
            except Exception:
                pass
        # Save prediction to database
        from database.models import create_prediction
        user_id = session.get('user_id')
        prediction_record = create_prediction(
            user_id=user_id,
            input_type='static',
            input_path=None,
            input_text=None,
            predicted_text=pred_label,
            translated_text=None,
            confidence=conf_v,
            model_used='static_model',
            tts_audio_path=None
        )
        return jsonify({"prediction": pred_label, "confidence": conf_v, "tts_audio": tts_audio, "tts_mime": tts_mime, "tts_url": tts_url, "tts_success": bool(tts_audio), "tts_error": "" if tts_audio else "TTS unavailable or failed", "prediction_id": prediction_record.get('id')})
    except Exception as e:
        print(f"[PREDICT_STATIC_ERROR] {e}")
        return jsonify({"prediction": GESTURE_NOT_CLEAR_MSG, "confidence": 0.0})

@app.route('/predict_dynamic', methods=['POST'])
@login_required
def predict_dynamic():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided (field 'video')"}), 400

    file = request.files['video']
    selected_mode = (request.form.get('mode') or request.args.get('mode') or '').strip().lower()
    kind_hint = request.form.get('kind') or request.args.get('kind')
    if selected_mode == "word":
        requested_kind = "word"
    elif selected_mode == "sentence":
        requested_kind = "sentence"
    else:
        requested_kind = _resolve_requested_kind(selected_mode, kind_hint)
    
    # AUTO mode: default to word for dynamic videos (can be overridden by model_agent if sentence detected)
    if selected_mode == "auto" and not requested_kind:
        requested_kind = "word"
        print(f"[AUTO_MODE_DYNAMIC] Auto mode detected for video, defaulting to word model")

    if requested_kind in ["letters", "numbers"]:
        return jsonify({"error": f"Mode '{selected_mode}' accepts images only. Please upload an image file."}), 400

    if requested_kind not in ["word", "sentence"]:
        return jsonify({"error": "Model type required: specify kind='word' or kind='sentence'"}), 400

    detected_model = None
    try:
        analysis = model_agent.analyze_file_storage(file)
        detected_model = analysis.recommended_model.value
    except Exception:
        detected_model = None
    finally:
        try:
            file.stream.seek(0)
        except Exception:
            pass

    _log_prediction_context(selected_mode or requested_kind, requested_kind, detected_model, "video")

    try:
        if not dynamic_registry:
            return jsonify({"prediction": GESTURE_NOT_CLEAR_MSG, "confidence": 0.0})

        candidates = [e for e in dynamic_registry if e.kind == requested_kind]
        if not candidates:
            return jsonify({"error": f"No models available for type '{requested_kind}'"}), 404

        best = _predict_dynamic_from_file(file, candidates)
        if not best or not best.get("label"):
            return jsonify({"prediction": GESTURE_NOT_CLEAR_MSG, "confidence": 0.0})

        thr = best.get("threshold") or (WORD_CONF_THRESHOLD if best.get("kind") == "word" else SENT_CONF_THRESHOLD)
        if best["confidence"] < float(thr):
            return jsonify({"prediction": GESTURE_NOT_CLEAR_MSG, "confidence": best["confidence"]})

        tts_audio = ""
        tts_mime = ""
        tts_url = ""
        if tts_service.is_available():
            try:
                tts_audio, tts_mime, ext = tts_service.get_audio_with_meta(best["label"], lang="en")
                if tts_audio:
                    try:
                        static_dir = os.path.join(os.path.dirname(__file__), "static", "tts")
                        os.makedirs(static_dir, exist_ok=True)
                        stamp = int(time.time() * 1000)
                        out_path = os.path.join(static_dir, f"dynamic_{stamp}.{ext}")
                        with open(out_path, "wb") as wf:
                            wf.write(base64.b64decode(tts_audio))
                        tts_url = f"/static/tts/dynamic_{stamp}.{ext}"
                    except Exception:
                        pass
            except Exception:
                pass
        # Save prediction to database
        from database.models import create_prediction
        user_id = session.get('user_id')
        prediction_record = create_prediction(
            user_id=user_id,
            input_type='dynamic',
            input_path=None,
            input_text=None,
            predicted_text=best["label"],
            translated_text=None,
            confidence=best["confidence"],
            model_used=best["model"],
            tts_audio_path=None
        )

        return jsonify({"prediction": best["label"], "confidence": best["confidence"], "model_type": best.get("kind"), "prediction_idx": best.get("pred_idx"), "top_k": best.get("top_k", []), "tts_audio": tts_audio, "tts_mime": tts_mime, "tts_url": tts_url, "tts_success": bool(tts_audio), "tts_error": "" if tts_audio else "TTS unavailable or failed", "prediction_id": prediction_record.get('id')})
    except Exception as e:
        print(f"[PREDICT_DYNAMIC_ERROR] {e}")
        return jsonify({"prediction": GESTURE_NOT_CLEAR_MSG, "confidence": 0.0})


@app.route('/api/feedback', methods=['POST'])
@login_required
def submit_feedback_api():
    """Submit user feedback for prediction correction (SQLite-backed)."""
    data = request.get_json(silent=True) or {}
    prediction_id = data.get('prediction_id')
    correction_text = (data.get('correction_text') or '').strip()
    original_text = data.get('original_text') or ''
    if not correction_text:
        return jsonify({"error": "Correction text required"}), 400
    try:
        from database.models import create_feedback
        user_id = session.get('user_id')
        feedback = create_feedback(user_id, prediction_id, correction_text, original_text)
        return jsonify({"success": True, "feedback": feedback})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided (field 'file')"}), 400

    file = request.files['file']
    if not file or not file.filename:
        return jsonify({"error": "Empty upload"}), 400

    start_time = time.time()
    conversation_id = request.form.get("conversation_id") or request.args.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "conversation_id is required"}), 400

    user_id = session.get("user_id")
    convo = _get_conversation_for_user(int(conversation_id), user_id)
    if not convo:
        return jsonify({"error": "Invalid conversation"}), 403
    filename = file.filename
    mimetype = (file.mimetype or "").lower()
    target_lang = translation_service.normalize_lang(request.form.get('lang') or request.args.get('lang') or 'en')
    selected_mode = (request.form.get('mode') or request.args.get('mode') or 'auto').strip().lower()
    kind_hint = request.form.get('kind') or request.args.get('kind')

    if selected_mode not in {"auto", "alphabet", "number", "word", "sentence"}:
        return jsonify({"error": "Invalid mode. Use one of: auto, alphabet, number, word, sentence"}), 400

    is_image, is_video, file_type = _detect_file_type(filename, mimetype)

    if not is_image and not is_video:
        return jsonify({"error": "Unsupported file type"}), 400

    # For AUTO mode, skip early validation—let file type determine model selection
    if selected_mode != "auto":
        if selected_mode == "word":
            requested_kind = "word"
        elif selected_mode == "sentence":
            requested_kind = "sentence"
        elif selected_mode == "alphabet":
            requested_kind = "letters"
        elif selected_mode == "number":
            requested_kind = "numbers"
        else:
            requested_kind = _resolve_requested_kind(selected_mode, kind_hint)
        # STRICT VALIDATION: enforce input type matching for manual modes
        if requested_kind in ["letters", "numbers"] and not is_image:
            return jsonify({"error": f"Mode '{selected_mode}' accepts images only. Please upload an image file."}), 400
        if requested_kind in ["word", "sentence"] and not is_video:
            return jsonify({"error": f"Mode '{selected_mode}' accepts videos only. Please upload a video file."}), 400
    else:
        # AUTO mode: will be determined later based on file type
        requested_kind = None

    detected_model = None
    if is_video:
        try:
            analysis = model_agent.analyze_file_storage(file)
            detected_model = analysis.recommended_model.value
        finally:
            try:
                file.stream.seek(0)
            except Exception:
                pass

    _log_prediction_context(selected_mode, requested_kind, detected_model, file_type)

    # ========== AUTO MODE DETECTION: File type-based routing ==========
    if selected_mode == "auto":
        # AUTO mode overrides all manual restrictions and routes based on file type
        if is_image:
            requested_kind = "letters"
            print(f"[AUTO_MODE_IMAGE] Auto mode detected image input, routing to STATIC model")
        elif is_video:
            # If video in AUTO mode and detected_model is available, use it
            if detected_model in ["word", "sentence"]:
                requested_kind = detected_model
                print(f"[AUTO_MODE_VIDEO] Auto mode detected video input, detected model type: {detected_model}")
            else:
                # Default video to word-level if detection failed
                requested_kind = "word"
                print(f"[AUTO_MODE_VIDEO_DEFAULT] Auto mode detected video input, defaulting to word")

    try:
        # Save upload to user-specific folder
        user_dir = os.path.join(UPLOADS_DIR, f"user_{user_id}")
        os.makedirs(user_dir, exist_ok=True)
        safe_name = secure_filename(filename) or f"upload_{int(time.time() * 1000)}"
        stamped_name = f"{int(time.time() * 1000)}_{safe_name}"
        saved_path = os.path.join(user_dir, stamped_name)
        file.stream.seek(0)
        file.save(saved_path)
        file.stream.seek(0)
        upload_rel = f"user_{user_id}/{stamped_name}"
        upload_url = f"/uploads/{upload_rel}"

        try:
            from database.conversation_models import create_message, touch_conversation, apply_auto_title
            create_message(
                conversation_id=int(conversation_id),
                sender="user",
                message_type="video",
                video_path=upload_url,
            )
            apply_auto_title(int(conversation_id), "Media upload")
            touch_conversation(int(conversation_id))
        except Exception:
            pass

        model_type = "static"
        prediction_en = GESTURE_NOT_CLEAR_MSG
        confidence = 0.0
        model_used = "unknown"
        success = False

        if is_image:
            pred_label, conf_v, mismatch_msg = _predict_static_from_file(
                file,
                requested_kind=requested_kind,
                selected_mode=selected_mode,
            )
            if mismatch_msg:
                processing_time = time.time() - start_time
                performance_monitor.record_metric(PerformanceMetric(
                    timestamp=datetime.now().isoformat(),
                    model_type="static",
                    processing_time=processing_time,
                    confidence=conf_v,
                    prediction=mismatch_msg,
                    success=False
                ))
                return jsonify({"error": mismatch_msg, "confidence": conf_v, "mode_mismatch": True}), 422
            if not pred_label:
                processing_time = time.time() - start_time
                performance_monitor.record_metric(PerformanceMetric(
                    timestamp=datetime.now().isoformat(),
                    model_type="static",
                    processing_time=processing_time,
                    confidence=conf_v,
                    prediction=GESTURE_NOT_CLEAR_MSG,
                    success=False
                ))
                return jsonify({"prediction": GESTURE_NOT_CLEAR_MSG, "confidence": conf_v})

            prediction_en = pred_label
            confidence = conf_v
            model_type = "static"
            model_used = "static_model"
            success = True
        else:
            if not dynamic_registry:
                return jsonify({"prediction": GESTURE_NOT_CLEAR_MSG, "confidence": 0.0})

            if requested_kind in ["word", "sentence"]:
                candidates = [e for e in dynamic_registry if e.kind == requested_kind]
            elif selected_mode == "auto":
                candidates = dynamic_registry
            else:
                return jsonify({"error": "Invalid model selection for video input"}), 400

            if not candidates:
                return jsonify({"error": f"No models available for type '{requested_kind}'"}), 404

            best = _predict_dynamic_from_file(file, candidates)
            if not best or not best.get("label"):
                processing_time = time.time() - start_time
                performance_monitor.record_metric(PerformanceMetric(
                    timestamp=datetime.now().isoformat(),
                    model_type=requested_kind or "dynamic",
                    processing_time=processing_time,
                    confidence=0.0,
                    prediction=GESTURE_NOT_CLEAR_MSG,
                    success=False
                ))
                return jsonify({"prediction": GESTURE_NOT_CLEAR_MSG, "confidence": 0.0})

            thr = best.get("threshold") or (WORD_CONF_THRESHOLD if best.get("kind") == "word" else SENT_CONF_THRESHOLD)
            if best["confidence"] < float(thr):
                processing_time = time.time() - start_time
                performance_monitor.record_metric(PerformanceMetric(
                    timestamp=datetime.now().isoformat(),
                    model_type=best.get("kind", "dynamic"),
                    processing_time=processing_time,
                    confidence=best["confidence"],
                    prediction=GESTURE_NOT_CLEAR_MSG,
                    success=False
                ))
                return jsonify({"prediction": GESTURE_NOT_CLEAR_MSG, "confidence": best["confidence"]})

            prediction_en = best["label"]
            confidence = best["confidence"]
            model_type = best.get("kind", "dynamic")
            model_used = best.get("model", "dynamic_model")
            success = True

        processing_time = time.time() - start_time

        # Record performance metric
        performance_monitor.record_metric(PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            model_type=model_type,
            processing_time=processing_time,
            confidence=confidence,
            prediction=prediction_en,
            success=success
        ))

        print(
            f"[PREDICT_RESULT] file_type={file_type}, selected_mode={selected_mode}, "
            f"requested_kind={requested_kind}, model_used={model_used}, "
            f"prediction={prediction_en}, confidence={confidence:.4f}"
        )

        # Step 3: Translate to selected language
        translated_text = translation_service.translate(prediction_en, target_lang)

        # Generate confidence visualization
        confidence_img = confidence_viz.create_confidence_bar(confidence)

        # Step 4: Generate TTS audio in selected language
        tts_audio = ""
        tts_mime = ""
        tts_url = ""
        if tts_service.is_available():
            try:
                # Use translated text for speech if not English
                tts_text = translated_text if target_lang != "en" else prediction_en
                tts_audio, tts_mime, ext = tts_service.get_audio_with_meta(tts_text, lang=target_lang)
                if tts_audio:
                    try:
                        static_dir = os.path.join(os.path.dirname(__file__), "static", "tts")
                        os.makedirs(static_dir, exist_ok=True)
                        stamp = int(time.time() * 1000)
                        out_path = os.path.join(static_dir, f"dynamic_{stamp}.{ext}")
                        with open(out_path, "wb") as wf:
                            wf.write(base64.b64decode(tts_audio))
                        tts_url = f"/static/tts/dynamic_{stamp}.{ext}"
                    except Exception:
                        pass
            except Exception:
                pass

        # Save prediction to database
        from database.models import create_prediction
        user_id = session.get('user_id')
        prediction_record = create_prediction(
            user_id=user_id,
            input_type=model_type,
            input_path=None,
            input_text=None,
            predicted_text=prediction_en,
            translated_text=translated_text,
            confidence=confidence,
            model_used=model_used,
            tts_audio_path=tts_url if tts_url else None
        )

        # Save to conversation if successful prediction
        if success and user_id:
            from database.conversation_models import create_message, touch_conversation
            create_message(
                conversation_id=int(conversation_id),
                sender="system",
                message_type="text",
                text_content=prediction_en,
                prediction=prediction_en,
                confidence=confidence,
            )
            touch_conversation(int(conversation_id))

        return jsonify({
            "prediction": prediction_en,          # English base text
            "translated": translated_text,        # Text in selected language (or English)
            "lang": target_lang,
            "confidence": confidence,
            "model_type": model_type,
            "model_used": model_used,
            "prediction_idx": best.get("pred_idx") if (not is_image and best) else None,
            "top_k": best.get("top_k", []) if (not is_image and best) else [],
            "processing_time": round(processing_time, 3),
            "confidence_image": confidence_img,
            "tts_audio": tts_audio,
            "tts_mime": tts_mime,
            "tts_url": tts_url,
            "tts_success": bool(tts_audio),
            "tts_error": "" if tts_audio else "TTS unavailable or failed",
            "prediction_id": prediction_record.get('id'),
            "mode_info": f"Auto mode selected: using {model_type} model" if selected_mode == "auto" else f"Manual mode: {selected_mode}"
        })

    except Exception as e:
        processing_time = time.time() - start_time
        performance_monitor.record_metric(PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            model_type="unknown",
            processing_time=processing_time,
            confidence=0.0,
            prediction="error",
            success=False
        ))
        return jsonify({"prediction": GESTURE_NOT_CLEAR_MSG, "confidence": 0.0, "error": str(e)})


# -----------------------------
# NEW FEATURE ROUTES
# -----------------------------

@app.route('/api/model_recommendation', methods=['POST'])
def get_model_recommendation():
    """Get intelligent model recommendation for uploaded file"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    tmp_file = None
    try:
        import tempfile
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tmp_path = tmp_file.name
        tmp_file.close()
        file.save(tmp_path)
        
        recommendation = model_agent.get_model_recommendation(tmp_path)
        return jsonify(recommendation)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_file and os.path.exists(tmp_file.name):
            try:
                os.unlink(tmp_file.name)
            except:
                pass

@app.route('/api/session', methods=['GET'])
def get_session_status():
    user_id = session.get('user_id')
    username = session.get('username')
    return jsonify({
        "logged_in": bool(user_id),
        "user": {"id": user_id, "username": username} if user_id else None
    })

@app.route('/api/conversation/active', methods=['GET', 'POST'])
@login_required
def get_or_create_active_conversation():
    """Get or create the active conversation for current user."""
    from database.conversation_models import create_conversation
    user_id = session.get('user_id')
    
    if request.method == 'POST':
        title = request.get_json(silent=True).get('title', 'New Chat') if request.is_json else 'New Chat'
        convo = create_conversation(user_id, title)
        session['active_conversation_id'] = convo.get('id')
        return jsonify({"success": True, "conversation": convo})
    
    # GET: return active only
    active_id = session.get('active_conversation_id')
    if active_id:
        from database.conversation_models import get_conversation
        convo = get_conversation(active_id)
        if convo and convo.get('user_id') == user_id:
            return jsonify({"success": True, "conversation": convo})
    return jsonify({"error": "No active conversation"}), 404

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback for prediction correction"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    try:
        feedback = FeedbackEntry(
            timestamp=datetime.now().isoformat(),
            input_file=data.get('input_file', ''),
            predicted_label=data.get('predicted_label', ''),
            actual_label=data.get('actual_label', ''),
            confidence=float(data.get('confidence', 0.0)),
            model_used=data.get('model_used', 'unknown'),
            user_id=data.get('user_id')
        )
        
        success = feedback_system.collect_feedback(feedback)
        if success:
            return jsonify({"success": True, "message": "Feedback recorded"})
        else:
            return jsonify({"error": "Failed to save feedback"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/feedback/stats', methods=['GET'])
def get_feedback_stats():
    """Get feedback statistics"""
    try:
        stats = feedback_system.get_feedback_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/performance/stats', methods=['GET'])
def get_performance_stats():
    """Get performance statistics"""
    try:
        days = int(request.args.get('days', 7))
        stats = performance_monitor.get_performance_stats(days=days)
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/tts', methods=['POST'])
def text_to_speech():
    """Generate TTS audio for text. For Indian languages (non-en), text is translated first then spoken."""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"success": False, "error": "No text provided"}), 400
    
    try:
        text = (data['text'] or '').strip()
        if not text:
            return jsonify({"success": False, "error": "Empty text"}), 400
        lang = translation_service.normalize_lang(data.get('lang', 'en'))
        # For Indian languages: translate English text to target language so TTS speaks in that language
        if lang != 'en':
            text = translation_service.translate(text, lang)
        available = tts_service.is_available()
        audio_b64, mime, ext = tts_service.get_audio_with_meta(text, lang)
        resp = {
            "success": bool(audio_b64),
            "available": available,
            "audio_base64": audio_b64 or "",
            "mime": mime or "",
            "ext": ext or ""
        }
        if audio_b64:
            try:
                static_dir = os.path.join(os.path.dirname(__file__), "static", "tts")
                os.makedirs(static_dir, exist_ok=True)
                stamp = int(time.time() * 1000)
                out_path = os.path.join(static_dir, f"api_tts_{stamp}.{ext}")
                with open(out_path, "wb") as wf:
                    wf.write(base64.b64decode(audio_b64))
                resp["url"] = f"/static/tts/api_tts_{stamp}.{ext}"
            except Exception as se:
                resp["error"] = f"Saved file error: {str(se)}"
        else:
            resp["error"] = "TTS not available or failed"
        return jsonify(resp)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/confidence_viz', methods=['POST'])
def get_confidence_visualization():
    """Generate confidence visualization"""
    data = request.get_json()
    if not data or 'confidence' not in data:
        return jsonify({"error": "No confidence value provided"}), 400
    
    try:
        confidence = float(data['confidence'])
        threshold = float(data.get('threshold', 0.7))
        image_base64 = confidence_viz.create_confidence_bar(confidence, threshold)
        
        if image_base64:
            return jsonify({
                "success": True,
                "image": image_base64,
                "format": "png"
            })
        else:
            return jsonify({"error": "Visualization not available"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# DEBUG/INSPECTION ROUTES
# -----------------------------
@app.route('/labels/static', methods=['GET'])
def get_static_labels():
    classes = [str(c) for c in getattr(static_label_encoder, 'classes_', [])]
    contains_numeric_only = any(c.isdigit() for c in classes)
    contains_zero_or_one = any(c in ['0', '1'] for c in classes)
    return jsonify({
        "count": len(classes),
        "classes": classes,
        "contains_numeric_only_labels": contains_numeric_only,
        "contains_0_or_1": contains_zero_or_one
    })

@app.route('/debug/phrase_index', methods=['GET'])
def debug_phrase_index():
    query = request.args.get('q', default='', type=str)
    norm_q = _normalize_text(query) if query else ''
    hit = phrase_video_index.get(norm_q) if norm_q else None
    sample = dict(list(phrase_video_index.items())[:10]) if phrase_video_index else {}
    return jsonify({
        "bases": PHRASE_BASE_DIRS,
        "query": query,
        "normalized": norm_q,
        "hit": hit,
        "count": len(phrase_video_index),
        "sample": sample
    })


@app.route('/static_image/<path:relpath>')
def serve_static_image(relpath):
    """Serve images from data/static safely."""
    relpath = relpath.replace("\\", "/")
    candidate = os.path.normpath(os.path.join(DATA_STATIC_DIR, relpath))
    try:
        common = os.path.commonpath([DATA_STATIC_DIR, candidate])
    except Exception:
        common = ''
    if common != DATA_STATIC_DIR:
        return jsonify({"error": "Invalid path"}), 400
    if not os.path.exists(candidate):
        return jsonify({"error": "Image not found", "relpath": relpath}), 404
    ext = os.path.splitext(candidate)[1].lower()
    mimetype = 'image/jpeg'
    if ext == '.png':
        mimetype = 'image/png'
    elif ext == '.webp':
        mimetype = 'image/webp'
    return send_file(candidate, mimetype=mimetype)

@app.route('/debug/static_preprocess_check', methods=['POST'])
def debug_static_preprocess_check():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided (field 'image')"}), 400
    file = request.files['image']
    try:
        # Ensure stream is at start for first read
        try:
            file.stream.seek(0)
        except Exception:
            pass
        tensor = preprocess_static_image(file, scaler_path=CHOSEN_STATIC_SCALER_PATH, device=DEVICE)
        expected = static_model.fc1.in_features if hasattr(static_model, 'fc1') else 126
        # Rewind stream for second read
        try:
            file.stream.seek(0)
        except Exception:
            pass
        raw = extract_static_raw_126(file)
        scaler_in = get_scaler_features_in(CHOSEN_STATIC_SCALER_PATH)
        return jsonify({
            "tensor_shape": list(tensor.shape),
            "expected_in_features": expected,
            "match": (tensor.dim() == 2 and tensor.shape[1] == expected),
            "raw_shape": list(raw.shape),
            "scaler_n_features_in": scaler_in,
            "chosen_scaler_path": CHOSEN_STATIC_SCALER_PATH
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/sanity/static', methods=['GET'])
def sanity_static():
    try:
        expected = static_model.fc1.in_features if hasattr(static_model, 'fc1') else 126
        # Create dummy input of correct size
        dummy = torch.zeros((1, expected), dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            out = static_model(dummy)
        return jsonify({
            "expected_in_features": expected,
            "chosen_scaler_path": CHOSEN_STATIC_SCALER_PATH,
            "chosen_scaler_n_features_in": CHOSEN_STATIC_SCALER_NF,
            "model_output_shape": list(out.shape),
            "ok": True
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False)


