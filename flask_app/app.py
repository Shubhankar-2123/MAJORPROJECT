# import sys
# import os

# # Add scripts folder to sys.path for model imports
# sys.path.append(os.path.join(os.path.dirname(__file__), "../scripts"))

# import cv2
# import numpy as np
# import torch
# import pickle
# import joblib
# import mediapipe as mp
# from flask import Flask, request, jsonify, render_template
# from train_static_model import StaticModel
# from train_dynamic_new import DynamicLSTM

# # -----------------------------
# # CONFIG
# # -----------------------------
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# STATIC_MODEL_PATH = "models/static_model.pth"
# STATIC_LE_PATH = "models/static_label_encoder.pkl"
# STATIC_SCALER_PATH= "models/static_scaler.pkl"
# DYNAMIC_MODEL_PATH= "models/dynamic_augmented_model.pth"
# DYNAMIC_LE_PATH = "models/dynamic_label_encoder.pkl"

# MAX_FRAMES = 30

# # -----------------------------
# # LOAD MODELS
# # -----------------------------
# static_le = joblib.load(STATIC_LE_PATH)
# print("Static LabelEncoder classes:", static_le.classes_)

# static_scaler = joblib.load(STATIC_SCALER_PATH)

# input_dim_static = 126
# num_classes_static = len(static_le.classes_)

# static_model = StaticModel(input_dim_static, num_classes_static).to(DEVICE)
# static_model.load_state_dict(torch.load(STATIC_MODEL_PATH, map_location=DEVICE))
# static_model.eval()

# with open(DYNAMIC_LE_PATH, "rb") as f:
#     dynamic_le = pickle.load(f)
# print("Dynamic LabelEncoder classes:", dynamic_le.classes_)

# num_classes_dynamic = len(dynamic_le.classes_)

# dynamic_model = DynamicLSTM(input_size=99, hidden_size=128, num_layers=2, num_classes=num_classes_dynamic).to(DEVICE)
# dynamic_model.load_state_dict(torch.load(DYNAMIC_MODEL_PATH, map_location=DEVICE))
# dynamic_model.eval()

# # -----------------------------
# # FLASK APP
# # -----------------------------
# app = Flask(__name__)

# @app.route("/")
# def index():
#     return render_template("index.html")

# # -----------------------------
# # STATIC PREDICTION (Image)
# # -----------------------------
# def extract_static_keypoints(image_path):
#     mp_pose = mp.solutions.pose
#     pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
#     img = cv2.imread(image_path)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = pose.process(img_rgb)
#     if results.pose_landmarks:
#         kp = []
#         for lm in results.pose_landmarks.landmark:
#             kp.extend([lm.x, lm.y, lm.z])
#         if len(kp) < 126:
#             kp.extend([0] * (126 - len(kp)))
#         else:
#             kp = kp[:126]
#         pose.close()
#         return np.array(kp)
#     pose.close()
#     return None

# @app.route("/predict_static", methods=["POST"])
# def predict_static():
#     if "image" not in request.files:
#         return jsonify({"error": "No image uploaded"}), 400
#     img_file = request.files["image"]
#     temp_path = "temp_image.jpg"
#     img_file.save(temp_path)

#     keypoints = extract_static_keypoints(temp_path)
#     os.remove(temp_path)

#     if keypoints is None:
#         return jsonify({"error": "No hand detected"}), 400

#     keypoints = keypoints.reshape(1, -1)
#     keypoints = static_scaler.transform(keypoints)

#     keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32).to(DEVICE)

#     with torch.no_grad():
#         outputs = static_model(keypoints_tensor)
#         pred_idx = torch.argmax(outputs, dim=1).item()

#     pred_label = static_le.inverse_transform([pred_idx])[0]

#     print(f"Static prediction index: {pred_idx}, label: {pred_label}")

#     return jsonify({"prediction": str(pred_label)})

# # -----------------------------
# # DYNAMIC PREDICTION (Video)
# # -----------------------------
# def preprocess_video(video_path):
#     mp_pose = mp.solutions.pose
#     pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
#     cap = cv2.VideoCapture(video_path)
#     frames = []

#     while len(frames) < MAX_FRAMES:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(frame_rgb)
#         if results.pose_landmarks:
#             kp = []
#             for lm in results.pose_landmarks.landmark:
#                 kp.extend([lm.x, lm.y, lm.z])
#             frames.append(kp)
#         else:
#             frames.append([0]*99)
#     cap.release()
#     pose.close()
#     while len(frames) < MAX_FRAMES:
#         frames.append([0]*99)

#     return np.array(frames, dtype=np.float32)

# @app.route("/predict_dynamic", methods=["POST"])
# def predict_dynamic():
#     if "video" not in request.files:
#         return jsonify({"error": "No video uploaded"}), 400
#     video_file = request.files["video"]
#     temp_path = "temp_video.mp4"
#     video_file.save(temp_path)

#     frames = preprocess_video(temp_path)
#     os.remove(temp_path)

#     frames_tensor = torch.tensor(frames).unsqueeze(0).to(DEVICE)

#     with torch.no_grad():
#         outputs = dynamic_model(frames_tensor)
#         pred_idx = torch.argmax(outputs, dim=1).item()

#     pred_label = dynamic_le.inverse_transform([pred_idx])[0]

#     print(f"Dynamic prediction index: {pred_idx}, label: {pred_label}")

#     return jsonify({"prediction": str(pred_label)})

# if __name__ == "__main__":
#     app.run(debug=True)


import os
import torch
import torch.nn.functional as F
import joblib
import json
from rapidfuzz import process
from flask import Flask, request, jsonify, render_template, send_file
from urllib.parse import quote
import re

# Ensure imports work regardless of run directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils")))

from train_static_model import StaticModel
from train_dynamic_new import DynamicLSTM
from preprocessing import (
    preprocess_static_image,
    preprocess_dynamic_video,
    extract_static_raw_126,
    get_scaler_features_in,
)
# from text_to_sign_service import TextToSignService  # Deprecated: no JSON-based lookup

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_FRAMES = 30

# Base paths (use only models from models_main)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_MAIN_DIR = os.path.join(PROJECT_ROOT, "models")
STATIC_MAIN_DIR = os.path.join(MODELS_MAIN_DIR, "static images")
WORDS_MAIN_DIR = os.path.join(MODELS_MAIN_DIR, "words")
SENTENCES_MAIN_DIR = os.path.join(MODELS_MAIN_DIR, "sentences")

# Confidence thresholds
DYNAMIC_DEFAULT_THRESHOLD = float(os.getenv("DYNAMIC_DEFAULT_THRESHOLD", 0.60))
WORD_CONF_THRESHOLD = float(os.getenv("WORD_CONF_THRESHOLD", 0.55))  # 55%
SENT_CONF_THRESHOLD = float(os.getenv("SENT_CONF_THRESHOLD", 0.75))  # 75%

# -----------------------------
# LOAD MODELS (Static + Dynamic registries)
# -----------------------------
# Static (A–Z, 1–9)
STATIC_MODEL_PATH = os.path.join(STATIC_MAIN_DIR, "static_model.pth")
STATIC_LABEL_ENCODER_PATH = os.path.join(STATIC_MAIN_DIR, "static_label_encoder.pkl")
STATIC_SCALER_PATH = os.path.join(STATIC_MAIN_DIR, "static_scaler.pkl")

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
from typing import List, Optional, Dict, Any

@dataclass
class DynamicEntry:
    name: str
    kind: str  # "word" or "sentence"
    model: Any
    label_encoder: Any
    threshold: float

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
                mdl.load_state_dict(state, strict=True)
                mdl.eval()
                name = f"{kind}#{idx}"
                entries.append(DynamicEntry(name=name, kind=kind, model=mdl, label_encoder=le, threshold=DYNAMIC_DEFAULT_THRESHOLD))
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

"""
Text-to-Sign helpers: index available media by scanning data/ folders directly.
"""

def _normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _iter_existing_dirs(paths):
    for p in paths:
        if os.path.isdir(p):
            yield p

def _build_phrase_index(base_dirs):
    phrase_to_path = {}
    for base_dir in _iter_existing_dirs(base_dirs):
        for root, _dirs, files in os.walk(base_dir):
            for name in files:
                if not name.lower().endswith((".mp4", ".mov", ".mkv", ".avi", ".webm")):
                    continue
                abs_p = os.path.join(root, name)
                # Candidates from folder and stem
                folder_phrase = os.path.basename(root)
                stem = os.path.splitext(name)[0]
                for candidate in {folder_phrase, stem, stem.replace("_", " "), folder_phrase.replace("_", " ")}:
                    norm = _normalize_text(candidate)
                    if norm and norm not in phrase_to_path:
                        phrase_to_path[norm] = abs_p
    return phrase_to_path

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
DATA_STATIC_DIR = os.path.join(DATA_DIR, "static")
PHRASE_BASE_DIRS = [
    os.path.join(DATA_DIR, "dynamic"),
    *[os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR) if d.lower().startswith("dynamic") and os.path.isdir(os.path.join(DATA_DIR, d))],
]
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

# -----------------------------
# FLASK APP
# -----------------------------
app = Flask(__name__)

## Preprocessing is imported from utils/preprocessing.py


# -----------------------------
# ROUTES
# -----------------------------
@app.route('/')
def index():
    return render_template("index.html")

# -----------------------------
# TEXT-TO-SIGN FUNCTIONALITY
# -----------------------------

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
        return jsonify({"success": True, "phrase": {"url": f"/dyn_video/{enc}", "text": text}, "words": [], "missing": []})

    # 2) Per-word lookup
    tokens = [t for t in re.split(r"\s+", text) if t]
    results = []
    missing = []

    # Build a quick word->video index using folder names that look like single words
    def _build_word_index():
        idx = {}
        for base in _iter_existing_dirs(PHRASE_BASE_DIRS):
            for root, dirs, files in os.walk(base):
                word = os.path.basename(root)
                if ' ' in word:
                    continue  # likely a sentence
                for name in files:
                    if name.lower().endswith((".mp4", ".mov", ".mkv", ".avi", ".webm")):
                        key = _normalize_text(word)
                        idx.setdefault(key, os.path.join(root, name))
                        break
        return idx

    word_video_index = _build_word_index()


    for tok in tokens:
        norm_tok = _normalize_text(tok)
        # Single character image case first
        if len(norm_tok) == 1 and norm_tok.isalnum():
            img_abs = _find_letter_image(norm_tok)
            if img_abs and os.path.exists(img_abs):
                # Serve from data/static via a dedicated image route
                rel = os.path.relpath(img_abs, DATA_STATIC_DIR)
                enc = quote(rel.replace("\\", "/"))
                url = f"/static_image/{enc}"
                results.append({"word": tok, "url": url})
                continue
        # Word video lookup
        vid_abs = word_video_index.get(norm_tok)
        if vid_abs and os.path.exists(vid_abs):
            # Serve via /dyn_video after converting to relpath under the closest base
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
            results.append({"word": tok, "url": f"/dyn_video/{enc}"})
        else:
            missing.append({"word": tok, "message": "media not found"})

    return jsonify({"success": True, "phrase": None, "words": results, "missing": missing, "original_text": text})

@app.route('/video/<path:filename>')
def serve_video(filename):
    """Serve video files from the app's static/videos directory using absolute paths."""
    base_dir = os.path.dirname(__file__)
    videos_dir = os.path.join(base_dir, "static", "videos")
    abs_path = os.path.join(videos_dir, filename)
    if os.path.exists(abs_path):
        ext = os.path.splitext(abs_path)[1].lower()
        mimetype = 'application/octet-stream'
        if ext in ['.mp4', '.m4v']:
            mimetype = 'video/mp4'
        elif ext in ['.webm']:
            mimetype = 'video/webm'
        elif ext in ['.mov']:
            mimetype = 'video/quicktime'
        elif ext in ['.mkv']:
            mimetype = 'video/x-matroska'
        return send_file(abs_path, mimetype=mimetype)
    return jsonify({"error": "Video not found", "filename": filename}), 404


@app.route('/frames_video/<path:relpath>')
def serve_frames_video(relpath):
    """Serve videos from data/Frames_Word_Level by relative path, ignoring images.

    Security: ensures request stays within base directory.
    """
    # Base directory for frames/word-level videos
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "Frames_Word_Level"))
    # Normalize separators in relpath
    relpath = relpath.replace("\\", "/")
    # Build absolute candidate
    candidate = os.path.normpath(os.path.join(base_dir, relpath))
    # Ensure path traversal protection
    try:
        common = os.path.commonpath([base_dir, candidate])
    except Exception:
        common = ''
    if common != base_dir:
        return jsonify({"error": "Invalid path"}), 400
    if not os.path.exists(candidate):
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
            return send_file(candidate)
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
def predict_static():
    # Expect form field name 'image'
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided (field 'image')"}), 400
    file = request.files['image']
    try:
        tensor = preprocess_static_image(file, scaler_path=CHOSEN_STATIC_SCALER_PATH, device=DEVICE)
        # Runtime guards to avoid shape mismatch
        expected_in_features = static_model.fc1.in_features if hasattr(static_model, 'fc1') else 126
        if tensor.dim() != 2:
            return jsonify({"error": "unable to recognize"}), 200
        if tensor.shape[1] != expected_in_features:
            return jsonify({"error": "unable to recognize"}), 200
        with torch.no_grad():
            logits = static_model(tensor)
            probs = F.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            conf_v = float(conf.item())
            pred_idx_v = int(pred_idx.item())
        if conf_v < 0.80:
            return jsonify({"prediction": "unable to recognize"})
        pred_label = static_label_encoder.inverse_transform([pred_idx_v])[0]
        return jsonify({"prediction": str(pred_label)})
    except Exception:
        return jsonify({"prediction": "unable to recognize"})

@app.route('/predict_dynamic', methods=['POST'])
def predict_dynamic():
    # Expect form field name 'video'; optional 'kind' in form or query to hint (word|sentence)
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided (field 'video')"}), 400
    file = request.files['video']
    kind_hint = request.form.get('kind') or request.args.get('kind')  # 'word' | 'sentence' | None
    try:
        if not dynamic_registry:
            return jsonify({"prediction": "unable to recognize"})
        candidates = [e for e in dynamic_registry if (not kind_hint or e.kind == kind_hint)] or dynamic_registry
        tensor = preprocess_dynamic_video(file, max_frames=MAX_FRAMES, device=DEVICE)
        best = {"label": None, "confidence": 0.0, "model": None, "kind": None}
        with torch.no_grad():
            for entry in candidates:
                out = entry.model(tensor)
                probs = F.softmax(out, dim=1).squeeze(0)
                conf, idx = torch.max(probs, dim=0)
                conf_v = float(conf.item())
                idx_v = int(idx.item())
                if conf_v > best["confidence"]:
                    label = entry.label_encoder.inverse_transform([idx_v])[0]
                    best = {"label": str(label), "confidence": conf_v, "model": entry.name, "kind": entry.kind, "threshold": entry.threshold}
        # Threshold per kind (words 0.55, sentences 0.75)
        thr = best.get("threshold") or (WORD_CONF_THRESHOLD if best.get("kind") == "word" else SENT_CONF_THRESHOLD)
        if best["confidence"] < float(thr):
            return jsonify({"prediction": "unable to recognize"})
        return jsonify({"prediction": best["label"]})
    except Exception:
        return jsonify({"prediction": "unable to recognize"})


@app.route('/predict', methods=['POST'])
def predict_unified():
    # Accepts a single file field: 'file' (preferred) or fallback to 'image'/'video'
    file = None
    for key in ['file', 'image', 'video']:
        if key in request.files:
            file = request.files[key]
            break
    if file is None:
        return jsonify({"error": "No file provided. Use field 'file' (image/video)."}), 400

    filename = getattr(file, 'filename', '') or ''
    mimetype = getattr(file, 'mimetype', '') or ''
    lower_name = filename.lower()
    is_image = any(lower_name.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']) or mimetype.startswith('image/')
    is_video = any(lower_name.endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']) or mimetype.startswith('video/')

    try:
        if is_image and not is_video:
            tensor = preprocess_static_image(file, scaler_path=CHOSEN_STATIC_SCALER_PATH, device=DEVICE)
            expected_in_features = static_model.fc1.in_features if hasattr(static_model, 'fc1') else 126
            if tensor.dim() != 2 or tensor.shape[1] != expected_in_features:
                return jsonify({"prediction": "unable to recognize"})
            with torch.no_grad():
                logits = static_model(tensor)
                probs = F.softmax(logits, dim=1)
                conf, pred_idx = torch.max(probs, dim=1)
                conf_v = float(conf.item())
            if conf_v < 0.80:
                return jsonify({"prediction": "unable to recognize"})
            pred_label = static_label_encoder.inverse_transform([int(pred_idx.item())])[0]
            return jsonify({"prediction": str(pred_label)})

        # Default to video if ambiguous or explicitly video
        if not dynamic_registry:
            return jsonify({"prediction": "unable to recognize"})
        kind_hint = request.form.get('kind') or request.args.get('kind')
        candidates = [e for e in dynamic_registry if (not kind_hint or e.kind == kind_hint)] or dynamic_registry
        tensor = preprocess_dynamic_video(file, max_frames=MAX_FRAMES, device=DEVICE)
        best = {"label": None, "confidence": 0.0, "threshold": None, "kind": None}
        with torch.no_grad():
            for entry in candidates:
                out = entry.model(tensor)
                probs = F.softmax(out, dim=1).squeeze(0)
                conf, idx = torch.max(probs, dim=0)
                conf_v = float(conf.item())
                if conf_v > best["confidence"]:
                    label = entry.label_encoder.inverse_transform([int(idx.item())])[0]
                    best = {"label": str(label), "confidence": conf_v, "threshold": entry.threshold, "kind": entry.kind}
        thr = best.get("threshold") or (WORD_CONF_THRESHOLD if best.get("kind") == "word" else SENT_CONF_THRESHOLD)
        if best["confidence"] < float(thr):
            return jsonify({"prediction": "unable to recognize"})
        return jsonify({"prediction": best["label"]})
    except Exception:
        return jsonify({"prediction": "unable to recognize"})


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


