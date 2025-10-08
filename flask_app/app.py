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
from flask import Flask, request, jsonify, render_template

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

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_FRAMES = 30

# Paths
STATIC_MODEL_PATH = "models/static_model.pth"
STATIC_LABEL_ENCODER_PATH = "models/static_label_encoder.pkl"
STATIC_SCALER_PATH = "models/static_scaler.pkl"
STATIC_HUMAN_LABEL_ENCODER_PATH = "data/processed/static_label_encoder.pkl"  # optional, preferred if compatible
STATIC_SCALER_FALLBACK_PATH = "data/processed/static_scaler.pkl"

DYNAMIC_MODEL_PATH = "models/dynamic_augmented_model.pth"
DYNAMIC_LABEL_ENCODER_PATH = "models/dynamic_label_encoder.pkl"

# -----------------------------
# LOAD MODELS
# -----------------------------
with open(STATIC_LABEL_ENCODER_PATH, "rb") as f:
    static_label_encoder = joblib.load(f)

# Try load human-readable encoder (folder names) from processed data if available
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

with open(DYNAMIC_LABEL_ENCODER_PATH, "rb") as f:
    dynamic_label_encoder = joblib.load(f)

num_classes_dynamic = len(dynamic_label_encoder.classes_)
dynamic_model = DynamicLSTM(input_size=99, hidden_size=128, num_layers=2, num_classes=num_classes_dynamic).to(DEVICE)
dynamic_model.load_state_dict(torch.load(DYNAMIC_MODEL_PATH, map_location=DEVICE))
dynamic_model.eval()

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
            return jsonify({
                "error": f"Static tensor must be 2D (batch,input_dim), got shape {tuple(tensor.shape)}"
            }), 500
        if tensor.shape[1] != expected_in_features:
            return jsonify({
                "error": f"Static features mismatch: got {tensor.shape[1]}, expected {expected_in_features}.",
                "chosen_scaler_path": CHOSEN_STATIC_SCALER_PATH,
                "chosen_scaler_n_features_in": CHOSEN_STATIC_SCALER_NF
            }), 500
        with torch.no_grad():
            output = static_model(tensor)
            pred_idx = torch.argmax(F.softmax(output, dim=1), dim=1).item()
            # Always use the encoder used during training to ensure exact mapping
            pred_label = static_label_encoder.inverse_transform([pred_idx])[0]
        return jsonify({"prediction": str(pred_label)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_dynamic', methods=['POST'])
def predict_dynamic():
    # Expect form field name 'video'
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided (field 'video')"}), 400
    file = request.files['video']
    try:
        tensor = preprocess_dynamic_video(file, max_frames=MAX_FRAMES, device=DEVICE)
        with torch.no_grad():
            output = dynamic_model(tensor)
            pred_idx = torch.argmax(F.softmax(output, dim=1), dim=1).item()
            pred_label = dynamic_label_encoder.inverse_transform([pred_idx])[0]
        return jsonify({"prediction": str(pred_label)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
                return jsonify({
                    "error": f"Static features mismatch: got {tuple(tensor.shape)}, expected (1,{expected_in_features})."
                }), 500
            with torch.no_grad():
                output = static_model(tensor)
                pred_idx = torch.argmax(F.softmax(output, dim=1), dim=1).item()
                pred_label = static_label_encoder.inverse_transform([pred_idx])[0]
            return jsonify({"prediction": str(pred_label), "type": "image"})

        # Default to video if ambiguous or explicitly video
        tensor = preprocess_dynamic_video(file, max_frames=MAX_FRAMES, device=DEVICE)
        with torch.no_grad():
            output = dynamic_model(tensor)
            pred_idx = torch.argmax(F.softmax(output, dim=1), dim=1).item()
            pred_label = dynamic_label_encoder.inverse_transform([pred_idx])[0]
        return jsonify({"prediction": str(pred_label), "type": "video"})
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


