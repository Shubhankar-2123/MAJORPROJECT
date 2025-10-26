import os
import cv2
import numpy as np
import torch
import joblib
import tempfile
import logging
import json

# Mediapipe modules are imported lazily inside functions to reduce import time for Flask
logger = logging.getLogger(__name__)

class PreprocessError(Exception):
    def __init__(self, code: str, message: str, details: dict = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def to_dict(self):
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


def _log_event(event: str, level: str = "info", **data):
    payload = {"event": event, **data}
    msg = json.dumps(payload, ensure_ascii=False)
    if level == "debug":
        logger.debug(msg)
    elif level == "warning":
        logger.warning(msg)
    elif level == "error":
        logger.error(msg)
    else:
        logger.info(msg)


def _extract_static_hand_keypoints_from_bgr(image_bgr):
    """
    Extract 126 features (2 hands × 21 landmarks × xyz) from a single BGR image.
    If a hand or landmarks are missing, zeros are used for that hand.
    Returns a Python list of length 126.
    """
    if image_bgr is None:
        _log_event("static_image.decode_failed", level="error")
        raise PreprocessError("IMAGE_DECODE_FAILED", "Unable to decode image.")
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        hands_model = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
        )

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = hands_model.process(image_rgb)

        # Initialize two hands with zeros
        hand0 = [0.0] * (21 * 3)
        hand1 = [0.0] * (21 * 3)

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                if idx == 0:
                    hand0 = coords
                elif idx == 1:
                    hand1 = coords

        hands_model.close()
        return hand0 + hand1  # 126 features
    except Exception as e:
        _log_event("mediapipe.hands.failure", level="error", error=str(e))
        raise PreprocessError("MEDIAPIPE_HANDS_FAILED", "Failed to run hand landmark extraction.", {"exception": str(e)})


def preprocess_static_image(file_storage, scaler_path="models/static_scaler.pkl", device=None):
    """
    Convert an uploaded image (werkzeug FileStorage) into a normalized torch tensor
    of shape (1, 126) suitable for the StaticModel.

    - Extracts 2-hand keypoints using Mediapipe Hands (126 features)
    - Loads StandardScaler from scaler_path (joblib)
    - Returns torch.float32 tensor on the specified device
    """
    _log_event("preprocess.static.start", filename=getattr(file_storage, "filename", None))
    # Persist temporarily then read with OpenCV
    tmp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()
    file_storage.save(tmp_path)
    try:
        img = cv2.imread(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    if img is None:
        _log_event("preprocess.static.image_read_failed", level="error")
        raise PreprocessError("IMAGE_DECODE_FAILED", "Invalid image upload: cannot decode.")

    keypoints_126 = _extract_static_hand_keypoints_from_bgr(img)
    # Enforce correct feature length
    if len(keypoints_126) < 126:
        keypoints_126 = keypoints_126 + [0.0] * (126 - len(keypoints_126))
    elif len(keypoints_126) > 126:
        keypoints_126 = keypoints_126[:126]

    features = np.array(keypoints_126, dtype=np.float32).reshape(1, -1)

    # Load scaler used in training for static model (fallback to data/processed if needed)
    static_scaler = None
    try:
        static_scaler = joblib.load(scaler_path)
    except Exception:
        fallback_path = os.path.join("data", "processed", "static_scaler.pkl")
        if os.path.exists(fallback_path):
            static_scaler = joblib.load(fallback_path)
        else:
            _log_event("preprocess.static.scaler_missing", level="error", scaler_path=scaler_path)
            raise PreprocessError("SCALER_NOT_FOUND", f"Static scaler not found at {scaler_path}", {"fallback": fallback_path})

    features = static_scaler.transform(features)

    tensor = torch.tensor(features, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    _log_event("preprocess.static.success", shape=list(tensor.shape))
    return tensor  # (1, 126)


def extract_static_raw_126(file_storage):
    """
    Return raw unscaled static features as np.ndarray with shape (1, 126).
    """
    _log_event("preprocess.static_raw.start", filename=getattr(file_storage, "filename", None))
    tmp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()
    file_storage.save(tmp_path)
    try:
        img = cv2.imread(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    if img is None:
        _log_event("preprocess.static_raw.image_read_failed", level="error")
        raise PreprocessError("IMAGE_DECODE_FAILED", "Invalid image upload: cannot decode.")

    keypoints_126 = _extract_static_hand_keypoints_from_bgr(img)
    if len(keypoints_126) < 126:
        keypoints_126 = keypoints_126 + [0.0] * (126 - len(keypoints_126))
    elif len(keypoints_126) > 126:
        keypoints_126 = keypoints_126[:126]
    arr = np.array(keypoints_126, dtype=np.float32).reshape(1, -1)
    _log_event("preprocess.static_raw.success", shape=list(arr.shape))
    return arr


def get_scaler_features_in(scaler_path="models/static_scaler.pkl"):
    """
    Return the expected feature count (n_features_in_) from the static scaler.
    Falls back to data/processed/static_scaler.pkl if needed.
    """
    scaler = None
    try:
        scaler = joblib.load(scaler_path)
    except Exception:
        fallback_path = os.path.join("data", "processed", "static_scaler.pkl")
        if os.path.exists(fallback_path):
            scaler = joblib.load(fallback_path)
        else:
            return None
    return getattr(scaler, 'n_features_in_', None)


def _extract_dynamic_pose_keypoints_sequence_from_video(temp_video_path, max_frames=30):
    """
    Extract a sequence of up to max_frames of 33 pose landmarks × xyz = 99 features per frame.
    Pads with zeros if frames are missing. Returns np.ndarray of shape (max_frames, 99).
    MATCHES TRAINING PREPROCESSING EXACTLY.
    """
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        # Match training MediaPipe configuration exactly
        pose_model = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    except Exception as e:
        _log_event("mediapipe.pose.failure", level="error", error=str(e))
        raise PreprocessError("MEDIAPIPE_POSE_FAILED", "Failed to initialize pose model.", {"exception": str(e)})

    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        _log_event("video.open_failed", level="error", path=temp_video_path)
        pose_model.close()
        raise PreprocessError("VIDEO_OPEN_FAILED", "Unable to open uploaded video.", {"path": temp_video_path})

    keypoints_seq = []

    try:
        while len(keypoints_seq) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_model.process(frame_rgb)

            if results.pose_landmarks:
                kp = []
                for lm in results.pose_landmarks.landmark:
                    kp.extend([lm.x, lm.y, lm.z])
                keypoints_seq.append(kp)
            else:
                keypoints_seq.append([0]*99)  # Match training: use integers, not floats
    finally:
        cap.release()
        pose_model.close()

    if len(keypoints_seq) == 0:
        _log_event("video.no_frames", level="error")
        raise PreprocessError("VIDEO_DECODE_FAILED", "No readable frames in uploaded video.")

    while len(keypoints_seq) < max_frames:
        keypoints_seq.append([0]*99)  # Match training: use integers, not floats

    _log_event("video.keypoints_extracted", frames=len(keypoints_seq), max_frames=max_frames)
    return np.array(keypoints_seq)


def normalize_keypoints_training_style(seq):
    """
    Normalize keypoints using the EXACT same method as training.
    This matches the normalize_keypoints function in train_dynamic_new.py
    """
    seq = np.array(seq)
    seq[:, 0::3] = (seq[:, 0::3] - np.min(seq[:, 0::3])) / (np.max(seq[:, 0::3]) - np.min(seq[:, 0::3]) + 1e-6)
    seq[:, 1::3] = (seq[:, 1::3] - np.min(seq[:, 1::3])) / (np.max(seq[:, 1::3]) - np.min(seq[:, 1::3]) + 1e-6)
    return seq


def preprocess_dynamic_video(file_storage, max_frames=30, device=None):
    """
    Convert an uploaded video (werkzeug FileStorage) into a torch tensor
    of shape (1, max_frames, 99) suitable for the DynamicLSTM.

    - Extracts 33 pose landmarks per frame (99 features) using EXACT training config
    - Uses EXACT same normalization as training (normalize_keypoints function)
    - Pads to max_frames
    - Returns torch.float32 tensor on the specified device
    
    MATCHES TRAINING PREPROCESSING EXACTLY.
    """
    _log_event("preprocess.dynamic.start", filename=getattr(file_storage, "filename", None), max_frames=max_frames)
    tmp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()
    file_storage.save(tmp_path)

    try:
        seq = _extract_dynamic_pose_keypoints_sequence_from_video(tmp_path, max_frames=max_frames)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    
    # Use the EXACT same normalization as training
    seq = normalize_keypoints_training_style(seq)

    tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1, T, 99)
    if device is not None:
        tensor = tensor.to(device)
    _log_event("preprocess.dynamic.success", shape=list(tensor.shape))
    return tensor


