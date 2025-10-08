import os
import cv2
import numpy as np
import torch
import joblib

# Mediapipe modules are imported lazily inside functions to reduce import time for Flask


def _extract_static_hand_keypoints_from_bgr(image_bgr):
    """
    Extract 126 features (2 hands × 21 landmarks × xyz) from a single BGR image.
    If a hand or landmarks are missing, zeros are used for that hand.
    Returns a Python list of length 126.
    """
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


def preprocess_static_image(file_storage, scaler_path="models/static_scaler.pkl", device=None):
    """
    Convert an uploaded image (werkzeug FileStorage) into a normalized torch tensor
    of shape (1, 126) suitable for the StaticModel.

    - Extracts 2-hand keypoints using Mediapipe Hands (126 features)
    - Loads StandardScaler from scaler_path (joblib)
    - Returns torch.float32 tensor on the specified device
    """
    # Persist temporarily then read with OpenCV
    tmp_path = "__tmp_static.jpg"
    file_storage.save(tmp_path)
    img = cv2.imread(tmp_path)
    os.remove(tmp_path)

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
            raise FileNotFoundError(
                f"Static scaler not found at {scaler_path} or fallback {fallback_path}")

    features = static_scaler.transform(features)

    tensor = torch.tensor(features, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor  # (1, 126)


def extract_static_raw_126(file_storage):
    """
    Return raw unscaled static features as np.ndarray with shape (1, 126).
    """
    tmp_path = "__tmp_static_raw.jpg"
    file_storage.save(tmp_path)
    img = cv2.imread(tmp_path)
    os.remove(tmp_path)

    keypoints_126 = _extract_static_hand_keypoints_from_bgr(img)
    if len(keypoints_126) < 126:
        keypoints_126 = keypoints_126 + [0.0] * (126 - len(keypoints_126))
    elif len(keypoints_126) > 126:
        keypoints_126 = keypoints_126[:126]
    return np.array(keypoints_126, dtype=np.float32).reshape(1, -1)


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
    """
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    pose_model = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(temp_video_path)
    frames = []

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_model.process(frame_rgb)

        if results.pose_landmarks:
            kp = []
            for lm in results.pose_landmarks.landmark:
                kp.extend([lm.x, lm.y, lm.z])
            # 33 landmarks × 3 = 99
            if len(kp) > 99:
                kp = kp[:99]
            elif len(kp) < 99:
                kp.extend([0.0] * (99 - len(kp)))
            frames.append(kp)
        else:
            frames.append([0.0] * 99)

    cap.release()
    pose_model.close()

    while len(frames) < max_frames:
        frames.append([0.0] * 99)

    return np.asarray(frames, dtype=np.float32)


def preprocess_dynamic_video(file_storage, max_frames=30, device=None):
    """
    Convert an uploaded video (werkzeug FileStorage) into a torch tensor
    of shape (1, max_frames, 99) suitable for the DynamicLSTM.

    - Extracts 33 pose landmarks per frame (99 features)
    - Per-sequence min-max normalization for x and y separately, as in training
    - Pads to max_frames
    - Returns torch.float32 tensor on the specified device
    """
    tmp_path = "__tmp_video.mp4"
    file_storage.save(tmp_path)

    seq = _extract_dynamic_pose_keypoints_sequence_from_video(tmp_path, max_frames=max_frames)

    # Per-sequence normalization like training
    # Normalize x (cols 0::3) and y (cols 1::3); keep z unchanged (or scale if desired)
    x_cols = np.arange(0, 99, 3)
    y_cols = np.arange(1, 99, 3)

    if seq.size > 0:
        # x
        x_min = np.min(seq[:, x_cols])
        x_max = np.max(seq[:, x_cols])
        if x_max - x_min > 1e-6:
            seq[:, x_cols] = (seq[:, x_cols] - x_min) / (x_max - x_min)
        # y
        y_min = np.min(seq[:, y_cols])
        y_max = np.max(seq[:, y_cols])
        if y_max - y_min > 1e-6:
            seq[:, y_cols] = (seq[:, y_cols] - y_min) / (y_max - y_min)

    os.remove(tmp_path)

    tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1, T, 99)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


