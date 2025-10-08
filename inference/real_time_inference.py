# import os
# import pickle
# import torch
# import cv2
# import numpy as np
# import mediapipe as mp

# # ------------------------
# # Device
# # ------------------------
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # ------------------------
# # Paths to models
# # ------------------------
# STATIC_MODEL_PKL_PATH = "models/static_sign_model.pkl"
# STATIC_MODEL_STATE_PATH = "models/static_model.pth"
# STATIC_LABEL_ENCODER_PATH = "models/static_label_encoder.pkl"
# STATIC_SCALER_PATH = "models/static_scaler.pkl"

# DYNAMIC_MODEL_PATH = "models/dynamic_augmented_model.pth"
# # Dynamic model input parameters
# DYNAMIC_INPUT_SIZE = 126
# DYNAMIC_HIDDEN_SIZE = 128
# DYNAMIC_NUM_LAYERS = 2

# # ------------------------
# # Load Static Model
# # ------------------------
# static_le = None
# try:
#     with open(STATIC_LABEL_ENCODER_PATH, "rb") as f:
#         static_le = pickle.load(f)
# except Exception:
#     static_le = None

# # Define static model architecture (from training script)
# class StaticModel(torch.nn.Module):
#     def __init__(self, input_dim, num_classes):
#         super().__init__()
#         self.fc1 = torch.nn.Linear(input_dim, 256)
#         self.drop1 = torch.nn.Dropout(0.3)
#         self.fc2 = torch.nn.Linear(256, 128)
#         self.drop2 = torch.nn.Dropout(0.3)
#         self.fc3 = torch.nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.drop1(x)
#         x = torch.relu(self.fc2(x))
#         x = self.drop2(x)
#         return self.fc3(x)

# # Try loading state_dict first; fallback to pickle if needed
# static_model = None
# state = None
# num_static_classes = None
# if os.path.exists(STATIC_MODEL_STATE_PATH):
#     try:
#         state = torch.load(STATIC_MODEL_STATE_PATH, map_location=device)
#         if isinstance(state, dict):
#             # infer classes from final layer
#             for key in ["fc3.weight", "module.fc3.weight"]:
#                 if key in state:
#                     num_static_classes = state[key].shape[0]
#                     break
#         if num_static_classes is None:
#             num_static_classes = len(getattr(static_le, "classes_", [])) or 10
#         static_model = StaticModel(input_dim=DYNAMIC_INPUT_SIZE, num_classes=num_static_classes)
#         static_model.load_state_dict(state)
#     except Exception:
#         static_model = None

# if static_model is None and os.path.exists(STATIC_MODEL_PKL_PATH):
#     try:
#         with open(STATIC_MODEL_PKL_PATH, "rb") as f:
#             static_model = pickle.load(f)
#     except Exception:
#         static_model = None

# if static_model is None:
#     raise RuntimeError("Unable to load static model from state_dict or pickle.")

# static_model.to(device)
# static_model.eval()

# # Build static class names if encoder missing
# static_class_names = None
# if static_le is not None and hasattr(static_le, "classes_") and len(static_le.classes_) == num_static_classes:
#     static_class_names = [str(c) for c in static_le.classes_]
# else:
#     static_static_dir = "data/static"
#     if os.path.isdir(static_static_dir):
#         names = sorted([d for d in os.listdir(static_static_dir) if os.path.isdir(os.path.join(static_static_dir, d))])
#         if len(names) == num_static_classes:
#             static_class_names = names
#     if static_class_names is None:
#         static_class_names = [str(i) for i in range(num_static_classes or 0)]

# static_scaler = None
# if os.path.exists(STATIC_SCALER_PATH):
#     try:
#         with open(STATIC_SCALER_PATH, "rb") as f:
#             static_scaler = pickle.load(f)
#     except Exception:
#         static_scaler = None

# # ------------------------
# # Load Dynamic Model (support two variants)
# # ------------------------
# class KeypointEncoder(torch.nn.Module):
#     def __init__(self, input_size=DYNAMIC_INPUT_SIZE, hidden_size=256, embedding_dim=128):
#         super(KeypointEncoder, self).__init__()
#         self.fc1 = torch.nn.Linear(input_size, hidden_size)
#         self.fc2 = torch.nn.Linear(hidden_size, embedding_dim)
#         self.relu = torch.nn.ReLU()

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         return self.fc2(x)


# class DynamicSignModel(torch.nn.Module):
#     def __init__(self, input_size=DYNAMIC_INPUT_SIZE, hidden_size=DYNAMIC_HIDDEN_SIZE, num_layers=DYNAMIC_NUM_LAYERS, num_classes=10):
#         super(DynamicSignModel, self).__init__()
#         self.encoder = KeypointEncoder(input_size=input_size)
#         self.lstm = torch.nn.LSTM(128, hidden_size, num_layers, batch_first=True)
#         self.fc = torch.nn.Linear(hidden_size, num_classes)

#     def forward(self, x):
#         b, t, f = x.shape
#         x = x.view(b * t, f)
#         embeddings = self.encoder(x)
#         embeddings = embeddings.view(b, t, -1)
#         out, _ = self.lstm(embeddings)
#         out = out[:, -1, :]
#         return self.fc(out)


# class SimpleDynamicLSTM(torch.nn.Module):
#     def __init__(self, input_size=99, hidden_size=DYNAMIC_HIDDEN_SIZE, num_layers=DYNAMIC_NUM_LAYERS, num_classes=10):
#         super(SimpleDynamicLSTM, self).__init__()
#         self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = torch.nn.Linear(hidden_size, num_classes)

#     def forward(self, x):
#         out, _ = self.lstm(x)
#         out = out[:, -1, :]
#         return self.fc(out)


# # Inspect checkpoint to determine architecture, num_classes, and expected input
# state_dict = torch.load(DYNAMIC_MODEL_PATH, map_location=device)
# has_encoder = any(k.startswith("encoder.") for k in state_dict.keys())
# num_dynamic_classes = None
# expected_dynamic_input = None

# for k in ["fc.weight", "module.fc.weight"]:
#     if k in state_dict:
#         num_dynamic_classes = state_dict[k].shape[0]
#         break
# if num_dynamic_classes is None:
#     num_dynamic_classes = 10

# if "lstm.weight_ih_l0" in state_dict:
#     expected_dynamic_input = state_dict["lstm.weight_ih_l0"].shape[1]

# if has_encoder:
#     dynamic_model = DynamicSignModel(input_size=DYNAMIC_INPUT_SIZE, num_classes=num_dynamic_classes)
# else:
#     # fall back to simple LSTM with inferred input size
#     if expected_dynamic_input is None:
#         expected_dynamic_input = 99
#     dynamic_model = SimpleDynamicLSTM(input_size=expected_dynamic_input, num_classes=num_dynamic_classes)

# dynamic_model.load_state_dict(state_dict)
# dynamic_model.to(device)
# dynamic_model.eval()

# # Derive dynamic class names from folders if available
# dynamic_class_names = None
# dynamic_data_dir = "data/dynamic"
# if os.path.isdir(dynamic_data_dir):
#     dynamic_class_names = sorted([d for d in os.listdir(dynamic_data_dir) if os.path.isdir(os.path.join(dynamic_data_dir, d))])
#     if len(dynamic_class_names) != num_dynamic_classes:
#         dynamic_class_names = None

# # ------------------------
# # ------------------------
# # Mediapipe Hands and OpenCV Webcam
# # ------------------------
# mp_hands = mp.solutions.hands
# mp_pose = mp.solutions.pose
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
# pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     cap = cv2.VideoCapture(1)

# # Buffer for dynamic sequence
# SEQUENCE_LENGTH = 30
# dynamic_buffer = []

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # ------------------------
#     # Extract keypoints for static (hands-126) and dynamic (pose-99 or hands-126)
#     # ------------------------
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Static uses Hands: 126-dim
#     res_hands = hands.process(frame_rgb)
#     hand0 = [0.0] * 21 * 3
#     hand1 = [0.0] * 21 * 3
#     if res_hands.multi_hand_landmarks:
#         for idx, hand_landmarks in enumerate(res_hands.multi_hand_landmarks):
#             coords = []
#             for lm in hand_landmarks.landmark:
#                 coords.extend([lm.x, lm.y, lm.z])
#             if idx == 0:
#                 hand0 = coords
#             elif idx == 1:
#                 hand1 = coords
#     static_keypoints = np.array(hand0 + hand1, dtype=np.float32)

#     # Dynamic uses Pose 99 if model expects 99, else Hands 126
#     if isinstance(dynamic_model, SimpleDynamicLSTM) and expected_dynamic_input == 99:
#         res_pose = pose.process(frame_rgb)
#         if res_pose.pose_landmarks:
#             vals = []
#             for lm in res_pose.pose_landmarks.landmark:
#                 vals.extend([lm.x, lm.y, lm.z])
#             dynamic_keypoints = np.array(vals[:99], dtype=np.float32)
#             if dynamic_keypoints.shape[0] < 99:
#                 dynamic_keypoints = np.pad(dynamic_keypoints, (0, 99 - dynamic_keypoints.shape[0]))
#         else:
#             dynamic_keypoints = np.zeros(99, dtype=np.float32)
#     else:
#         dynamic_keypoints = static_keypoints.copy()  # 126-dim

#     # --- Static Prediction ---
#     x_static_np = static_keypoints.copy()
#     if static_scaler is not None:
#         try:
#             x_static_np = static_scaler.transform(x_static_np.reshape(1, -1)).reshape(-1)
#         except Exception:
#             pass
#     x_static = torch.tensor(x_static_np).unsqueeze(0).to(device)
#     with torch.no_grad():
#         out_static = static_model(x_static)
#         pred_static = torch.argmax(out_static, dim=1).item()
#         label_static = static_class_names[pred_static] if 0 <= pred_static < len(static_class_names) else str(int(pred_static))

#     # --- Dynamic Prediction ---
#     dynamic_buffer.append(dynamic_keypoints)
#     if len(dynamic_buffer) > SEQUENCE_LENGTH:
#         dynamic_buffer.pop(0)

#     label_dynamic = ""
#     if len(dynamic_buffer) == SEQUENCE_LENGTH:
#         x_dynamic = torch.tensor([dynamic_buffer], dtype=torch.float32).to(device)
#         with torch.no_grad():
#             out_dynamic = dynamic_model(x_dynamic)
#             pred_dynamic = torch.argmax(out_dynamic, dim=1).cpu().numpy()[0]
#             if dynamic_class_names:
#                 if 0 <= pred_dynamic < len(dynamic_class_names):
#                     label_dynamic = dynamic_class_names[pred_dynamic]
#             else:
#                 label_dynamic = str(int(pred_dynamic))

#     # --- Display ---
#     try:
#         cv2.putText(frame, f"Static Sign: {label_static}", (30, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         if label_dynamic:
#             cv2.putText(frame, f"Dynamic Sentence: {label_dynamic}", (30, 80),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#         cv2.imshow("Sign Language Inference", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     except Exception:
#         # Headless fallback: print predictions
#         print(f"Static: {label_static} | Dynamic: {label_dynamic}")

# cap.release()
# hands.close()
# pose.close()
# cv2.destroyAllWindows()
import sys
import os
import cv2
import torch
import pickle
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QTextEdit, QVBoxLayout, QFileDialog
from PyQt5.QtCore import QThread, pyqtSignal
import mediapipe as mp

# -------------------------------
# Paths to models
# -------------------------------
STATIC_MODEL_PATH = "models/static_model.pth"
STATIC_LABEL_ENCODER_PATH = "models/static_label_encoder.pkl"
STATIC_SCALER_PATH = "models/static_scaler.pkl"
DYNAMIC_MODEL_PATH = "models/dynamic_augmented_model.pth"
# -------------------------------
# Device
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Load Static Model
# -------------------------------
with open(STATIC_LABEL_ENCODER_PATH, "rb") as f:
    static_le = pickle.load(f)

with open(STATIC_SCALER_PATH, "rb") as f:
    static_scaler = pickle.load(f)

class StaticModel(torch.nn.Module):
    def __init__(self, input_dim=126, num_classes=len(static_le.classes_)):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 256)
        self.drop1 = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(256, 128)
        self.drop2 = torch.nn.Dropout(0.3)
        self.fc3 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.drop1(x)
        x = torch.relu(self.fc2(x))
        x = self.drop2(x)
        return self.fc3(x)

static_model = StaticModel().to(device)
static_model.load_state_dict(torch.load(STATIC_MODEL_PATH, map_location=device))
static_model.eval()
static_class_names = list(static_le.classes_)

# -------------------------------
# Load Dynamic Model
# -------------------------------
class KeypointEncoder(torch.nn.Module):
    def __init__(self, input_size=99, hidden_size=128, embedding_dim=64):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, embedding_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class DynamicSignModel(torch.nn.Module):
    def __init__(self, input_size=99, hidden_size=128, num_layers=2, num_classes=10):
        super().__init__()
        self.encoder = KeypointEncoder(input_size=input_size)
        self.lstm = torch.nn.LSTM(64, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        b, t, f = x.shape
        x = x.view(b*t, f)
        embeddings = self.encoder(x)
        embeddings = embeddings.view(b, t, -1)
        out, _ = self.lstm(embeddings)
        out = out[:, -1, :]
        return self.fc(out)

state_dict = torch.load(DYNAMIC_MODEL_PATH, map_location=device)
num_dynamic_classes = state_dict['fc.weight'].shape[0]
dynamic_model = DynamicSignModel(num_classes=num_dynamic_classes).to(device)
dynamic_model.load_state_dict(state_dict)
dynamic_model.eval()

# -------------------------------
# Mediapipe
# -------------------------------
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# -------------------------------
# Video Processing Thread
# -------------------------------
class VideoProcessor(QThread):
    result_signal = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.SEQUENCE_LENGTH = 30

    def run(self):
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

        cap = cv2.VideoCapture(self.video_path)
        dynamic_buffer = []
        final_text = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --------------------
            # Static keypoints (hands)
            # --------------------
            res_hands = hands.process(frame_rgb)
            hand0 = [0.0]*63  # 21*3
            hand1 = [0.0]*63
            if res_hands.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(res_hands.multi_hand_landmarks):
                    coords = [lm.x for lm in hand_landmarks.landmark] + \
                             [lm.y for lm in hand_landmarks.landmark] + \
                             [lm.z for lm in hand_landmarks.landmark]
                    if idx == 0:
                        hand0 = coords
                    else:
                        hand1 = coords
            static_keypoints = np.array(hand0 + hand1, dtype=np.float32)  # 126
            try:
                static_keypoints = static_scaler.transform(static_keypoints.reshape(1,-1)).reshape(-1)
            except:
                pass
            x_static = torch.tensor(static_keypoints).unsqueeze(0).to(device)
            with torch.no_grad():
                out_static = static_model(x_static)
                pred_static = torch.argmax(out_static, dim=1).item()
                label_static = static_class_names[pred_static]

            # --------------------
            # Dynamic keypoints (pose)
            # --------------------
            res_pose = pose.process(frame_rgb)
            pose_keypoints = [0.0]*99  # 33*3
            if res_pose.pose_landmarks:
                for idx, lm in enumerate(res_pose.pose_landmarks.landmark):
                    pose_keypoints[idx*3:idx*3+3] = [lm.x, lm.y, lm.z]
            dynamic_buffer.append(pose_keypoints)
            if len(dynamic_buffer) > self.SEQUENCE_LENGTH:
                dynamic_buffer.pop(0)
            label_dynamic = ""
            if len(dynamic_buffer) == self.SEQUENCE_LENGTH:
                x_dynamic = torch.tensor([dynamic_buffer], dtype=torch.float32).to(device)
                with torch.no_grad():
                    out_dynamic = dynamic_model(x_dynamic)
                    pred_dynamic = torch.argmax(out_dynamic, dim=1).item()
                    label_dynamic = str(pred_dynamic)

            # Combine static & dynamic results
            if label_dynamic:
                final_text.append(label_dynamic)

        cap.release()
        hands.close()
        pose.close()
        self.result_signal.emit(" ".join(final_text))

# -------------------------------
# GUI
# -------------------------------
class SignLanguageApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Recognition")
        self.layout = QVBoxLayout()

        self.record_btn = QPushButton("Record Video")
        self.process_btn = QPushButton("Process Video")
        self.text_box = QTextEdit()
        self.text_box.setReadOnly(True)

        self.layout.addWidget(self.record_btn)
        self.layout.addWidget(self.process_btn)
        self.layout.addWidget(QLabel("Recognized Text:"))
        self.layout.addWidget(self.text_box)
        self.setLayout(self.layout)

        self.record_btn.clicked.connect(self.record_video)
        self.process_btn.clicked.connect(self.process_video)

        self.video_path = None

    def record_video(self):
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Video", "", "Video Files (*.mp4 *.avi)")
        if not save_path:
            return
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, 20.0, (640,480))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            # cv2.imshow("Recording... Press Q to stop", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        self.video_path = save_path

    def process_video(self):
        if not self.video_path or not os.path.exists(self.video_path):
            self.text_box.setText("No video selected or recorded!")
            return
        self.text_box.setText("Processing video...")
        self.thread = VideoProcessor(self.video_path)
        self.thread.result_signal.connect(self.show_result)
        self.thread.start()

    def show_result(self, text):
        self.text_box.setText(text)

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec_())
