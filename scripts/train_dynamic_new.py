

import os
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle
from datetime import datetime
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DEVICE, MAX_FRAMES, PROJECT_ROOT, WORDS_MAIN_DIR, DATA_DIR

# ---------------------------
# Model Classes
# ---------------------------

class Encoder(nn.Module):
    def __init__(self, input_size=99, hidden_size=128, encoded_size=64):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, encoded_size)

    def forward(self, x):
        b, t, f = x.shape
        x = x.view(b * t, f)  # flatten sequence
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.view(b, t, -1)  # reshape back
        return x


class DynamicLSTM(nn.Module):
    def __init__(self, input_size=99, hidden_size=128, num_layers=2, num_classes=10):
        super(DynamicLSTM, self).__init__()
        self.encoder = Encoder(input_size=input_size, hidden_size=128, encoded_size=64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        out = self.fc(out)
        return out


# ---------------------------
# Data Helpers
# ---------------------------

mp_pose = mp.solutions.pose

def extract_keypoints(video_path, max_frames=30):
    if not os.path.exists(video_path):
        print(f"⚠️ Video not found: {video_path}")
        return np.zeros((max_frames, 99))

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(video_path)
    keypoints_seq = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_read = min(max_frames, total_frames)

    for _ in range(frames_to_read):
        try:
            ret, frame = cap.read()
            if not ret:
                break
        except Exception as e:
            print(f"Error reading frame: {e}")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            kp = []
            for lm in results.pose_landmarks.landmark:
                kp.extend([lm.x, lm.y, lm.z])
            keypoints_seq.append(kp)
        else:
            keypoints_seq.append([0]*99)

    cap.release()
    pose.close()

    while len(keypoints_seq) < max_frames:
        keypoints_seq.append([0]*99)

    return np.array(keypoints_seq)


def normalize_keypoints(seq):
    seq = np.array(seq)
    seq[:, 0::3] = (seq[:, 0::3] - np.min(seq[:, 0::3])) / (np.max(seq[:, 0::3]) - np.min(seq[:, 0::3]) + 1e-6)
    seq[:, 1::3] = (seq[:, 1::3] - np.min(seq[:, 1::3])) / (np.max(seq[:, 1::3]) - np.min(seq[:, 1::3]) + 1e-6)
    return seq


def augment_sequence(seq, n_augments=5):
    augmented = []
    for _ in range(n_augments):
        new_seq = seq.copy().astype(np.float32)
        new_seq += np.random.normal(0, 0.02, new_seq.shape)  # noise
        new_seq[:, 0::3] = 1 - new_seq[:, 0::3]  # horizontal flip
        augmented.append(new_seq)
    return augmented


def encode_labels(y, save_path=os.path.join(WORDS_MAIN_DIR, "word_label_encoder_2.pkl")):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(le, f)
    print(f"✅ LabelEncoder saved at {save_path}")
    return y_enc, le


# ---------------------------
# Main Script
# ---------------------------

if __name__ == "__main__":
    VIDEO_SOURCES = [
        os.path.join(PROJECT_ROOT, "data", "Frames_Word_Level_2"),
    ]
    VIDEO_EXTS = (".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm")

    # Using MAX_FRAMES from config
    AUGMENTATIONS = 5
    BATCH_SIZE = 16
    EPOCHS = 30
    LR = 0.001
    # Using DEVICE from config

    X, y = [], []

    for base in VIDEO_SOURCES:
        if not os.path.isdir(base):
            continue
        for cls_name in sorted(os.listdir(base)):
            cls_folder = os.path.join(base, cls_name)
            if not os.path.isdir(cls_folder):
                continue
            for vid_file in sorted(os.listdir(cls_folder)):
                if not vid_file.lower().endswith(VIDEO_EXTS):
                    continue
                vid_path = os.path.join(cls_folder, vid_file)
                seq = extract_keypoints(vid_path, MAX_FRAMES)
                seq = normalize_keypoints(seq)
                X.append(seq)
                y.append(cls_name)
                for a in augment_sequence(seq, AUGMENTATIONS):
                    X.append(a)
                    y.append(cls_name)

    if not X:
        raise RuntimeError("No training videos found in VIDEO_SOURCES. Ensure data exists.")

    X = np.array(X)
    y = np.array(y)
    print("Dataset shape after augmentation:", X.shape, y.shape)

    y, le = encode_labels(y)
    num_classes = len(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = DynamicLSTM(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        val_acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss={total_loss/len(train_loader):.4f}, Val_Acc={val_acc:.4f}")

    base_dir = WORDS_MAIN_DIR
    os.makedirs(base_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_path = os.path.join(base_dir, "words_augmented_model_2.pth")
    version_path = os.path.join(base_dir, f"words_augmented_model_{ts}.pth")
    torch.save(model.state_dict(), latest_path)
    torch.save(model.state_dict(), version_path)
    # Save label encoder versioned copy
    with open(os.path.join(base_dir, f"word_label_encoder_{ts}.pkl"), "wb") as f:
        pickle.dump(le, f)
    print(f"✅ Model saved at {latest_path} and versioned {version_path}")

    # Classification Report
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(y_batch.cpu().numpy())

    target_names = le.inverse_transform(sorted(set(y_true) | set(y_pred)))
    print("\nClassification Report:")
    report_text = classification_report(y_true, y_pred, labels=sorted(set(y_true) | set(y_pred)),
                                target_names=target_names, zero_division=0)
    print(report_text)

    # Save performance report
    perf_dir = os.path.join(DATA_DIR, "performance")
    os.makedirs(perf_dir, exist_ok=True)
    with open(os.path.join(perf_dir, f"dynamic_word_training_report_{ts}.txt"), "w", encoding="utf-8") as f:
        f.write("Classification Report\n\n")
        f.write(report_text)

    # Sanity check on random samples
    print("\nSanity check on random test samples:")
    num_samples = 5
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    for idx in indices:
        sample_seq = X_test[idx]
        true_label = y_test[idx]
        sample_tensor = torch.tensor(sample_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(sample_tensor)
            pred_label = torch.argmax(output, dim=1).item()
        print(f"Sample {idx}: True = {le.inverse_transform([true_label])[0]}, Pred = {le.inverse_transform([pred_label])[0]}")
