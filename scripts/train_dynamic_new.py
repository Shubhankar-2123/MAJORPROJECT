# # # train_dynamic_full.py

# # import os
# # import cv2
# # import numpy as np
# # import mediapipe as mp
# # import torch
# # import torch.nn as nn
# # from torch.utils.data import TensorDataset, DataLoader
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import classification_report

# # # ---------------------------
# # # 1. CONFIG
# # # ---------------------------
# # VIDEO_DIR = "data/dynamic"  # folder containing subfolders per class
# # MAX_FRAMES = 30       # frames per video
# # AUGMENTATIONS = 5     # augmented samples per video
# # BATCH_SIZE = 16
# # EPOCHS = 30
# # LR = 0.001
# # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # # ---------------------------
# # # 2. MEDIAPIPE SETUP
# # # ---------------------------
# # mp_pose = mp.solutions.pose
# # pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# # def extract_keypoints(video_path, max_frames=MAX_FRAMES):
# #     cap = cv2.VideoCapture(video_path)
# #     keypoints_seq = []

# #     while len(keypoints_seq) < max_frames:
# #         ret, frame = cap.read()
# #         if not ret:
# #             break
# #         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #         results = pose.process(frame_rgb)

# #         if results.pose_landmarks:
# #             kp = []
# #             for lm in results.pose_landmarks.landmark:
# #                 kp.extend([lm.x, lm.y, lm.z])
# #             keypoints_seq.append(kp)
# #         else:
# #             keypoints_seq.append([0]*99)  # 33 keypoints * 3 coords

# #     cap.release()
# #     # Pad or truncate
# #     while len(keypoints_seq) < max_frames:
# #         keypoints_seq.append([0]*99)
# #     return np.array(keypoints_seq)

# # def normalize_keypoints(seq):
# #     seq = np.array(seq)
# #     seq[:, 0::3] = (seq[:, 0::3] - np.min(seq[:, 0::3])) / (np.max(seq[:, 0::3]) - np.min(seq[:, 0::3]) + 1e-6)
# #     seq[:, 1::3] = (seq[:, 1::3] - np.min(seq[:, 1::3])) / (np.max(seq[:, 1::3]) - np.min(seq[:, 1::3]) + 1e-6)
# #     return seq

# # def augment_sequence(seq, n_augments=AUGMENTATIONS):
# #     augmented = []
# #     for _ in range(n_augments):
# #         new_seq = seq.copy()
# #         new_seq += np.random.normal(0, 0.02, new_seq.shape)  # small noise
# #         new_seq[:, 0::3] = 1 - new_seq[:, 0::3]  # horizontal flip
# #         augmented.append(new_seq)
# #     return augmented

# # # ---------------------------
# # # 3. LOAD DATA
# # # ---------------------------
# # X, y = [], []
# # classes = sorted(os.listdir(VIDEO_DIR))
# # class_map = {cls_name: idx for idx, cls_name in enumerate(classes)}

# # for cls_name in classes:
# #     cls_idx = class_map[cls_name]
# #     cls_folder = os.path.join(VIDEO_DIR, cls_name)
# #     for vid_file in os.listdir(cls_folder):
# #         vid_path = os.path.join(cls_folder, vid_file)
# #         seq = extract_keypoints(vid_path)
# #         seq = normalize_keypoints(seq)
# #         X.append(seq)
# #         y.append(cls_idx)

# #         # Augmentation
# #         aug_seqs = augment_sequence(seq, AUGMENTATIONS)
# #         for a in aug_seqs:
# #             X.append(a)
# #             y.append(cls_idx)

# # X = np.array(X)
# # y = np.array(y)
# # print("Dataset shape after augmentation:", X.shape, y.shape)

# # from sklearn.preprocessing import LabelEncoder
# # import pickle

# # # Encode class labels as integers
# # le = LabelEncoder()
# # y = le.fit_transform(y)

# # # Save LabelEncoder for inference
# # os.makedirs("models", exist_ok=True)
# # with open("models/dynamic_label_encoder.pkl", "wb") as f:
# #     pickle.dump(le, f)
# # print("✅ Dynamic LabelEncoder saved")


# # # ---------------------------
# # # 4. TRAIN/TEST SPLIT
# # # ---------------------------
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# # train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
# #                               torch.tensor(y_train, dtype=torch.long))
# # test_dataset  = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
# #                               torch.tensor(y_test, dtype=torch.long))

# # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# # test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# # # ---------------------------
# # # 5. ENCODER + LSTM MODEL
# # # ---------------------------
# # class Encoder(nn.Module):
# #     def __init__(self, input_size=99, hidden_size=128, encoded_size=64):
# #         super(Encoder, self).__init__()
# #         self.fc1 = nn.Linear(input_size, hidden_size)
# #         self.relu = nn.ReLU()
# #         self.fc2 = nn.Linear(hidden_size, encoded_size)

# #     def forward(self, x):
# #         # x shape: (batch, seq_len, input_size)
# #         b, t, f = x.shape
# #         x = x.view(b * t, f)       # flatten sequence
# #         x = self.fc1(x)
# #         x = self.relu(x)
# #         x = self.fc2(x)
# #         x = x.view(b, t, -1)       # reshape back for LSTM
# #         return x

# # # ---------------------------
# # # 6. LSTM MODEL
# # # ---------------------------
# # class DynamicLSTM(nn.Module):
# #     def __init__(self, input_size=99, hidden_size=128, num_layers=2, num_classes=len(classes)):
# #         super(DynamicLSTM, self).__init__()
# #         self.encoder = Encoder(input_size=input_size, hidden_size=128, encoded_size=64)
# #         self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
# #         self.fc = nn.Linear(hidden_size, num_classes)

# #     def forward(self, x):
# #         x = self.encoder(x)       # 🔹 Pass through encoder first
# #         out, _ = self.lstm(x)
# #         out = out[:, -1, :]
# #         out = self.fc(out)
# #         return out

# # model = DynamicLSTM().to(DEVICE)
# # criterion = nn.CrossEntropyLoss()
# # optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# # # ---------------------------
# # # 6. TRAINING
# # # ---------------------------
# # for epoch in range(EPOCHS):
# #     model.train()
# #     total_loss = 0
# #     for X_batch, y_batch in train_loader:
# #         X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
# #         optimizer.zero_grad()
# #         outputs = model(X_batch)
# #         loss = criterion(outputs, y_batch)
# #         loss.backward()
# #         optimizer.step()
# #         total_loss += loss.item()

# #     # Calculate validation accuracy
# #     model.eval()
# #     correct = 0
# #     total = 0
# #     with torch.no_grad():
# #         for X_batch, y_batch in test_loader:
# #             X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
# #             outputs = model(X_batch)
# #             preds = torch.argmax(outputs, dim=1)
# #             correct += (preds == y_batch).sum().item()
# #             total += y_batch.size(0)
# #     val_acc = correct / total

# #     print(f"Epoch {epoch+1}/{EPOCHS}, Loss={total_loss/len(train_loader):.4f}, Val_Acc={val_acc:.4f}")

# # # ---------------------------
# # # 7. EVALUATION
# # # ---------------------------
# # model.eval()
# # y_pred = []
# # with torch.no_grad():
# #     for X_batch, _ in test_loader:
# #         X_batch = X_batch.to(DEVICE)
# #         outputs = model(X_batch)
# #         preds = torch.argmax(outputs, dim=1)
# #         y_pred.extend(preds.cpu().numpy())

# # print("Classification Report:")
# # print(classification_report(y_test, y_pred, target_names=classes, zero_division=0))


# # # ---------------------------
# # # 8. SANITY CHECK
# # # ---------------------------
# # print("\n✅ Sanity Check on a few test samples:")

# # num_samples = 5  # number of samples to check
# # indices = np.random.choice(len(X_test), num_samples, replace=False)

# # for idx in indices:
# #     sample_seq = X_test[idx]
# #     true_label = y_test[idx]
    
# #     sample_tensor = torch.tensor(sample_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
# #     model.eval()
# #     with torch.no_grad():
# #         output = model(sample_tensor)
# #         pred_label = torch.argmax(output, dim=1).item()
    
# #     print(f"Sample {idx}: True label = {true_label}, Predicted = {pred_label}")


# # # ---------------------------
# # # 8. SAVE MODEL + ENCODER
# # # ---------------------------
# # os.makedirs("models", exist_ok=True)
# # torch.save(model.state_dict(), "models/dynamic_augmented_model.pth")
# #  # 🔹 save encoder separately
# # print("✅ Model + Encoder saved")

# # train_dynamic_full.py

# import os
# import cv2
# import numpy as np
# import mediapipe as mp
# import torch
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# import pickle

# # ---------------------------
# # 1. MODEL CLASSES
# # ---------------------------

# class Encoder(nn.Module):
#     def __init__(self, input_size=99, hidden_size=128, encoded_size=64):
#         super(Encoder, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, encoded_size)

#     def forward(self, x):
#         b, t, f = x.shape
#         x = x.view(b * t, f)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = x.view(b, t, -1)
#         return x


# class DynamicLSTM(nn.Module):
#     def __init__(self, input_size=99, hidden_size=128, num_layers=2, num_classes=10):
#         super(DynamicLSTM, self).__init__()
#         self.encoder = Encoder(input_size=input_size, hidden_size=128, encoded_size=64)
#         self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_classes)

#     def forward(self, x):
#         x = self.encoder(x)
#         out, _ = self.lstm(x)
#         out = out[:, -1, :]
#         out = self.fc(out)
#         return out

# # ---------------------------
# # 2. DATA PREPROCESSING FUNCTIONS
# # ---------------------------

# mp_pose = mp.solutions.pose

# def extract_keypoints(video_path, max_frames=30):
#     pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
#     cap = cv2.VideoCapture(video_path)
#     keypoints_seq = []

#     while len(keypoints_seq) < max_frames:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(frame_rgb)
#         if results.pose_landmarks:
#             kp = []
#             for lm in results.pose_landmarks.landmark:
#                 kp.extend([lm.x, lm.y, lm.z])
#             keypoints_seq.append(kp)
#         else:
#             keypoints_seq.append([0]*99)
#     cap.release()

#     while len(keypoints_seq) < max_frames:
#         keypoints_seq.append([0]*99)
#     return np.array(keypoints_seq)

# def normalize_keypoints(seq):
#     seq = np.array(seq)
#     seq[:, 0::3] = (seq[:, 0::3] - np.min(seq[:, 0::3])) / (np.max(seq[:, 0::3]) - np.min(seq[:, 0::3]) + 1e-6)
#     seq[:, 1::3] = (seq[:, 1::3] - np.min(seq[:, 1::3])) / (np.max(seq[:, 1::3]) - np.min(seq[:, 1::3]) + 1e-6)
#     return seq

# def augment_sequence(seq, n_augments=5):
#     augmented = []
#     for _ in range(n_augments):
#         new_seq = seq.copy().astype(np.float32)
#         new_seq += np.random.normal(0, 0.02, new_seq.shape)
#         new_seq[:, 0::3] = 1 - new_seq[:, 0::3]
#         augmented.append(new_seq)
#     return augmented

# def encode_labels(y, save_path="models/dynamic_label_encoder.pkl"):
#     le = LabelEncoder()
#     y_enc = le.fit_transform(y)
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     with open(save_path, "wb") as f:
#         pickle.dump(le, f)
#     print(f"✅ LabelEncoder saved at {save_path}")
#     return y_enc, le


# # ---------------------------
# # 3. MAIN TRAINING CODE
# # ---------------------------

# if __name__ == "__main__":
#     VIDEO_DIR = "data/dynamic"
#     MAX_FRAMES = 30
#     AUGMENTATIONS = 5
#     BATCH_SIZE = 16
#     EPOCHS = 30
#     LR = 0.001
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#     # Load dataset
#     X, y = [], []
#     classes = sorted(os.listdir(VIDEO_DIR))
#     class_map = {cls_name: idx for idx, cls_name in enumerate(classes)}

#     for cls_name in classes:
#         cls_idx = class_map[cls_name]
#         cls_folder = os.path.join(VIDEO_DIR, cls_name)
#         for vid_file in os.listdir(cls_folder):
#             vid_path = os.path.join(cls_folder, vid_file)
#             seq = extract_keypoints(vid_path)
#             seq = normalize_keypoints(seq)
#             X.append(seq)
#             y.append(cls_idx)
#             for a in augment_sequence(seq, AUGMENTATIONS):
#                 X.append(a)
#                 y.append(cls_idx)

#     X = np.array(X)
#     y = np.array(y)
#     print("Dataset shape after augmentation:", X.shape, y.shape)

#     # Encode labels
#     y, le = encode_labels(y)

#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#     train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
#     test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

#     # Instantiate model
#     model = DynamicLSTM(num_classes=len(classes)).to(DEVICE)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#     # Training loop
#     for epoch in range(EPOCHS):
#         model.train()
#         total_loss = 0
#         for X_batch, y_batch in train_loader:
#             X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
#             optimizer.zero_grad()
#             outputs = model(X_batch)
#             loss = criterion(outputs, y_batch)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         # Validation accuracy
#         model.eval()
#         correct, total = 0, 0
#         with torch.no_grad():
#             for X_batch, y_batch in test_loader:
#                 X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
#                 outputs = model(X_batch)
#                 preds = torch.argmax(outputs, dim=1)
#                 correct += (preds == y_batch).sum().item()
#                 total += y_batch.size(0)
#         val_acc = correct / total

#         print(f"Epoch {epoch+1}/{EPOCHS}, Loss={total_loss/len(train_loader):.4f}, Val_Acc={val_acc:.4f}")

#     # Save model
#     os.makedirs("models", exist_ok=True)
#     torch.save(model.state_dict(), "models/dynamic_augmented_model.pth")
#     print("✅ Model saved at models/dynamic_augmented_model.pth")


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
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    keypoints_seq = []

    while len(keypoints_seq) < max_frames:
        ret, frame = cap.read()
        if not ret:
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


def encode_labels(y, save_path="models/dynamic_label_encoder.pkl"):
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
    VIDEO_DIR = "data/dynamic"
    MAX_FRAMES = 30
    AUGMENTATIONS = 5
    BATCH_SIZE = 16
    EPOCHS = 30
    LR = 0.001
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    X, y = [], []
    classes = sorted(os.listdir(VIDEO_DIR))

    for cls_name in classes:
        cls_folder = os.path.join(VIDEO_DIR, cls_name)
        for vid_file in os.listdir(cls_folder):
            vid_path = os.path.join(cls_folder, vid_file)
            seq = extract_keypoints(vid_path, MAX_FRAMES)
            seq = normalize_keypoints(seq)
            X.append(seq)
            y.append(cls_name)   # use class names instead of int
            # augment
            for a in augment_sequence(seq, AUGMENTATIONS):
                X.append(a)
                y.append(cls_name)

    X = np.array(X)
    y = np.array(y)
    print("Dataset shape after augmentation:", X.shape, y.shape)

    # Encode labels (string -> int)
    y, le = encode_labels(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Model
    model = DynamicLSTM(num_classes=len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Training loop
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

        # Validation accuracy
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

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/dynamic_augmented_model.pth")
    print("✅ Model saved at models/dynamic_augmented_model.pth")

    # ---------------------------
    # Classification Report
    # ---------------------------
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(y_batch.cpu().numpy())

    labels_in_use = sorted(set(y_true) | set(y_pred))
    target_names = le.inverse_transform(labels_in_use)  # returns strings now ✅

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=labels_in_use, target_names=target_names, zero_division=0))

    # Sanity check
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
