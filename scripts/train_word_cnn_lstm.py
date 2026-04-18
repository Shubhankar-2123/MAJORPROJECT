import argparse
import json
import math
import os
import pickle
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DEVICE, DATA_DIR, MAX_FRAMES, WORDS_MAIN_DIR


VIDEO_EXTS = (".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm")
USER_TOKEN_RE = re.compile(r"(?:^|[_\-])(u\d+|user\d+|p\d+)(?:[_\-]|$)", re.IGNORECASE)


@dataclass
class VideoItem:
    label: str
    path: str
    filename: str


class WordCnnLstm(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, conv_channels: int = 128, lstm_hidden: int = 192):
        super().__init__()
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(feature_dim, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=0.30,
            bidirectional=True,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, T, F]
        x = x.transpose(1, 2)  # [B, F, T]
        x = self.temporal_cnn(x)
        x = x.transpose(1, 2)  # [B, T, C]
        lstm_out, _ = self.lstm(x)
        pooled = torch.cat([lstm_out[:, -1, :], torch.mean(lstm_out, dim=1)], dim=1)
        return self.classifier(pooled)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train improved word-level sign model (MediaPipe Hands + CNN+LSTM)")
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        default=[
            os.path.join(DATA_DIR, "Frames_Word_Level_1"),
            os.path.join(DATA_DIR, "Frames_Word_Level_2"),
            os.path.join(DATA_DIR, "Frames_Word_Level_3"),
            os.path.join(DATA_DIR, "Frames_Word_Level_4"),
            os.path.join(DATA_DIR, "Frames_Word_Level_5"),
            os.path.join(DATA_DIR, "Frames_Word_Level_6"),
        ],
        help="Word-level dataset roots. Each root should contain one subfolder per word.",
    )
    parser.add_argument("--seq-len", type=int, default=MAX_FRAMES, help="Fixed sequence length (frames).")
    parser.add_argument("--min-samples", type=int, default=20, help="Minimum videos per word class.")
    parser.add_argument("--recommended-samples", type=int, default=30, help="Recommended videos per class.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--augment-per-video", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stop", type=int, default=8, help="Early stopping patience by val loss.")
    parser.add_argument("--balance-train", action="store_true", help="Oversample train split so all classes have equal count.")
    parser.add_argument("--target-train-per-class", type=int, default=0, help="If >0, oversample each class to this count in train split.")
    parser.add_argument("--audit-only", action="store_true", help="Only run dataset quality audit and exit.")
    return parser.parse_args()


def collect_videos(data_dirs: Sequence[str]) -> List[VideoItem]:
    items: List[VideoItem] = []
    for base in data_dirs:
        if not os.path.isdir(base):
            continue
        for word in sorted(os.listdir(base)):
            word_dir = os.path.join(base, word)
            if not os.path.isdir(word_dir):
                continue
            for name in sorted(os.listdir(word_dir)):
                if not name.lower().endswith(VIDEO_EXTS):
                    continue
                items.append(VideoItem(label=word, path=os.path.join(word_dir, name), filename=name))
    return items


def infer_user_token(filename: str) -> str:
    m = USER_TOKEN_RE.search(filename)
    return m.group(1).lower() if m else "unknown"


def audit_dataset(items: Sequence[VideoItem], min_samples: int, recommended_samples: int) -> Dict[str, object]:
    label_counts = Counter(i.label for i in items)
    users_per_label: Dict[str, set] = defaultdict(set)
    for i in items:
        users_per_label[i.label].add(infer_user_token(i.filename))

    low_classes = sorted([k for k, v in label_counts.items() if v < min_samples])
    below_recommended = sorted([k for k, v in label_counts.items() if v < recommended_samples])
    low_diversity = sorted([k for k, users in users_per_label.items() if len(users) < 2])

    summary = {
        "total_videos": len(items),
        "num_classes": len(label_counts),
        "min_samples": min_samples,
        "recommended_samples": recommended_samples,
        "classes_below_min": low_classes,
        "classes_below_recommended": below_recommended,
        "classes_low_user_diversity": low_diversity,
        "counts": dict(sorted(label_counts.items(), key=lambda kv: kv[0])),
    }
    return summary


def print_audit(summary: Dict[str, object]) -> None:
    print("\n===== DATASET AUDIT =====")
    print(f"Total videos: {summary['total_videos']}")
    print(f"Word classes: {summary['num_classes']}")
    print(f"Classes below min samples ({summary['min_samples']}): {len(summary['classes_below_min'])}")
    print(f"Classes below recommended ({summary['recommended_samples']}): {len(summary['classes_below_recommended'])}")
    print(f"Classes with low user diversity (<2 detected users): {len(summary['classes_low_user_diversity'])}")

    if summary["classes_below_min"]:
        print("\nClasses below minimum:")
        print(", ".join(summary["classes_below_min"]))

    if summary["classes_low_user_diversity"]:
        print("\nClasses with low user diversity:")
        print(", ".join(summary["classes_low_user_diversity"]))


def extract_hand_keypoints(frame: np.ndarray, hands_model) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_model.process(frame_rgb)

    hand0 = np.zeros((21, 3), dtype=np.float32)
    hand1 = np.zeros((21, 3), dtype=np.float32)
    if results.multi_hand_landmarks:
        for idx, landmarks in enumerate(results.multi_hand_landmarks[:2]):
            arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)
            if idx == 0:
                hand0 = arr
            else:
                hand1 = arr
    both = np.vstack([hand0, hand1])  # [42, 3]
    return both.reshape(-1)  # [126]


def sample_to_fixed_length(seq: np.ndarray, seq_len: int) -> np.ndarray:
    n = seq.shape[0]
    if n == 0:
        return np.zeros((seq_len, seq.shape[1]), dtype=np.float32)
    if n == seq_len:
        return seq.astype(np.float32)
    if n > seq_len:
        idx = np.linspace(0, n - 1, seq_len).astype(np.int32)
        return seq[idx].astype(np.float32)

    # Pad short sequences using the last valid frame to avoid abrupt trailing zeros.
    out = np.zeros((seq_len, seq.shape[1]), dtype=np.float32)
    out[:n] = seq
    out[n:] = seq[n - 1]
    return out


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    arr = seq.reshape(seq.shape[0], 42, 3).copy()
    for t in range(arr.shape[0]):
        pts = arr[t]
        non_zero = np.any(np.abs(pts) > 1e-8, axis=1)
        if not np.any(non_zero):
            continue
        active = pts[non_zero]
        center = np.mean(active, axis=0)
        pts = pts - center
        scale = np.max(np.linalg.norm(pts[non_zero], axis=1)) + 1e-6
        arr[t] = pts / scale
    return arr.reshape(seq.shape[0], -1).astype(np.float32)


def random_rotate_xy(seq: np.ndarray, max_deg: float = 8.0) -> np.ndarray:
    arr = seq.reshape(seq.shape[0], 42, 3).copy()
    theta = np.deg2rad(np.random.uniform(-max_deg, max_deg))
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=np.float32)
    xy = arr[:, :, :2]
    arr[:, :, :2] = np.einsum("ij,tfj->tfi", rot, xy)
    return arr.reshape(seq.shape[0], -1).astype(np.float32)


def temporal_jitter(seq: np.ndarray) -> np.ndarray:
    # Jitter by mildly re-sampling around original timeline.
    t = seq.shape[0]
    anchors = np.linspace(0, t - 1, t)
    jitter = np.random.normal(loc=0.0, scale=0.6, size=t)
    idx = np.clip(np.round(anchors + jitter), 0, t - 1).astype(np.int32)
    return seq[idx].astype(np.float32)


def augment_sequence(seq: np.ndarray) -> List[np.ndarray]:
    variants = []

    # Horizontal flip around normalized center.
    flip = seq.reshape(seq.shape[0], 42, 3).copy()
    flip[:, :, 0] *= -1.0
    variants.append(flip.reshape(seq.shape[0], -1).astype(np.float32))

    # Small Gaussian noise.
    noise = seq + np.random.normal(0, 0.015, size=seq.shape).astype(np.float32)
    variants.append(noise.astype(np.float32))

    # Slight rotation in XY plane.
    variants.append(random_rotate_xy(seq, max_deg=8.0))

    # Mild timeline perturbation.
    variants.append(temporal_jitter(seq))
    return variants


def extract_video_sequence(video_path: str, seq_len: int, hands_model) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(extract_hand_keypoints(frame, hands_model))
    cap.release()

    if not frames:
        return None
    seq = np.asarray(frames, dtype=np.float32)
    seq = sample_to_fixed_length(seq, seq_len)
    seq = normalize_sequence(seq)
    return seq


def build_dataset(
    items: Sequence[VideoItem],
    seq_len: int,
    min_samples: int,
    augment_per_video: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], int]:
    per_label = Counter(i.label for i in items)
    filtered = [i for i in items if per_label[i.label] >= min_samples]
    dropped = len(items) - len(filtered)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
    )

    X, y = [], []
    skipped = 0
    for item in filtered:
        seq = extract_video_sequence(item.path, seq_len, hands)
        if seq is None:
            skipped += 1
            continue

        X.append(seq)
        y.append(item.label)

        variants = augment_sequence(seq)
        for aug_seq in variants[: max(0, augment_per_video)]:
            X.append(aug_seq)
            y.append(item.label)

    hands.close()

    if not X:
        raise RuntimeError("No usable video sequences extracted. Check dataset paths and video quality.")

    return np.asarray(X, dtype=np.float32), np.asarray(y), dict(per_label), dropped + skipped


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += float(loss.item()) * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += int((preds == yb).sum().item())
            total += int(yb.size(0))
    avg_loss = total_loss / max(total, 1)
    acc = total_correct / max(total, 1)
    return avg_loss, acc


def rebalance_train_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    target_per_class: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices_by_class: Dict[int, np.ndarray] = {}
    for cls in sorted(np.unique(y_train).tolist()):
        indices_by_class[int(cls)] = np.where(y_train == cls)[0]

    if target_per_class > 0:
        target = int(target_per_class)
    else:
        target = max(len(v) for v in indices_by_class.values())

    selected_indices: List[np.ndarray] = []
    for cls, idxs in indices_by_class.items():
        if len(idxs) == 0:
            continue
        if len(idxs) >= target:
            chosen = rng.choice(idxs, size=target, replace=False)
        else:
            chosen = rng.choice(idxs, size=target, replace=True)
        selected_indices.append(chosen)

    all_idx = np.concatenate(selected_indices)
    rng.shuffle(all_idx)
    return X_train[all_idx], y_train[all_idx]


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    videos = collect_videos(args.data_dirs)
    if not videos:
        raise RuntimeError("No videos found. Check --data-dirs paths.")

    audit = audit_dataset(videos, args.min_samples, args.recommended_samples)
    print_audit(audit)

    perf_dir = os.path.join(DATA_DIR, "performance")
    os.makedirs(perf_dir, exist_ok=True)
    audit_path = os.path.join(perf_dir, f"word_dataset_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print(f"\nSaved dataset audit to: {audit_path}")

    if args.audit_only:
        print("Audit-only mode complete.")
        return

    X, y_raw, per_label, dropped = build_dataset(
        videos,
        seq_len=args.seq_len,
        min_samples=args.min_samples,
        augment_per_video=args.augment_per_video,
    )
    print(f"\nDataset ready: X={X.shape}, y={y_raw.shape}, dropped/skipped={dropped}")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(le.classes_)
    print(f"Classes after filtering: {num_classes}")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=y,
    )

    if args.balance_train or args.target_train_per_class > 0:
        X_train, y_train = rebalance_train_split(
            X_train,
            y_train,
            seed=args.seed,
            target_per_class=int(args.target_train_per_class),
        )
        train_counts = Counter(y_train.tolist())
        print(f"Balanced train split active. Per-class count range: {min(train_counts.values())}..{max(train_counts.values())}")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = WordCnnLstm(feature_dim=X.shape[2], num_classes=num_classes).to(DEVICE)
    class_counts = Counter(y_train.tolist())
    class_weights = np.zeros(num_classes, dtype=np.float32)
    for cls_idx in range(num_classes):
        class_weights[cls_idx] = 1.0 / max(class_counts.get(cls_idx, 1), 1)
    class_weights = class_weights / np.mean(class_weights)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=DEVICE),
        label_smoothing=0.05,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            running_correct += int((preds == yb).sum().item())
            running_total += int(yb.size(0))

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.early_stop:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training finished without a valid checkpoint.")

    model.load_state_dict(best_state)

    # Final validation report using best checkpoint.
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb.to(DEVICE))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(yb.numpy().tolist())

    labels_union = sorted(set(y_true) | set(y_pred))
    target_names = le.inverse_transform(labels_union)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels_union,
        target_names=target_names,
        zero_division=0,
    )
    print("\n===== VALIDATION CLASSIFICATION REPORT =====")
    print(report)

    os.makedirs(WORDS_MAIN_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    latest_model_path = os.path.join(WORDS_MAIN_DIR, "word_cnn_lstm_model_latest.pth")
    version_model_path = os.path.join(WORDS_MAIN_DIR, f"word_cnn_lstm_model_{ts}.pth")
    latest_encoder_path = os.path.join(WORDS_MAIN_DIR, "word_cnn_lstm_label_encoder_latest.pkl")
    version_encoder_path = os.path.join(WORDS_MAIN_DIR, f"word_cnn_lstm_label_encoder_{ts}.pkl")
    metadata_path = os.path.join(WORDS_MAIN_DIR, f"word_model_metadata_{ts}.json")

    torch.save(best_state, latest_model_path)
    torch.save(best_state, version_model_path)
    with open(latest_encoder_path, "wb") as f:
        pickle.dump(le, f)
    with open(version_encoder_path, "wb") as f:
        pickle.dump(le, f)

    metadata = {
        "timestamp": ts,
        "architecture": "WordCnnLstm",
        "device": str(DEVICE),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "num_classes": int(num_classes),
        "classes": le.classes_.tolist(),
        "sequence_length": int(args.seq_len),
        "feature_dim": int(X.shape[2]),
        "training_args": vars(args),
        "dataset": {
            "total_videos_before_filter": len(videos),
            "per_label_video_counts": per_label,
            "dropped_or_skipped": dropped,
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_val)),
        },
        "history": history,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    report_path = os.path.join(perf_dir, f"word_training_report_{ts}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Word-Level CNN+LSTM Validation Report\n\n")
        f.write(report)

    print("\n===== SAVED ARTIFACTS =====")
    print(f"Latest model:   {latest_model_path}")
    print(f"Version model:  {version_model_path}")
    print(f"Latest encoder: {latest_encoder_path}")
    print(f"Version encoder:{version_encoder_path}")
    print(f"Metadata:       {metadata_path}")
    print(f"Report:         {report_path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)