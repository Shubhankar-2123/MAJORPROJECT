
import argparse
import hashlib
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from config import STATIC_MAIN_DIR, PROJECT_ROOT
from utils.preprocessing import _extract_static_hand_keypoints_from_bgr

class StaticModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        return self.fc3(x)


def _seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_label(label: str) -> str:
    return str(label).strip().upper()


def _default_image_roots() -> list:
    candidates = [
        os.path.join(PROJECT_ROOT, "data", "Indian"),
        os.path.join(PROJECT_ROOT, "data", "static"),
        os.path.join(PROJECT_ROOT, "data", "static_images"),
    ]
    return [p for p in candidates if os.path.isdir(p)]


def _iter_image_files(root_dir: str):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for label_name in sorted(os.listdir(root_dir)):
        label_dir = os.path.join(root_dir, label_name)
        if not os.path.isdir(label_dir):
            continue
        label = _normalize_label(label_name)
        for fn in os.listdir(label_dir):
            ext = os.path.splitext(fn)[1].lower()
            if ext in exts:
                yield label, os.path.join(label_dir, fn), fn


def _is_alphabet_label(label: str) -> bool:
    return len(label) == 1 and "A" <= label <= "Z"


def _load_from_image_roots(image_roots: list, alphabet_only: bool = True) -> tuple:
    rows = []
    skipped = 0

    for root in image_roots:
        print(f"Extracting landmarks from images in: {root}")
        for label, image_path, filename in _iter_image_files(root):
            if alphabet_only and not _is_alphabet_label(label):
                continue

            img = cv2.imread(image_path)
            if img is None:
                skipped += 1
                continue

            try:
                keypoints = _extract_static_hand_keypoints_from_bgr(img)
            except Exception:
                skipped += 1
                continue

            if len(keypoints) < 126:
                keypoints = keypoints + [0.0] * (126 - len(keypoints))
            elif len(keypoints) > 126:
                keypoints = keypoints[:126]

            rows.append((label, "image", filename, -1, np.array(keypoints, dtype=np.float32)))

    if not rows:
        return np.empty((0, 126), dtype=np.float32), np.array([], dtype=str)

    X = np.vstack([r[4] for r in rows]).astype(np.float32)
    y = np.array([r[0] for r in rows], dtype=str)
    print(f"Loaded {len(rows)} image-derived samples (skipped {skipped} unreadable/failed files).")
    return X, y


def _load_from_static_csv(static_csv: str, alphabet_only: bool = True) -> tuple:
    if not static_csv or not os.path.exists(static_csv):
        return np.empty((0, 126), dtype=np.float32), np.array([], dtype=str)

    df = pd.read_csv(static_csv, dtype={"label": str}, low_memory=False)
    if "type" in df.columns:
        df = df[df["type"].astype(str).str.lower() == "image"]

    if "label" not in df.columns:
        return np.empty((0, 126), dtype=np.float32), np.array([], dtype=str)

    df["label"] = df["label"].astype(str).map(_normalize_label)
    if alphabet_only:
        df = df[df["label"].map(_is_alphabet_label)]

    drop_cols = [c for c in ["label", "type", "file", "frame"] if c in df.columns]
    feature_df = df.drop(columns=drop_cols)
    if feature_df.shape[1] != 126:
        raise ValueError(
            f"Expected 126 static features in CSV, found {feature_df.shape[1]}. "
            "Regenerate static_keypoints.csv with the current extractor."
        )

    X = feature_df.values.astype(np.float32)
    y = df["label"].values.astype(str)
    print(f"Loaded {len(y)} CSV samples from: {static_csv}")
    return X, y


def _dedupe_samples(X: np.ndarray, y: np.ndarray) -> tuple:
    if len(y) == 0:
        return X, y

    seen = set()
    keep_indices = []
    for idx, (label, row) in enumerate(zip(y, X)):
        payload = f"{label}|".encode("utf-8") + np.round(row, 5).astype(np.float32).tobytes()
        h = hashlib.sha1(payload).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        keep_indices.append(idx)

    return X[keep_indices], y[keep_indices]


def _remove_noisy_samples(X: np.ndarray, y: np.ndarray, min_nonzero_ratio: float = 0.08) -> tuple:
    if len(y) == 0:
        return X, y

    # Remove samples with almost no detected landmarks.
    nonzero_ratio = (np.abs(X) > 1e-8).sum(axis=1) / X.shape[1]
    keep = nonzero_ratio >= min_nonzero_ratio
    X = X[keep]
    y = y[keep]

    # Remove extreme outliers per class using landmark-vector norm IQR fences.
    by_class = defaultdict(list)
    norms = np.linalg.norm(X, axis=1)
    for i, label in enumerate(y):
        by_class[label].append(i)

    keep_mask = np.ones(len(y), dtype=bool)
    for label, indices in by_class.items():
        if len(indices) < 20:
            continue
        n = norms[indices]
        q1 = np.percentile(n, 25)
        q3 = np.percentile(n, 75)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        for local_idx, global_idx in enumerate(indices):
            if n[local_idx] < lo or n[local_idx] > hi:
                keep_mask[global_idx] = False

    return X[keep_mask], y[keep_mask]


def _filter_min_samples(X: np.ndarray, y: np.ndarray, min_samples_per_class: int) -> tuple:
    counts = Counter(y)
    valid = {k for k, v in counts.items() if v >= min_samples_per_class}
    mask = np.array([label in valid for label in y], dtype=bool)
    return X[mask], y[mask]


def _build_balanced_sampler(y_encoded: np.ndarray) -> WeightedRandomSampler:
    class_counts = Counter(y_encoded.tolist())
    weights = np.array([1.0 / class_counts[int(c)] for c in y_encoded], dtype=np.float64)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def _split_data(X: np.ndarray, y: np.ndarray, test_size: float, val_size: float, seed: int):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    val_adjusted = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_adjusted,
        random_state=seed,
        stratify=y_trainval,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _print_distribution(name: str, labels: np.ndarray):
    counts = Counter(labels)
    print(f"{name} class distribution ({len(counts)} classes, {len(labels)} samples):")
    preview = sorted(counts.items(), key=lambda kv: kv[0])
    print(", ".join([f"{k}:{v}" for k, v in preview]))


def train_static_model(
    static_csv: str = os.path.join(PROJECT_ROOT, "data_keypoints", "static_keypoints.csv"),
    image_roots: list = None,
    output_dir: str = STATIC_MAIN_DIR,
    alphabet_only: bool = True,
    min_samples_per_class: int = 30,
    test_size: float = 0.15,
    val_size: float = 0.15,
    num_epochs: int = 80,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 42,
    early_stopping_patience: int = 12,
):
    _seed_everything(seed)

    if image_roots is None:
        image_roots = _default_image_roots()

    print("\n=== Static Alphabet Retraining (Combined Dataset) ===")
    print(f"CSV source: {static_csv if static_csv else 'None'}")
    print(f"Image roots: {image_roots}")
    print(f"Alphabet-only mode: {alphabet_only}")

    X_csv, y_csv = _load_from_static_csv(static_csv, alphabet_only=alphabet_only)
    X_img, y_img = _load_from_image_roots(image_roots, alphabet_only=alphabet_only)

    if len(y_csv) == 0 and len(y_img) == 0:
        raise ValueError("No training data found from CSV or image roots.")

    X = np.vstack([arr for arr in [X_csv, X_img] if len(arr) > 0]).astype(np.float32)
    y = np.concatenate([arr for arr in [y_csv, y_img] if len(arr) > 0]).astype(str)
    print(f"Combined samples (before cleanup): {len(y)}")

    X, y = _dedupe_samples(X, y)
    print(f"After de-duplication: {len(y)}")

    X, y = _remove_noisy_samples(X, y)
    print(f"After noisy-sample filtering: {len(y)}")

    X, y = _filter_min_samples(X, y, min_samples_per_class=min_samples_per_class)
    print(f"After min-samples-per-class filter ({min_samples_per_class}): {len(y)}")

    if len(np.unique(y)) < 2:
        raise ValueError("Need at least 2 classes after filtering to train a classifier.")

    _print_distribution("Cleaned", y)

    X_train, X_val, X_test, y_train_raw, y_val_raw, y_test_raw = _split_data(
        X, y, test_size=test_size, val_size=val_size, seed=seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    y_test = le.transform(y_test_raw)

    _print_distribution("Train", y_train_raw)
    _print_distribution("Val", y_val_raw)
    _print_distribution("Test", y_test_raw)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    train_sampler = _build_balanced_sampler(y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    num_classes = len(le.classes_)

    model = StaticModel(input_dim, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    class_weights = Counter(y_train.tolist())
    weight_tensor = np.array([1.0 / class_weights[i] for i in range(num_classes)], dtype=np.float32)
    weight_tensor = weight_tensor / weight_tensor.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight_tensor, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
        min_lr=1e-5,
    )

    best_val_acc = -1.0
    best_state = None
    best_epoch = 0
    no_improve = 0
    history = []

    print("\n=== Training ===")
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == yb).sum().item()
            train_total += yb.size(0)

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss_sum += loss.item() * xb.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        scheduler.step(val_acc)

        history.append((epoch, train_loss, train_acc, val_loss, val_acc))
        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid model state.")

    model.load_state_dict(best_state)
    model.eval()

    y_pred = []
    y_true = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            y_pred.extend(preds.cpu().numpy().tolist())
            y_true.extend(yb.numpy().tolist())

    labels_in_use = sorted(set(y_true) | set(y_pred))
    target_names = le.inverse_transform(labels_in_use)
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels_in_use,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    test_acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    print("\n=== Test Evaluation ===")
    print(f"Best epoch: {best_epoch} | Best val acc: {best_val_acc:.4f} | Test acc: {test_acc:.4f}")
    print("\nClassification Report:\n")
    print(report_text)

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_latest_path = os.path.join(output_dir, "static_model.pth")
    model_version_path = os.path.join(output_dir, f"static_model_{ts}.pth")
    le_path = os.path.join(output_dir, "static_label_encoder.pkl")
    scaler_path = os.path.join(output_dir, "static_scaler.pkl")

    torch.save(model.state_dict(), model_latest_path)
    torch.save(model.state_dict(), model_version_path)
    joblib.dump(le, le_path)
    joblib.dump(scaler, scaler_path)

    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    np.save(os.path.join(processed_dir, "X_train_static.npy"), X_train)
    np.save(os.path.join(processed_dir, "X_test_static.npy"), X_test)
    np.save(os.path.join(processed_dir, "y_train_static.npy"), y_train)
    np.save(os.path.join(processed_dir, "y_test_static.npy"), y_test)
    joblib.dump(le, os.path.join(processed_dir, "static_label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(processed_dir, "static_scaler.pkl"))

    perf_dir = os.path.join(PROJECT_ROOT, "data", "performance")
    os.makedirs(perf_dir, exist_ok=True)
    report_path = os.path.join(perf_dir, f"static_training_report_{ts}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Static Retraining Report\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {ts}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.6f}\n")
        f.write(f"Test Accuracy: {test_acc:.6f}\n")
        f.write(f"Num Classes: {num_classes}\n")
        f.write(f"Input Features: {input_dim}\n")
        f.write(f"Train Samples: {len(y_train)}\n")
        f.write(f"Validation Samples: {len(y_val)}\n")
        f.write(f"Test Samples: {len(y_test)}\n\n")
        f.write("Per-epoch History\n")
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
        for epoch, tr_loss, tr_acc, va_loss, va_acc in history:
            f.write(f"{epoch},{tr_loss:.6f},{tr_acc:.6f},{va_loss:.6f},{va_acc:.6f}\n")
        f.write("\nClassification Report\n\n")
        f.write(report_text)

    print("\nSaved artifacts:")
    print(f"- Model (latest): {model_latest_path}")
    print(f"- Model (versioned): {model_version_path}")
    print(f"- Label encoder: {le_path}")
    print(f"- Scaler: {scaler_path}")
    print(f"- Report: {report_path}")

    return {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "num_classes": num_classes,
        "num_samples": len(y),
        "model_path": model_latest_path,
        "report_path": report_path,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain static alphabet model with combined old+new data.")
    parser.add_argument(
        "--static-csv",
        default=os.path.join(PROJECT_ROOT, "data_keypoints", "static_keypoints.csv"),
        help="Path to legacy/static keypoints CSV (optional).",
    )
    parser.add_argument(
        "--image-roots",
        nargs="*",
        default=None,
        help="Image dataset roots containing label subfolders (defaults to known data roots).",
    )
    parser.add_argument("--epochs", type=int, default=80, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--min-samples", type=int, default=30, help="Min samples required per class.")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--include-non-alphabet",
        action="store_true",
        help="Include non A-Z labels found in datasets.",
    )
    args = parser.parse_args()

    train_static_model(
        static_csv=args.static_csv,
        image_roots=args.image_roots,
        alphabet_only=not args.include_non_alphabet,
        min_samples_per_class=args.min_samples,
        test_size=args.test_size,
        val_size=args.val_size,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
    )

