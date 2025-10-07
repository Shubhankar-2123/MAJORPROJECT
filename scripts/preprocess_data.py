# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# import os

# # ------------------------------
# # Process static dataset
# # ------------------------------
# def process_static(static_csv, output_dir, min_samples=10, test_size=0.2, random_state=42):
#     df = pd.read_csv(static_csv, dtype={'label': str}, low_memory=False)

#     # Only images
#     df = df[df['type'] == 'image']

#     # Filter classes with too few samples
#     class_counts = df['label'].value_counts()
#     valid_labels = class_counts[class_counts >= min_samples].index
#     df = df[df['label'].isin(valid_labels)]

#     # Extract features and labels
#     X = df.drop(['label','type','file','frame'], axis=1).values
#     y = df['label'].astype(str).values

#     # Encode labels
#     le = LabelEncoder()
#     y_encoded = le.fit_transform(y)

#     # Scale features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     # Train/test split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_scaled, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
#     )

#     # Save .npy files
#     os.makedirs(output_dir, exist_ok=True)
#     np.save(os.path.join(output_dir,'X_train_static.npy'), X_train)
#     np.save(os.path.join(output_dir,'X_test_static.npy'), X_test)
#     np.save(os.path.join(output_dir,'y_train_static.npy'), y_train)
#     np.save(os.path.join(output_dir,'y_test_static.npy'), y_test)

#     print(f"✅ Static dataset processed: {X_train.shape[0]} train / {X_test.shape[0]} test samples")
#     return le, scaler

# # ------------------------------
# # Process dynamic dataset
# # ------------------------------
# def process_dynamic(dynamic_csv, output_dir, min_videos_per_class=2, test_size=0.2, random_state=42, max_frames=30):
#     """
#     Process dynamic video dataset for sign language.
#     - Ensures all classes have at least `min_videos_per_class` videos.
#     - Pads or trims sequences to max_frames.
#     - Encodes labels and splits train/test sets.
#     - Saves .npy files.
#     """
#     # Load CSV
#     df = pd.read_csv(DYNAMIC_CSV, dtype={'label': str}, low_memory=False)
#     df = df[df['type'] == 'video']

#     # Count videos per label
#     video_counts = df.groupby('label')['file'].nunique()
#     valid_labels = video_counts[video_counts >= min_videos_per_class].index
#     df = df[df['label'].isin(valid_labels)]

#     if df.empty:
#         raise ValueError("No valid classes with enough videos. Reduce min_videos_per_class or check dataset.")

#     # Group by label + file (video)
#     grouped = df.groupby(['label','file'])

#     # Collect sequences
#     X_sequences = []
#     y_labels = []

#     for (label, file), group in grouped:
#         group_sorted = group.sort_values('frame')
#         keypoints_seq = group_sorted.drop(['label','type','file','frame'], axis=1).values

#         # Pad or trim to max_frames
#         if keypoints_seq.shape[0] < max_frames:
#             pad = np.zeros((max_frames - keypoints_seq.shape[0], keypoints_seq.shape[1]))
#             keypoints_seq = np.vstack([keypoints_seq, pad])
#         elif keypoints_seq.shape[0] > max_frames:
#             keypoints_seq = keypoints_seq[:max_frames]

#         X_sequences.append(keypoints_seq)
#         y_labels.append(label)

#     # Convert to arrays
#     X = np.array(X_sequences)
#     y = np.array(y_labels).astype(str)  # ensure all labels are strings

#     # Encode labels
#     le = LabelEncoder()
#     y_encoded = le.fit_transform(y)

#     # Stratified train/test split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
#     )

#     # Save .npy files
#     os.makedirs(output_dir, exist_ok=True)
#     np.save(os.path.join(output_dir,'X_train_dynamic.npy'), X_train)
#     np.save(os.path.join(output_dir,'X_test_dynamic.npy'), X_test)
#     np.save(os.path.join(output_dir,'y_train_dynamic.npy'), y_train)
#     np.save(os.path.join(output_dir,'y_test_dynamic.npy'), y_test)

#     print(f"✅ Dynamic dataset processed: {X_train.shape[0]} train / {X_test.shape[0]} test sequences")
#     print(f"Number of classes: {len(le.classes_)}")

#     return le

# # ------------------------------
# # Main
# # ------------------------------
# if __name__ == "__main__":
#     STATIC_CSV = "data_keypoints/static_keypoints.csv"
#     DYNAMIC_CSV = "data_keypoints/dynamic_keypoints.csv"
#     OUTPUT_DIR = "data/processed"

#     print("🔹 Processing static dataset...")
#     process_static(STATIC_CSV, OUTPUT_DIR, min_samples=10)

#     print("🔹 Processing dynamic dataset...")
#     process_dynamic(DYNAMIC_CSV, OUTPUT_DIR, min_videos_per_class=5, max_frames=30)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

# ------------------------------
# Process static dataset
# ------------------------------
def process_static(static_csv, output_dir, min_samples=10, test_size=0.2, random_state=42):
    df = pd.read_csv(static_csv, dtype={'label': str}, low_memory=False)

    # Only images
    df = df[df['type'] == 'image']

    # Filter classes with too few samples
    class_counts = df['label'].value_counts()
    valid_labels = class_counts[class_counts >= min_samples].index
    df = df[df['label'].isin(valid_labels)]

    # Extract features and labels
    X = df.drop(['label', 'type', 'file', 'frame'], axis=1).values
    y = df['label'].astype(str).values  # ensure labels are strings (sentence labels)

    # Encode labels (sentence strings)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    # Save processed data and encoders
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X_train_static.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test_static.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train_static.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test_static.npy'), y_test)
    
    # Save LabelEncoder and scaler for inference
    import joblib
    joblib.dump(le, os.path.join(output_dir, 'static_label_encoder.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'static_scaler.pkl'))

    print(f"✅ Static dataset processed: {X_train.shape[0]} train / {X_test.shape[0]} test samples")
    print(f"Static classes: {le.classes_}")

    return le, scaler


# ------------------------------
# Process dynamic dataset
# ------------------------------
def process_dynamic(dynamic_csv, output_dir, min_videos_per_class=2, test_size=0.2, random_state=42, max_frames=30):
    df = pd.read_csv(dynamic_csv, dtype={'label': str}, low_memory=False)
    df = df[df['type'] == 'video']

    # Count videos per label
    video_counts = df.groupby('label')['file'].nunique()
    valid_labels = video_counts[video_counts >= min_videos_per_class].index
    df = df[df['label'].isin(valid_labels)]

    if df.empty:
        raise ValueError("No valid classes with enough videos. Reduce min_videos_per_class or check dataset.")

    # Group by label and file (video)
    grouped = df.groupby(['label', 'file'])

    X_sequences = []
    y_labels = []

    for (label, file), group in grouped:
        group_sorted = group.sort_values('frame')
        keypoints_seq = group_sorted.drop(['label', 'type', 'file', 'frame'], axis=1).values

        # Pad or trim to max_frames
        if keypoints_seq.shape[0] < max_frames:
            pad = np.zeros((max_frames - keypoints_seq.shape[0], keypoints_seq.shape[1]))
            keypoints_seq = np.vstack([keypoints_seq, pad])
        elif keypoints_seq.shape[0] > max_frames:
            keypoints_seq = keypoints_seq[:max_frames]
        
        X_sequences.append(keypoints_seq)
        y_labels.append(label)  # sentence label as string

    X = np.array(X_sequences)
    y = np.array(y_labels).astype(str)  # ensure labels are strings

    # Encode labels (sentence strings)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    # Save processed data and encoder
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X_train_dynamic.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test_dynamic.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train_dynamic.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test_dynamic.npy'), y_test)

    import pickle
    with open(os.path.join(output_dir, 'dynamic_label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)

    print(f"✅ Dynamic dataset processed: {X_train.shape[0]} train / {X_test.shape[0]} test sequences")
    print(f"Dynamic classes: {le.classes_}")

    return le


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    STATIC_CSV = "data_keypoints/static_keypoints.csv"
    DYNAMIC_CSV = "data_keypoints/dynamic_keypoints.csv"
    OUTPUT_DIR = "data/processed"

    print("🔹 Processing static dataset...")
    process_static(STATIC_CSV, OUTPUT_DIR, min_samples=10)

    print("🔹 Processing dynamic dataset...")
    process_dynamic(DYNAMIC_CSV, OUTPUT_DIR, min_videos_per_class=5, max_frames=30)
