import os
import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

# ------------------------------
# Mediapipe setup
# ------------------------------
mp_hands = mp.solutions.hands

# ------------------------------
# Extract keypoints from one frame (both hands)
# ------------------------------
def extract_hand_keypoints(image, hands_model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_model.process(image_rgb)

    # Initialize both hands with zeros
    hand0 = [0.0] * 21 * 3
    hand1 = [0.0] * 21 * 3

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            if idx == 0:
                hand0 = coords
            elif idx == 1:
                hand1 = coords
            # Ignore hands > 2

    return hand0 + hand1  # 126 features

# ------------------------------
# Process static images
# ------------------------------
def process_static_images(data_dir, output_csv):
    data = []
    hands_model = mp_hands.Hands(static_image_mode=True,
                                 max_num_hands=2,
                                 min_detection_confidence=0.5)

    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue
        for file in tqdm(os.listdir(label_path), desc=f"Processing {label}"):
            if not file.lower().endswith(('.jpg', '.png')):
                continue
            filepath = os.path.join(label_path, file)
            img = cv2.imread(filepath)
            keypoints = extract_hand_keypoints(img, hands_model)
            data.append([label, 'image', file, -1] + keypoints)

    # Create column names
    columns = ['label', 'type', 'file', 'frame'] + \
              [f'h0_{i}_{coord}' for i in range(21) for coord in ['x', 'y', 'z']] + \
              [f'h1_{i}_{coord}' for i in range(21) for coord in ['x', 'y', 'z']]

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"✅ Static keypoints saved to {output_csv}")

# ------------------------------
# Process dynamic videos
# ------------------------------
def process_dynamic_videos(data_dir, output_csv, max_frames=30):
    data = []
    hands_model = mp_hands.Hands(static_image_mode=False,
                                 max_num_hands=2,
                                 min_detection_confidence=0.5)

    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue
        for file in tqdm(os.listdir(label_path), desc=f"Processing {label}"):
            if not file.lower().endswith(('.mp4', '.avi', '.mov')):
                continue
            filepath = os.path.join(label_path, file)
            cap = cv2.VideoCapture(filepath)
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret or frame_count >= max_frames:
                    break
                keypoints = extract_hand_keypoints(frame, hands_model)
                data.append([label, 'video', file, frame_count] + keypoints)
                frame_count += 1
            cap.release()

    # Create column names
    columns = ['label', 'type', 'file', 'frame'] + \
              [f'h0_{i}_{coord}' for i in range(21) for coord in ['x', 'y', 'z']] + \
              [f'h1_{i}_{coord}' for i in range(21) for coord in ['x', 'y', 'z']]

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"✅ Dynamic keypoints saved to {output_csv}")

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    # Change paths to your dataset folders
    STATIC_DIR = "data/static"
    STATIC_CSV = "data_keypoints/static_keypoints.csv"
    DYNAMIC_DIR = "data/dynamic"
    DYNAMIC_CSV = "data_keypoints/dynamic_keypoints.csv"

    print("🔹 Extracting static image keypoints...")
    process_static_images(STATIC_DIR, STATIC_CSV)

    print("🔹 Extracting dynamic video keypoints...")
    process_dynamic_videos(DYNAMIC_DIR, DYNAMIC_CSV, max_frames=30)


# import os
# import cv2
# import mediapipe as mp
# import pandas as pd
# from tqdm import tqdm

# # ------------------------------
# # Mediapipe setup
# # ------------------------------
# mp_hands = mp.solutions.hands

# # ------------------------------
# # Extract keypoints from one frame (both hands)
# # ------------------------------
# def extract_hand_keypoints(image, hands_model):
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = hands_model.process(image_rgb)

#     hand0 = [0.0] * 21 * 3
#     hand1 = [0.0] * 21 * 3

#     if results.multi_hand_landmarks:
#         for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
#             coords = []
#             for lm in hand_landmarks.landmark:
#                 coords.extend([lm.x, lm.y, lm.z])
#             if idx == 0:
#                 hand0 = coords
#             elif idx == 1:
#                 hand1 = coords

#     return hand0 + hand1  # 126 features


# # ------------------------------
# # Process dynamic videos (APPEND MODE)
# # ------------------------------
# def process_dynamic_videos(data_dir, output_csv, max_frames=30):
#     data = []

#     hands_model = mp_hands.Hands(
#         static_image_mode=False,
#         max_num_hands=2,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5
#     )

#     for label in os.listdir(data_dir):
#         label_path = os.path.join(data_dir, label)
#         if not os.path.isdir(label_path):
#             continue

#         for file in tqdm(os.listdir(label_path), desc=f"Processing {label}"):
#             if not file.lower().endswith(('.mp4', '.avi', '.mov')):
#                 continue

#             filepath = os.path.join(label_path, file)
#             cap = cv2.VideoCapture(filepath)
#             frame_count = 0

#             while frame_count < max_frames:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 keypoints = extract_hand_keypoints(frame, hands_model)
#                 data.append([label, 'video', file, frame_count] + keypoints)
#                 frame_count += 1

#             cap.release()

#     hands_model.close()

#     # Column names (126 hand features)
#     columns = (
#         ['label', 'type', 'file', 'frame'] +
#         [f'h0_{i}_{c}' for i in range(21) for c in ['x', 'y', 'z']] +
#         [f'h1_{i}_{c}' for i in range(21) for c in ['x', 'y', 'z']]
#     )

#     df = pd.DataFrame(data, columns=columns)

#     # ✅ APPEND instead of overwrite
#     if os.path.exists(output_csv):
#         df.to_csv(output_csv, mode='a', header=False, index=False)
#         print(f"➕ Appended dynamic keypoints to {output_csv}")
#     else:
#         df.to_csv(output_csv, index=False)
#         print(f"✅ Dynamic keypoints saved to {output_csv}")


# # ------------------------------
# # Run only dynamic extraction
# # ------------------------------
# if __name__ == "__main__":
#     DYNAMIC_DIR = "data/dynamic_1"
#     DYNAMIC_CSV = "data_keypoints/dynamic_keypoints.csv"

#     print("🔹 Extracting dynamic video keypoints...")
#     process_dynamic_videos(DYNAMIC_DIR, DYNAMIC_CSV, max_frames=30)
