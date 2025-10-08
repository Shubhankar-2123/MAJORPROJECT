# test_example.py

import requests
import os
import glob

# -------------------------------
# CONFIG
# -------------------------------
FLASK_URL = "http://127.0.0.1:5000"  # make sure Flask is running
STATIC_ROUTE  = f"{FLASK_URL}/predict_static"
DYNAMIC_ROUTE = f"{FLASK_URL}/predict_dynamic"
SANITY_STATIC = f"{FLASK_URL}/sanity/static"
DEBUG_STATIC  = f"{FLASK_URL}/debug/static_preprocess_check"

# -------------------------------
# TEST STATIC ROUTE (image upload)
# -------------------------------
def find_first_image(root_dir):
    patterns = ["**/*.jpg", "**/*.jpeg", "**/*.png"]
    for p in patterns:
        matches = glob.glob(os.path.join(root_dir, p), recursive=True)
        for m in matches:
            if os.path.isfile(m):
                return m
    return None

static_image_root = os.path.join("data", "static")
static_image_path = find_first_image(static_image_root)

if not static_image_path:
    print("No static image found under:", static_image_root)
else:
    try:
        # Sanity check (GET)
        sresp = requests.get(SANITY_STATIC)
        print("/sanity/static:", sresp.status_code, sresp.text)

        # Debug preprocess check (POST, multipart)
        with open(static_image_path, "rb") as f:
            files = {"image": f}
            dresp = requests.post(DEBUG_STATIC, files=files)
        print("/debug/static_preprocess_check:", dresp.status_code, dresp.text)

        # Actual prediction (POST, multipart)
        with open(static_image_path, "rb") as f:
            files = {"image": f}
            presp = requests.post(STATIC_ROUTE, files=files)
        if presp.status_code == 200:
            print("Static Prediction:", presp.json())
        else:
            print("Static route failed:", presp.status_code, presp.text)
    except requests.exceptions.RequestException as e:
        print("Error connecting to static route:", e)

# -------------------------------
# TEST DYNAMIC ROUTE
# -------------------------------
# Replace with an actual small test video, or try to pick one
video_path = r"C:\Users\Shubhankar\OneDrive\Desktop\MAJORPROJECT\data\dynamic\i do not agree\agree (3).MP4"
if not os.path.exists(video_path):
    candidates = glob.glob(os.path.join("data", "dynamic", "**", "*.mp4"), recursive=True)
    video_path = candidates[0] if candidates else None

if not video_path or not os.path.exists(video_path):
    print("Dynamic test video not found at:", video_path)
else:
    try:
        with open(video_path, "rb") as f:
            files = {"video": f}
            response = requests.post(DYNAMIC_ROUTE, files=files)

        if response.status_code == 200:
            print("Dynamic Prediction:", response.json())
        else:
            print("Dynamic route failed:", response.status_code, response.text)

    except requests.exceptions.RequestException as e:
        print("Error connecting to dynamic route:", e)
