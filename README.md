# Sign Language Recognition Project

This repository hosts a Flask-based application and supporting scripts for static (image) and dynamic (video) sign recognition, plus utilities for dataset preprocessing and model training.

## Setup

- Prerequisites
  - Python 3.8–3.11 on Windows
  - (Optional) NVIDIA GPU for faster inference with `torch`
- Create and activate a virtual environment
  - `python -m venv .venv`
  - `.\.venv\Scripts\activate`
- Install dependencies
  - `pip install -r requirements.txt`
- Run the web app
  - `python flask_app/app.py`
  - Opens on `http://127.0.0.1:5000` (host `0.0.0.0`, port `5000`)

Environment knobs
- `DYNAMIC_DEFAULT_THRESHOLD` (default `0.60`) – base confidence for dynamic models
- `WORD_CONF_THRESHOLD` (default `0.55`) – word-level threshold
- `SENT_CONF_THRESHOLD` (default `0.75`) – sentence-level threshold

## Data Layout

The app and scripts expect data under `data/`:
- `data/static/` – static image classes (e.g., `A/`, `B/`, `1/`).
- `data/dynamic*/` – one or more dynamic video datasets; subfolders are class labels, files are videos (`.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`, `.m4v`).
- `data_keypoints/` – extracted keypoint CSVs
  - `static_keypoints.csv`
  - `dynamic_keypoints.csv`
- `data/processed/` – created by preprocessing scripts (npy splits, encoders, scaler)

Models
- `models/static images/` – static models and artifacts
  - `static_model.pth`
  - `static_label_encoder.pkl`
  - `static_scaler.pkl`
- `models/words/` – word-level dynamic models and encoders
- `models/sentences/` – sentence-level dynamic models and encoders

Key scripts
- `scripts/preprocess_data.py` – build processed datasets and save encoders/scaler
- `scripts/train_static_model.py` – train static image classifier
- `scripts/train_dynamic_new.py` – train dynamic LSTM for videos (word-level)

Utilities
- `utils/preprocessing.py` – production preprocessing, with structured logging and robust temp-file handling
- `utils/keypoints_extraction.py` – dataset keypoint extraction to CSVs
- `utils/text_to_sign_service.py` – legacy text-to-sign mapping (not used by the Flask app)

## Training

Static (images)
1. Ensure `data_keypoints/static_keypoints.csv` is generated (see keypoints script if needed).
2. Build processed splits and artifacts:
   - `python scripts/preprocess_data.py`
   - Saves `static_label_encoder.pkl` and `static_scaler.pkl` into `data/processed/`.
3. Train the static model:
   - `python scripts/train_static_model.py`
   - Outputs `models/static images/static_model.pth`.

Dynamic (videos)
1. Ensure videos exist under a word-level dataset (e.g., `data/Frames_Word_Level_2/<word>/*.mp4`).
2. Train the dynamic model:
   - `python scripts/train_dynamic_new.py`
   - Saves model weights and label encoders under `models/`.

Notes
- Normalization and keypoint extraction in `utils/preprocessing.py` matches training behavior to avoid distribution shift.
- For reproducibility, keep preprocessing artifacts (scaler/encoders) consistent between train and inference.

## API Routes

Base URL: `http://127.0.0.1:5000`

- GET `/`
  - Renders `templates/index.html`.

- GET `/available_words`
  - Scans `data/dynamic*` for word folders containing videos.
  - Returns `{ "count": <int>, "words": [<str>, ...] }`.

- POST `/predict_static`
  - Form-data: `image` (file)
  - Returns `{ "prediction": <label> }` on success.
  - On low confidence or errors, returns `{ "prediction": "unable to recognize" }`.
  - Internals: uses scaler and static model under `models/static images/`.

- POST `/predict_dynamic`
  - Form-data: `video` (file)
  - Returns `{ "prediction": <label> }` if confidence clears the per-model threshold.
  - Otherwise `{ "prediction": "unable to recognize" }`.

- POST `/debug/static_preprocess_check`
  - Form-data: `image` (file)
  - Returns shapes and sanity info: expected model input size, chosen scaler path, raw static keypoints size.

- GET `/sanity/static`
  - Runs a forward pass with zeros to validate static model wiring.
  - Returns `{ "ok": true, ... }` or `{ "ok": false, "error": <msg> }`.

### Example Requests

- Static prediction
  - `curl -X POST -F "image=@path/to/file.jpg" http://127.0.0.1:5000/predict_static`

- Dynamic prediction
  - `curl -X POST -F "video=@path/to/file.mp4" http://127.0.0.1:5000/predict_dynamic`

- Available words
  - `curl http://127.0.0.1:5000/available_words`

## Logging & Errors

- Preprocessing emits structured JSON logs via Python `logging` (events like `preprocess.static.start`, `video.open_failed`). Configure logging in your launcher if you want to capture them.
- Clear errors are raised internally as `PreprocessError` with fields `code`, `message`, `details`. Flask handlers generally convert unexpected errors into `{ "prediction": "unable to recognize" }` responses.

## Tips

- File uploads: Streams are handled safely with per-request temp files to avoid collisions under concurrency.
- GPU: If CUDA is available, `torch` uses it automatically; otherwise CPU is used.
- Dataset paths: The app scans `data/dynamic*` recursively; organize word/sentence videos under those folders for discovery.