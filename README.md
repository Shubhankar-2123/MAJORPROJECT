# Sign Language Recognition and Translation System

A real-time Sign Language Recognition and Translation System built using Flask, OpenCV, MediaPipe, and Deep Learning techniques. The system recognizes static signs from images and dynamic signs from videos, translates them into text, and supports text-to-speech output to improve communication between sign language users and non-signers.

---

## Project Overview

This project aims to bridge communication barriers by providing an AI-powered platform capable of recognizing sign language gestures in real time. The system supports both static hand signs and dynamic gestures representing words and sentences.

The application includes user authentication, custom sign creation, sign dictionaries, feedback mechanisms, text-to-speech integration, and a modern web interface.

---

## Key Features

### Recognition & Translation

* Static sign recognition from images
* Dynamic sign recognition from videos
* Word-level and sentence-level gesture recognition
* Real-time prediction and translation
* Confidence-based prediction filtering

### User Features

* User registration and authentication
* User profile management
* Dashboard for tracking usage
* Custom sign creation and management
* Interactive sign dictionary

### Accessibility

* Text-to-Speech (TTS) integration
* Visual confidence feedback
* User-friendly web interface

### Machine Learning

* Deep Learning based gesture classification
* MediaPipe landmark extraction
* Multi-model architecture for large vocabulary support
* Dataset preprocessing and augmentation pipeline

---

## Technologies Used

### Backend

* Python
* Flask

### Machine Learning & Computer Vision

* PyTorch
* OpenCV
* MediaPipe
* NumPy
* Pandas
* Scikit-learn

### Database

* SQLite

### Frontend

* HTML
* CSS
* JavaScript
* Tailwind CSS

### Development Tools

* Git
* GitHub
* VS Code

---

## System Architecture

### Static Sign Recognition Pipeline

Image Input
→ MediaPipe Landmark Extraction
→ Feature Preprocessing
→ Static Classification Model
→ Predicted Sign

### Dynamic Sign Recognition Pipeline

Video Input
→ Frame Extraction
→ MediaPipe Landmark Extraction
→ Sequence Processing
→ Dynamic Classification Model
→ Predicted Word / Sentence

---

## Multi-Model Architecture

Due to the large size of the dynamic sign language dataset and hardware limitations during training, the system uses a multi-model architecture.

Instead of training one large model, the dataset was divided into smaller subsets containing groups of words and sentences. Each subset was trained independently, producing specialized models and label encoders.

During inference:

1. Input video is processed using MediaPipe.
2. Keypoints are extracted and normalized.
3. The routing system selects the appropriate model.
4. The selected model performs prediction.
5. The result is translated into text and optionally converted into speech.

This architecture enabled support for a larger vocabulary while reducing memory and computation requirements during training.

---

## Challenges Solved

### Large Dataset Training

The complete dataset could not be efficiently trained as a single model on available hardware.

Solution:

* Partitioned dataset into multiple groups
* Trained specialized models for each group
* Developed model routing logic during inference

### Real-Time Processing

Maintaining acceptable prediction speed while processing videos and extracting landmarks.

Solution:

* Optimized preprocessing pipeline
* Reduced unnecessary frame processing
* Used confidence-based filtering

### Consistent Preprocessing

Ensuring training and inference pipelines remained identical.

Solution:

* Shared preprocessing utilities
* Saved encoders and scalers
* Implemented reusable preprocessing services

---

## Project Structure

```text
SignLanguageRecognition/
│
├── app.py
├── config.py
├── requirements.txt
│
├── flask_app/
├── routes/
├── services/
├── utils/
├── templates/
├── static/
├── database/
│
├── models/
│   ├── static/
│   ├── words/
│   ├── sentences/
│   └── dynamic/
│
├── scripts/
├── docs/
│
└── data_keypoints/
```

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd SignLanguageRecognition
```

### Create Virtual Environment

```bash
python -m venv .venv
```

Activate environment:

Windows

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## Environment Configuration

Create a `.env` file using the following variables:

```env
DATA_DIR=data/
MODELS_DIR=models/
USER_SIGNS_DIR=data/custom_signs/

SECRET_KEY=your_secret_key

MAX_FRAMES=30

WORD_CONF_THRESHOLD=0.55
SENT_CONF_THRESHOLD=0.75
DYNAMIC_DEFAULT_THRESHOLD=0.60
```

---

## Training Models

### Static Model

Generate preprocessing artifacts:

```bash
python scripts/preprocess_data.py
```

Train model:

```bash
python scripts/train_static_model.py
```

Output:

```text
models/static/static_model.pth
```

### Dynamic Model

Train dynamic model:

```bash
python scripts/train_dynamic_new.py
```

Outputs:

* Dynamic model weights
* Label encoders
* Training artifacts

---

## Available API Endpoints

| Method | Endpoint         | Description             |
| ------ | ---------------- | ----------------------- |
| GET    | /                | Redirect to login       |
| GET    | /login           | Login page              |
| GET    | /app             | Main application        |
| GET    | /dashboard       | User dashboard          |
| POST   | /predict_static  | Static sign prediction  |
| POST   | /predict_dynamic | Dynamic sign prediction |
| POST   | /predict         | Unified prediction      |
| GET    | /available_words | Available vocabulary    |
| GET    | /api/session     | Session information     |

---

## Performance Considerations

* Supports CPU and GPU execution
* Automatic CUDA detection
* Confidence threshold tuning
* Structured logging for debugging
* Safe file handling for concurrent requests

---

## Future Improvements

* Transformer-based sequence models
* Larger vocabulary support
* Real-time webcam sentence recognition
* Multilingual translation support
* Mobile application deployment
* Cloud-based inference API

---

## Documentation

Additional documentation is available in:

* docs/
* CUSTOM_SIGNS_QUICKSTART.md
* CUSTOM_SIGN_TECHNICAL_REFERENCE.md
* IMPLEMENTATION_AUDIT.md
* WORD_MODEL_CNN_LSTM_RETRAIN_GUIDE.md

---

## Author

Shubhankar Sawant

Bachelor of Engineering (Information Technology)

Passionate about Software Development, Machine Learning, Computer Vision, and Building Real-World Applications.
