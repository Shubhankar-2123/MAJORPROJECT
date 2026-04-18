# Word Model Retraining Guide (CNN + LSTM)

This guide uses the new script:
- scripts/train_word_cnn_lstm.py

## What It Improves

1. Dataset audit for class balance and diversity
2. MediaPipe **Hands** landmark extraction (2 hands, 126 features/frame)
3. Sequence normalization + fixed-length timeline (default 30 frames)
4. Stronger augmentation:
- Horizontal flip
- Gaussian noise
- Slight XY rotation
- Temporal jitter
5. CNN + LSTM architecture for temporal sequence learning
6. Train/validation split with metrics and early stopping
7. Saves model, label encoder, metadata, and report

## Current Audit Result (from your workspace)

- Total videos: 776
- Word classes: 114
- Classes below 20 samples: 114
- Classes below 30 samples: 114
- Classes with low user diversity: 114

Meaning: dataset growth is mandatory for strong accuracy gains.

## Data Requirements (Target)

For each word class:
- Minimum 20 videos (acceptable)
- Recommended 30+ videos (better)
- At least 2-3 different users
- Consistent framing: both hands visible where needed
- Similar lighting/background quality

## Run Commands

Use your configured Python environment executable.

### 1. Audit only (fast)
```powershell
& "e:/MAJORPROJECT/MAJOR PROJECT/MAJOR PROJECT DATA/venv/Scripts/python.exe" scripts/train_word_cnn_lstm.py --audit-only --min-samples 20 --recommended-samples 30
```

### 2. Full training (default settings)
```powershell
& "e:/MAJORPROJECT/MAJOR PROJECT/MAJOR PROJECT DATA/venv/Scripts/python.exe" scripts/train_word_cnn_lstm.py --epochs 35 --seq-len 30 --min-samples 20 --augment-per-video 3
```

### 3. Longer training for larger dataset
```powershell
& "e:/MAJORPROJECT/MAJOR PROJECT/MAJOR PROJECT DATA/venv/Scripts/python.exe" scripts/train_word_cnn_lstm.py --epochs 50 --seq-len 30 --min-samples 20 --augment-per-video 4 --batch-size 16
```

### 4. Balanced training (recommended for class-collapse prevention)
```powershell
& "e:/MAJORPROJECT/MAJOR PROJECT/MAJOR PROJECT DATA/venv/Scripts/python.exe" scripts/train_word_cnn_lstm.py --epochs 40 --seq-len 30 --min-samples 20 --augment-per-video 3 --balance-train
```

### 5. Debug repeated class predictions
```powershell
& "e:/MAJORPROJECT/MAJOR PROJECT/MAJOR PROJECT DATA/venv/Scripts/python.exe" scripts/debug_word_predictions.py --max-files 10
```

This command checks:
- model output dimension vs label encoder class count
- top-3 probabilities per input video
- whether predicted index changes across diverse labels

## Output Artifacts

Saved under models/words:
- word_cnn_lstm_model_latest.pth
- word_cnn_lstm_model_<timestamp>.pth
- word_cnn_lstm_label_encoder_latest.pkl
- word_cnn_lstm_label_encoder_<timestamp>.pkl
- word_model_metadata_<timestamp>.json

Saved under data/performance:
- word_dataset_audit_<timestamp>.json
- word_training_report_<timestamp>.txt

## Notes

- This CNN+LSTM model is trained with hand landmarks and should be paired with the same preprocessing in inference.
- If you want direct app usage, update inference loading/preprocessing to this architecture and feature format.
- Confidence rejection is already enforced in runtime using WORD_CONF_THRESHOLD in config.py. Increase this value to make predictions stricter.
