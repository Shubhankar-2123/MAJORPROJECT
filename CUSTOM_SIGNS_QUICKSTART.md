# 🚀 Custom Signs - Quick Start Guide

## One-Minute Setup

### 1. Register Blueprint
Add to `flask_app/app.py`:
```python
from routes.custom_signs import custom_signs_bp
app.register_blueprint(custom_signs_bp)
```

### 2. Restart App
```bash
python flask_app/app.py
```
*(Migration runs automatically)*

### 3. Navigate
Open browser: `http://localhost:5000/custom-signs/`

### 4. Upload
- Word: "test"
- Video: Choose MP4 file
- Click Upload

---

## Files Created

```
✅ database/migrate_custom_signs.sql
✅ database/custom_signs_models.py
✅ utils/custom_sign_storage.py
✅ flask_app/routes/custom_signs.py
✅ flask_app/templates/custom_signs/manage.html
```

## Files Modified

```
🔧 database/schema.py (added custom_signs table)
🔧 utils/text_to_sign_service.py (added fallback logic)
```

---

## Key Functions

### Upload Custom Sign
```python
POST /custom-signs/upload
Form: word, video (file)
```

### Use Custom Sign
```python
from utils.text_to_sign_service import TextToSignService
tts = TextToSignService()
video = tts.resolve_word_video_with_custom("hello", user_id)
```

### List Custom Signs
```python
from database.custom_signs_models import list_custom_signs_for_user
signs = list_custom_signs_for_user(user_id)
```

---

## Security Checklist

- ✅ Login required
- ✅ User ownership verified
- ✅ File type validated (mp4, mov, avi, webm, mkv)
- ✅ Size limit (50 MB)
- ✅ Filename sanitized
- ✅ Path injection prevented

---

## Testing Commands

```bash
# Check database
sqlite3 database.db "SELECT * FROM custom_signs;"

# Check uploads folder
ls uploads/custom_signs/user_*/

# Test upload via curl
curl -X POST http://localhost:5000/custom-signs/upload \
  -H "Cookie: session=..." \
  -F "word=hello" \
  -F "video=@test.mp4"
```

---

## Docs

- **`CUSTOM_SIGNS_SUMMARY.md`** - Complete feature overview
- **`CUSTOM_SIGNS_IMPLEMENTATION_GUIDE.md`** - Technical details
- **`CUSTOM_SIGNS_VIVA_GUIDE.md`** - Q&A for viva
- **`CUSTOM_SIGNS_INTEGRATION.md`** - Step-by-step setup

---

## Viva One-Liner

> "We implemented a user-specific custom sign video feature with database isolation, file storage management, and automatic fallback to defaults—all without affecting the ML models or other users."

---

**That's it!** 🎉
