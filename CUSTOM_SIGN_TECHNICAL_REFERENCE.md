# Custom Sign Customization - Complete Technical Reference

## 🏗️ Architecture Overview

```
Flask Sign Language System
├── Authentication Layer (Login/Session)
├── Core Modules:
│   ├── Text-to-Sign Service (with fallback to custom)
│   ├── Sign-to-Text ML Inference (unchanged)
│   └── Dictionary Service
├── Custom Sign Layer (NEW):
│   ├── Database Models (CRUD)
│   ├── Storage Manager (file organization)
│   ├── Validator (optional ML verification)
│   └── Flask Routes (API endpoints)
└── Frontend UI (new customize page)
```

## 📂 File Organization

### Database Layer
```
database/
├── schema.py                      # Enhanced with custom_signs table
├── custom_signs_models.py         # NEW: CRUD operations
└── sqlite.py                      # Connection pool
```

### Storage Layer
```
uploads/custom_signs/             # NEW directory structure
├── user_1/
│   ├── words/
│   ├── sentences/
│   ├── letters/
│   └── numbers/
├── user_2/
│   ├── words/
│   │   └── hello.mp4
│   ├── sentences/
│   ├── letters/
│   │   ├── A.jpg
│   │   └── B.jpg
│   └── numbers/
└── user_3/
    └── ...
```

### Service Layer
```
services/
├── dictionary_service.py          # (Existing)
├── email_service.py              # (Existing)
└── [custom sign logic integrated into main routes]

utils/
├── text_to_sign_service.py       # Modified: added fallback logic
├── inference_service.py           # (Existing, used by validator)
├── custom_sign_storage.py         # NEW: File management
├── custom_sign_validator.py       # NEW: ML verification
└── [other utilities]
```

### Routes Layer
```
flask_app/routes/
├── custom_signs.py               # NEW: All custom sign endpoints
├── chat.py                        # (Existing, uses fallback logic)
├── dictionary.py                  # (Existing, uses fallback logic)
└── [other routes]
```

### Template Layer
```
flask_app/templates/custom_signs/
├── manage.html                    # Existing: List/delete interface
└── customize.html                 # NEW: Two-panel upload interface
```

## 🔄 Data Flow

### Upload Flow
```
User Opens /customize
    ↓
Select Category & Item
    ↓
Upload File (video or image)
    ↓
Optional ML Validation
    ├─ SUCCESS → Mark verified=True, confidence=score
    └─ FAIL → Mark verified=False (or skip in advisory mode)
    ↓
Save to: uploads/custom_signs/user_<id>/<category>/<word>.*
    ↓
Insert into custom_signs table with metadata
    ↓
Success Response to User
```

### Playback Flow
```
User asks for word "hello" in TEXT-TO-SIGN
    ↓
text_to_sign_service.resolve_word_video_with_custom(
    word="hello", 
    user_id=5
)
    ↓
Query: SELECT * FROM custom_signs 
       WHERE user_id=5 AND word="hello"
    ├─ FOUND → Use custom video (with metadata)
    └─ NOT FOUND → Fallback to dataset default
    ↓
Return video path and metadata to player
```

### Validation Flow (Optional)
```
Uploaded Video File
    ↓
CustomSignValidator.validate_video(path, expected_word)
    ├─ Extract Frames (using OpenCV)
    ├─ Run Inference on each frame (using DynamicLSTM)
    ├─ Get Prediction + Confidence
    └─ Compare with expected_word
    ↓
Return: {
    valid: prediction == expected_word && confidence >= threshold,
    prediction: detected_word,
    confidence: confidence_score,
    message: user_friendly_message,
    matches: prediction == expected_word,
    meets_threshold: confidence >= 0.75
}
```

## 🗄️ Database Schema Details

### custom_signs Table
```sql
CREATE TABLE custom_signs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,              -- Owner
    word TEXT NOT NULL,                    -- Sign for this word
    category TEXT DEFAULT 'words',         -- words|sentences|letters|numbers
    video_path TEXT,                       -- Path to video (for words/sentences)
    image_path TEXT,                       -- Path to image (for letters/numbers)
    verified INTEGER DEFAULT 0,            -- 0=unverified, 1=verified by ML
    confidence REAL,                       -- ML confidence score (0-1)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE(user_id, word)                  -- One custom sign per user per word
);

-- Indexes
CREATE INDEX idx_custom_signs_user ON custom_signs(user_id);
CREATE INDEX idx_custom_signs_user_word ON custom_signs(user_id, word);
```

### Query Examples

```sql
-- Add a new custom sign
INSERT INTO custom_signs (user_id, word, category, video_path, verified, confidence)
SELECT 5, 'hello', 'words', 'custom_signs/user_5/words/hello.mp4', 1, 0.85
ON CONFLICT(user_id, word) DO UPDATE SET video_path=excluded.video_path;

-- Get user's custom signs
SELECT * FROM custom_signs WHERE user_id = 5 ORDER BY word;

-- Check for custom before fallback
SELECT video_path FROM custom_signs 
WHERE user_id = 5 AND word = 'hello';

-- Get verified signs
SELECT * FROM custom_signs WHERE user_id = 5 AND verified = 1;

-- Get unverified signs
SELECT * FROM custom_signs WHERE user_id = 5 AND verified = 0;

-- Count by category
SELECT category, COUNT(*) as count 
FROM custom_signs WHERE user_id = 5 
GROUP BY category;

-- Delete all custom signs for user
DELETE FROM custom_signs WHERE user_id = 5;
```

## 🔐 Security Model

### Authentication
```python
@login_required  # Decorator on all routes
def upload_custom_sign():
    user_id = session.get('user_id')  # Authenticated user
    # ... perform operations for this user only
```

### Authorization
```python
# Before deleting, verify ownership
existing = get_custom_sign_by_id(sign_id)
if existing.get('user_id') != session['user_id']:
    return error("Unauthorized"), 403
# Safe to delete now
```

### Input Sanitization
```python
# Filename sanitization prevents directory traversal
def sanitize_word(word: str) -> str:
    word = word.lower().strip()
    word = word.replace(" ", "_")
    word = re.sub(r'[^a-z0-9_-]', '', word)  # Only safe chars
    return word

# Example: "../../etc/passwd" → "" (empty)
# Example: "hello world" → "hello_world"
# Example: "HELLO!!!" → "hello"
```

### File Validation
```python
# Type checking
ext = os.path.splitext(filename.lower())[1]
if ext not in ALLOWED_EXTENSIONS:
    return error("Invalid file type")

# Size checking
file.seek(0, 2)
size = file.tell()
if size > MAX_FILE_SIZE:
    return error("File too large")
```

### Path Safety
```python
# Paths stored as relative, always constructed safely
relative_path = f"custom_signs/user_{user_id}/{category}/{word}.mp4"
absolute_path = os.path.join(UPLOADS_DIR, relative_path)

# Prevent path traversal by ensuring path starts with UPLOADS_DIR
if not os.path.abspath(absolute_path).startswith(UPLOADS_DIR):
    return error("Invalid path")
```

## 🔧 Configuration

### File Limits
```python
# Video configuration
MAX_VIDEO_SIZE = 50 * 1024 * 1024  # 50 MB
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.webm', '.mkv', '.m4v'}

# Image configuration  
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}

# Validation configuration
CONFIDENCE_THRESHOLD = 0.75  # 75% minimum confidence required
MAX_FRAMES_TO_EXTRACT = 30  # Process first 30 frames from video
```

### Directory Configuration
```python
PROJECT_ROOT = os.path.abspath(os.path.join(...))
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "uploads")
CUSTOM_SIGNS_DIR = os.path.join(UPLOADS_DIR, "custom_signs")
```

## 🎮 API Endpoints

### GET /customize
Returns the customization page with both panels
```
Request: GET /customize
Authentication: Required (session['user_id'])
Response: HTML page with two-panel interface
Status Codes:
  - 200: Success
  - 401: Not authenticated
  - 500: Server error processing categories
```

### GET /custom-signs/api/vocabulary/\<category\>
Get available vocabulary for a category
```
Request: GET /custom-signs/api/vocabulary/words
Parameters:
  - category: words|sentences|letters|numbers
Response: {
  "success": true,
  "category": "words",
  "vocabulary": ["hello", "thank you", ...],
  "count": 42
}
```

### GET /custom-signs/api/reference/\<category\>/\<item\>
Get default reference video/image
```
Request: GET /custom-signs/api/reference/words/hello
Response: {
  "success": true,
  "media_type": "video",
  "path": "data/Frames_Word_Level/...",
  "found": true
}
```

### POST /custom-signs/upload
Upload custom sign with optional validation
```
Request: POST /custom-signs/upload
Body: multipart/form-data
  - word: "hello"
  - category: "words"
  - video or image: <file>
  - enable_validation: "true" (optional)
  - skip_validation: "true" (optional)
  - strict_validation: "false" (optional)

Response: {
  "success": true,
  "message": "Custom sign created successfully",
  "custom_sign": {
    "id": 42,
    "user_id": 5,
    "word": "hello",
    "category": "words",
    "video_path": "custom_signs/user_5/words/hello.mp4",
    "verified": 1,
    "confidence": 0.85
  },
  "validation": { ... }
}

Status Codes:
  - 200: Success
  - 400: Validation error (word not provided, file too large, etc)
  - 401: Not authenticated
  - 500: Upload failed
```

### DELETE /custom-signs/delete/\<id\>
Delete a custom sign
```
Request: DELETE /custom-signs/delete/42
Response: {
  "success": true,
  "message": "Custom sign deleted successfully"
}

Status Codes:
  - 200: Success
  - 403: Unauthorized (not owner)
  - 404: Custom sign not found
  - 500: Deletion failed
```

### GET /custom-signs/video/\<user_id\>/\<word\>
Serve custom video file
```
Request: GET /custom-signs/video/5/hello
Response: Binary video file (video/mp4)

Status Codes:
  - 200: Success
  - 403: Unauthorized (not owner)
  - 404: Sign or file not found
  - 500: Serving failed
```

## 🧠 ML Validation Details

### CustomSignValidator Class

```python
class CustomSignValidator:
    def __init__(self, inference_service, confidence_threshold=0.75):
        self.inference_service = inference_service  # Pre-trained model
        self.confidence_threshold = confidence_threshold
    
    def validate_video(self, video_path: str, expected_word: str):
        """Main validation entry point"""
        # 1. Extract frames using cv2.VideoCapture
        frames = self._extract_frames(video_path, max=30)
        
        # 2. Run inference on frames
        predictions = []
        for frame in frames:
            pred = self.inference_service.predict(frame)
            predictions.append(pred)
        
        # 3. Aggregate predictions
        most_confident = max(predictions, key=lambda x: x['confidence'])
        
        # 4. Compare with expected
        normalized_expected = self._normalize_word(expected_word)
        normalized_detected = self._normalize_word(most_confident['label'])
        
        # 5. Return result
        return {
            'valid': normalized_detected == normalized_expected and 
                     most_confident['confidence'] >= self.threshold,
            'prediction': most_confident['label'],
            'confidence': most_confident['confidence'],
            'matches': normalized_detected == normalized_expected,
            'meets_threshold': most_confident['confidence'] >= self.threshold,
            'message': self._build_message(...)
        }
```

### Frame Extraction
```python
def _extract_frames(self, video_path: str, max_frames: int = 30):
    """Extract frames from video using OpenCV"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize to standard dimensions (640x480)
        frame = cv2.resize(frame, (640, 480))
        frames.append(frame)
        frame_count += 1
    
    cap.release()
    return frames
```

### Confidence Threshold
```
Video Quality Assessment:
- confidence >= 0.90 → "Excellent match!"
- 0.75 <= confidence < 0.90 → "Good match"
- 0.50 <= confidence < 0.75 → "Partial match - low confidence"
- confidence < 0.50 → "No match detected"

User-Facing Messages:
✅ VALID: "Sign verified! Looks like you signed 'hello' correctly"
⚠️ LOW CONF: "Low confidence - try better lighting/framing"
❌ INVALID: "Sign does not match selected word. You signed 'goodbye'"
```

## 🔄 Integration Points

### 1. Text-to-Sign Service Integration
```python
# In flask_app/routes/chat.py or dictionary.py:
video_info = tts_service.resolve_word_video_with_custom(
    word="hello",
    user_id=session['user_id']
)

if video_info.get('is_custom'):
    # User's custom sign - may show verification badge
    print(f"Playing custom sign (verified: {video_info['verified']})")
else:
    # Default sign
    print("Playing default sign")
```

### 2. Chat Integration
```python
# When rendering video in multimodal chat:
# 1. Check custom_signs for this user + word
# 2. Show custom if available, otherwise show default
# 3. Display metadata indicator [CUSTOM] or [DEFAULT]
```

### 3. Dictionary Page Integration
```python
# When showing vocabulary:
# 1. Display default video
# 2. Check if user has custom version
# 3. Show"Your Sign" button if custom exists
# 4. Link to edit/delete custom
```

## 📊 Monitoring & Analytics

### Useful Queries
```sql
-- How many users created custom signs?
SELECT COUNT(DISTINCT user_id) FROM custom_signs;

-- Total custom signs across system
SELECT COUNT(*) FROM custom_signs;

-- Verification statistics
SELECT 
    verified,
    COUNT(*) as count,
    AVG(confidence) as avg_confidence
FROM custom_signs
GROUP BY verified;

-- Top 10 most customized words
SELECT word, COUNT(*) as count
FROM custom_signs
GROUP BY word
ORDER BY count DESC
LIMIT 10;

-- Average confidence by category
SELECT 
    category,
    COUNT(*) as total,
    COUNT(CASE WHEN verified=1 THEN 1 END) as verified,
    AVG(confidence) as avg_confidence
FROM custom_signs
GROUP BY category;
```

### Performance Considerations
```python
# Index-backed queries (fast)
- SELECT ... WHERE user_id = 5           # Uses idx_custom_signs_user - O(log n)
- SELECT ... WHERE user_id=5 AND word=? # Uses idx_custom_signs_user_word - O(1)

# Full table scan (slower)
- SELECT ... WHERE verified = 1          # No index
- SELECT ... WHERE confidence > 0.8      # No index

# Optimization: Add indexes if needed
CREATE INDEX idx_verified ON custom_signs(verified);
CREATE INDEX idx_confidence ON custom_signs(confidence);
```

## 🚨 Error Recovery

### File Upload Failures
```python
if upload_failed:
    # Temporary file cleanup
    try:
        os.remove(temp_path)
    except:
        pass  # Ignore cleanup errors
    
    # Database rollback (automatic with exception)
    return error("Upload failed", 500)
```

### ML Validation Failures
```python
try:
    validation_result = validator.validate_video(path, word)
except Exception as e:
    # Log but don't block upload
    logger.error(f"Validation error: {e}")
    validation_result = None  # None = skip validation
    
    if skip_validation:  # If user didn't require validation
        continue_with_upload()
    elif strict_validation:  # If user required strict validation
        return error("Validation failed", 400)
```

### Database Failures
```python
try:
    create_custom_sign(...)
except sqlite3.IntegrityError as e:
    if "UNIQUE constraint failed" in str(e):
        # User already has custom for this word - update instead
        update_custom_sign(...)
    else:
        raise
except Exception as e:
    logger.error(f"Database error: {e}")
    return error("Failed to save", 500)
```

## 🎯 Summary

This Custom Sign Customization Feature provides:

✅ **Complete Modularity** - Isolated from core system  
✅ **Data Isolation** - Multi-user safe with strong constraints  
✅ **ML Integration** - Verification without retraining  
✅ **User Experience** - Intuitive two-panel interface  
✅ **Security** - Authentication, authorization, sanitization  
✅ **Fallback Logic** - Seamless custom+default integration  
✅ **Error Handling** - Graceful degradation everywhere  
✅ **Backward Compatibility** - Zero impact on existing features  
✅ **Extensibility** - Framework for future enhancements  
✅ **Production Ready** - Tested, validated, documented  

