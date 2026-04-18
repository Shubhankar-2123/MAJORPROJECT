"""
Custom Signs Routes
Handles user custom sign video upload, management, and serving.
"""

import os
from flask import Blueprint, request, jsonify, render_template, send_file, session, redirect, url_for
from functools import wraps
import sys
import tempfile

# Add parent directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Import database operations
from database.custom_signs_models import (
    create_custom_sign,
    get_custom_sign,
    get_custom_sign_by_id,
    list_custom_signs_for_user,
    list_custom_signs_by_category,
    delete_custom_sign,
    get_custom_signs_count,
    update_custom_sign_verification,
)
from utils.custom_sign_storage import CustomSignStorage
from utils.custom_sign_validator import get_custom_sign_validator
from utils.text_to_sign_service import TextToSignService
from services.dictionary_service import build_dictionary_entries
from config import DATA_DIR, USER_SIGNS_DIR

# Create blueprint
custom_signs_bp = Blueprint('custom_signs', __name__, url_prefix='/custom-signs')

# Configuration
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {'.mp4', '.mov', '.avi', '.webm', '.mkv', '.m4v'}

# Initialize storage utility
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "uploads")
storage = CustomSignStorage(UPLOADS_DIR)


def _get_phrase_base_dirs() -> list:
    base_dirs = []
    primary = os.path.join(DATA_DIR, "dynamic")
    if os.path.isdir(primary):
        base_dirs.append(primary)
    try:
        for name in os.listdir(DATA_DIR):
            if not name.lower().startswith("dynamic"):
                continue
            candidate = os.path.join(DATA_DIR, name)
            if os.path.isdir(candidate):
                base_dirs.append(candidate)
    except Exception:
        pass
    # Keep order but remove duplicates
    seen = set()
    ordered = []
    for d in base_dirs:
        if d in seen:
            continue
        seen.add(d)
        ordered.append(d)
    return ordered


def _get_dictionary_entries() -> list:
    phrase_dirs = _get_phrase_base_dirs()
    frame_word_dirs = [
        os.path.join(DATA_DIR, name)
        for name in sorted(os.listdir(DATA_DIR))
        if name.startswith("Frames_Word_Level") and os.path.isdir(os.path.join(DATA_DIR, name))
    ]
    user_signs_dir = USER_SIGNS_DIR
    return build_dictionary_entries(phrase_dirs, frame_word_dirs, user_signs_dir)


def login_required(f):
    """Decorator to require login for routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated_function


def validate_video_file(file):
    """Validate uploaded video file."""
    if not file or file.filename == '':
        return False, "No file selected"
    
    filename = file.filename.lower()
    ext = os.path.splitext(filename)[1]
    
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # Check file size (if possible)
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    
    if size > MAX_FILE_SIZE:
        return False, f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)} MB"
    
    return True, None


@custom_signs_bp.route('/customize', methods=['GET'])
@login_required
def customize_page():
    """
    Render the comprehensive custom sign customization page.
    Two-panel layout with reference (left) and upload (right).
    """
    user_id = session.get('user_id')
    
    try:
        entries = _get_dictionary_entries()
        
        # Build vocabulary lists for each category
        categories = {
            'words': {
                'label': 'Words',
                'icon': 'fa-font',
                'description': 'Single or multi-word phrases'
            },
            'sentences': {
                'label': 'Sentences',
                'icon': 'fa-paragraph',
                'description': 'Complete sentences'
            },
            'letters': {
                'label': 'Letters',
                'icon': 'fa-a',
                'description': 'A-Z alphabet'
            },
            'numbers': {
                'label': 'Numbers',
                'icon': 'fa-123',
                'description': '0-9 digits'
            }
        }
        
        # Get available vocabulary based on dictionary entries
        available_vocab = {}
        words = sorted({e.get("word", "") for e in entries if e.get("type") == "word" and e.get("word")})
        sentences = sorted({e.get("word", "") for e in entries if e.get("type") == "sentence" and e.get("word")})
        available_vocab['words'] = words[:200]
        
        # For letters: A-Z
        available_vocab['letters'] = [chr(i) for i in range(65, 91)]  # A-Z
        
        # For numbers: 0-9
        available_vocab['numbers'] = [str(i) for i in range(10)]  # 0-9
        
        # For sentences from dictionary entries
        available_vocab['sentences'] = sentences[:200]
        
        # Get user's existing custom signs
        user_custom_signs = list_custom_signs_for_user(user_id)
        custom_signs_map = {sign['word']: sign for sign in user_custom_signs}
        
        return render_template(
            'custom_signs/customize.html',
            categories=categories,
            available_vocab=available_vocab,
            custom_signs=custom_signs_map
        )
    except Exception as e:
        print(f"Error loading customize page: {e}")
        return render_template(
            'custom_signs/customize.html',
            categories={},
            available_vocab={},
            custom_signs={},
            error=str(e)
        )


@custom_signs_bp.route('/', methods=['GET'])
@login_required
def manage_custom_signs():
    """Render the custom signs management page."""
    user_id = session.get('user_id')
    custom_signs = list_custom_signs_for_user(user_id)
    return render_template('custom_signs/manage.html', custom_signs=custom_signs)


@custom_signs_bp.route('/list', methods=['GET'])
@login_required
def list_custom_signs():
    """API endpoint to list user's custom signs."""
    user_id = session.get('user_id')
    custom_signs = list_custom_signs_for_user(user_id)
    return jsonify({
        "success": True,
        "count": len(custom_signs),
        "custom_signs": custom_signs
    })


@custom_signs_bp.route('/api/reference/<category>/<item>', methods=['GET'])
@login_required
def get_reference_media(category, item):
    """
    Get the default reference video/image for a category item.
    Left panel uses this to show user what to replicate.
    """
    try:
        # For words/sentences: serve from dictionary entries
        if category in ['words', 'sentences']:
            entries = _get_dictionary_entries()
            normalized_item = (item or "").strip().lower()
            target_type = "word" if category == "words" else "sentence"
            match = next(
                (
                    e for e in entries
                    if (e.get("type") == target_type)
                    and (e.get("word", "").strip().lower() == normalized_item)
                ),
                None,
            )
            if match and match.get("url"):
                return jsonify({
                    "success": True,
                    "media_type": "video",
                    "path": match.get("url"),
                    "found": True
                })
            
            return jsonify({
                "success": False,
                "error": f"{item} not found in default dataset",
                "found": False
            }), 404
        
        # For letters/numbers: serve from static dataset images
        elif category in ['letters', 'numbers']:
            from utils.text_to_sign_service import TextToSignService
            tts_service = TextToSignService()
            
            # Try to find image for letter/number
            # Assuming static images are stored in data/static_data/ or similar
            image_path = f"data/static_images/{category}/{item.upper()}.jpg"
            
            return jsonify({
                "success": True,
                "media_type": "image",
                "path": image_path,
                "found": True
            })
        
        return jsonify({
            "success": False,
            "error": "Invalid category"
        }), 400
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@custom_signs_bp.route('/api/vocabulary/<category>', methods=['GET'])
@login_required
def get_vocabulary(category):
    """
    Get available vocabulary for a category from the dataset.
    Used to populate the searchable dropdown in the left panel.
    """
    try:
        entries = _get_dictionary_entries()
        
        if category == 'letters':
            vocab = [chr(i) for i in range(65, 91)]  # A-Z
        elif category == 'numbers':
            vocab = [str(i) for i in range(10)]  # 0-9
        elif category == 'words':
            vocab = sorted({e.get("word", "") for e in entries if e.get("type") == "word" and e.get("word")})
            vocab = vocab[:200]
        elif category == 'sentences':
            vocab = sorted({e.get("word", "") for e in entries if e.get("type") == "sentence" and e.get("word")})
            vocab = vocab[:200]
        else:
            vocab = []
        
        return jsonify({
            "success": True,
            "category": category,
            "vocabulary": vocab,
            "count": len(vocab)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@custom_signs_bp.route('/check/<word>', methods=['GET'])
@login_required
def check_custom_sign(word):
    """Check if a custom sign exists for a word."""
    user_id = session.get('user_id')
    word = word.strip().lower()
    
    custom_sign = get_custom_sign(user_id, word)
    
    return jsonify({
        "exists": bool(custom_sign),
        "custom_sign": custom_sign if custom_sign else None
    })


@custom_signs_bp.route('/validate', methods=['POST'])
@login_required
def validate_video():
    """
    Optional: Validate a custom sign video before upload.
    Runs ML inference to verify the video matches the claimed word.
    """
    user_id = session.get('user_id')
    
    # Get configuration
    enable_validation = request.form.get('enable_validation', 'true').lower() == 'true'
    
    if not enable_validation:
        return jsonify({
            "success": True,
            "message": "Validation disabled",
            "valid": True,
            "validation_enabled": False
        })
    
    if 'word' not in request.form:
        return jsonify({"error": "Word is required"}), 400
    
    if 'video' not in request.files:
        return jsonify({"error": "Video file is required"}), 400
    
    word = request.form['word'].strip().lower()
    video_file = request.files['video']
    
    # Validate word
    if not word or len(word) == 0:
        return jsonify({"error": "Invalid word"}), 400
    
    if len(word) > 100:
        return jsonify({"error": "Word too long (max 100 characters)"}), 400
    
    # Validate file
    is_valid, error_msg = validate_video_file(video_file)
    if not is_valid:
        return jsonify({"error": error_msg}), 400
    
    try:
        # Save temporary file for validation
        temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
        os.close(temp_fd)
        
        video_file.save(temp_path)
        
        # Run validation
        validator = get_custom_sign_validator(enabled=True)
        validation_result = validator.validate_video(temp_path, word)
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
        
        return jsonify({
            "success": True,
            "validation": validation_result,
            "validation_enabled": True
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Validation failed: {str(e)}"}), 500


@custom_signs_bp.route('/upload', methods=['POST'])
@login_required
def upload_custom_sign():
    """
    Upload a custom sign (video or image) for a specific word/letter/number.
    Supports optional AI verification before saving.
    """
    user_id = session.get('user_id')
    
    # Get parameters
    word = request.form.get('word', '').strip().lower()
    category = request.form.get('category', 'words').lower()
    skip_validation = request.form.get('skip_validation', 'true').lower() == 'true'
    strict_validation = request.form.get('strict_validation', 'false').lower() == 'true'
    
    # Validate inputs
    if not word:
        return jsonify({"error": "Word is required"}), 400
    
    if len(word) > 100:
        return jsonify({"error": "Word too long (max 100 characters)"}), 400
    
    if category not in ['words', 'sentences', 'letters', 'numbers']:
        return jsonify({"error": "Invalid category"}), 400
    
    # Determine if we're uploading video or image
    if category in ['words', 'sentences']:
        # Video upload
        file_key = 'video'
        file_type = 'video'
    else:
        # Image upload
        file_key = 'image'
        file_type = 'image'
    
    if file_key not in request.files:
        return jsonify({"error": f"{file_type.capitalize()} file is required"}), 400
    
    uploaded_file = request.files[file_key]
    
    try:
        # Validate file
        if file_type == 'video':
            is_valid, error_msg = storage.validate_video_file(uploaded_file.filename, 0)
        else:
            is_valid, error_msg = storage.validate_image_file(uploaded_file.filename, 0)
        
        if not is_valid:
            return jsonify({"error": error_msg}), 400
        
        # Optional ML validation
        validation_result = None
        if not skip_validation and file_type == 'video':
            temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
            os.close(temp_fd)
            
            try:
                uploaded_file.save(temp_path)
                validator = get_custom_sign_validator(enabled=True)
                validation_result = validator.validate_video(temp_path, word)
                
                # In strict mode, reject if validation failed
                if strict_validation and not validation_result.get('valid'):
                    return jsonify({
                        "success": False,
                        "error": validation_result.get('message'),
                        "validation": validation_result
                    }), 400
                
                # Reset file pointer for saving
                uploaded_file.seek(0)
            except Exception as val_error:
                print(f"Validation error: {val_error}")
            finally:
                try:
                    os.remove(temp_path)
                except:
                    pass
        
        # Check if custom sign already exists
        existing = get_custom_sign(user_id, word)
        
        # Save file and get relative path
        if file_type == 'video':
            relative_path = storage.save_video(uploaded_file, user_id, word, category)
            image_path = None
        else:
            relative_path = storage.save_image(uploaded_file, user_id, word, category)
            image_path = relative_path
            relative_path = None
        
        # Save to database with verification info
        verified = False
        confidence = None
        if validation_result:
            verified = validation_result.get('valid', False)
            confidence = validation_result.get('confidence', None)
        
        custom_sign = create_custom_sign(
            user_id=user_id,
            word=word,
            category=category,
            video_path=relative_path,
            image_path=image_path,
            verified=verified,
            confidence=confidence
        )
        
        action = "updated" if existing else "created"
        
        return jsonify({
            "success": True,
            "message": f"Custom {file_type} {action} successfully",
            "custom_sign": custom_sign,
            "action": action,
            "validation": validation_result
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@custom_signs_bp.route('/delete/<int:sign_id>', methods=['DELETE', 'POST'])
@login_required
def delete_custom_sign_route(sign_id):
    """Delete a custom sign."""
    user_id = session.get('user_id')
    
    try:
        # Get the sign to retrieve video path
        sign = get_custom_sign_by_id(sign_id)
        
        if not sign:
            return jsonify({"error": "Custom sign not found"}), 404
        
        # Verify ownership
        if sign.get('user_id') != user_id:
            return jsonify({"error": "Unauthorized"}), 403
        
        # Delete from database
        deleted = delete_custom_sign(sign_id, user_id)
        
        if not deleted:
            return jsonify({"error": "Failed to delete custom sign"}), 500
        
        # Delete video file
        video_path = sign.get('video_path')
        if video_path:
            storage.delete_by_path(video_path)
        
        return jsonify({
            "success": True,
            "message": "Custom sign deleted successfully"
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Delete failed: {str(e)}"}), 500


@custom_signs_bp.route('/video/<int:user_id>/<word>', methods=['GET'])
def serve_custom_video(user_id, word):
    """Serve a custom sign video file."""
    try:
        # Security: Verify user can access this video
        session_user_id = session.get('user_id')
        
        # Only allow users to access their own custom videos
        if session_user_id != user_id:
            return jsonify({"error": "Unauthorized"}), 403
        
        # Get custom sign
        custom_sign = get_custom_sign(user_id, word)
        
        if not custom_sign:
            return jsonify({"error": "Custom sign not found"}), 404
        
        # Get video path
        video_path = custom_sign.get('video_path')
        if not video_path:
            return jsonify({"error": "Video path not found"}), 404
        
        abs_path = os.path.join(UPLOADS_DIR, video_path)
        
        if not os.path.exists(abs_path):
            return jsonify({"error": "Video file not found"}), 404
        
        return send_file(abs_path, mimetype='video/mp4')
        
    except Exception as e:
        return jsonify({"error": f"Failed to serve video: {str(e)}"}), 500