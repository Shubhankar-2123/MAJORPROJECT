from flask import Blueprint, request, jsonify, session, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from typing import Optional, Tuple
import os
import sys
import re
from datetime import datetime, timedelta
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
import secrets

# Ensure imports work when blueprint is imported directly
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

from database.models import (
    create_user, get_user_by_username, get_user_welcome_status, get_user_by_email,
    get_user_by_id, update_user_password, 
    create_password_reset_token, get_password_reset_token_by_string,
    mark_password_reset_token_used, invalidate_password_reset_tokens
)
from database.sqlite import init_db
from services.email_service import email_service

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")

# Simple session-based login_required decorator
from functools import wraps

def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"error": "Authentication required"}), 401
        return fn(*args, **kwargs)
    return wrapper


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_password(password: str) -> Tuple[bool, str]:
    """
    Validate password strength.
    
    Requirements:
    - At least 8 characters
    - At least one uppercase letter
    - At least one number
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
    
    return True, "Password is valid"


def validate_mobile(mobile: str) -> bool:
    """Validate mobile number (10 digits for India)."""
    # Remove any non-digit characters
    digits_only = re.sub(r'\D', '', mobile)
    return len(digits_only) == 10


def validate_signup_data(data: dict) -> Tuple[bool, str]:
    """Validate all signup data."""
    # Check required fields
    required_fields = ['username', 'password', 'confirm_password', 'email', 'full_name', 'mobile']
    for field in required_fields:
        if not data.get(field) or not str(data.get(field, '')).strip():
            return False, f"{field.replace('_', ' ').title()} is required"
    
    username = str(data.get('username', '')).strip()
    password = data.get('password', '')
    confirm_password = data.get('confirm_password', '')
    email = str(data.get('email', '')).strip()
    full_name = str(data.get('full_name', '')).strip()
    mobile = str(data.get('mobile', '')).strip()
    
    # Validate username (alphanumeric and underscores only, 3-20 chars)
    if len(username) < 3 or len(username) > 20:
        return False, "Username must be between 3 and 20 characters"
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return False, "Username can only contain letters, numbers, and underscores"
    
    # Validate email
    if not validate_email(email):
        return False, "Email format is invalid"
    
    # Validate password
    is_valid, msg = validate_password(password)
    if not is_valid:
        return False, msg
    
    # Validate password confirmation
    if password != confirm_password:
        return False, "Passwords do not match"
    
    # Validate full name
    if len(full_name) < 2:
        return False, "Full name must be at least 2 characters"
    
    # Validate mobile
    if not validate_mobile(mobile):
        return False, "Mobile number must be 10 digits"
    
    return True, "Validation passed"


# ============================================================================
# AUTH ROUTES
# ============================================================================

@auth_bp.route("/signup", methods=["POST"])
def signup():
    """Enhanced signup with full validation and profile data."""
    data = request.get_json(silent=True) or {}
    
    # Validate all data
    is_valid, validation_msg = validate_signup_data(data)
    if not is_valid:
        return jsonify({"error": validation_msg}), 400
    
    username = str(data.get("username", "")).strip()
    password = data.get("password", "")
    email = str(data.get("email", "")).strip()
    full_name = str(data.get("full_name", "")).strip()
    mobile = str(data.get("mobile", "")).strip()
    dob = str(data.get("dob", "")).strip() or None
    gender = str(data.get("gender", "")).strip() or None
    disability = str(data.get("disability", "")).strip() or None
    
    # Initialize DB schema idempotently on first call
    try:
        init_db()
    except Exception:
        pass
    
    # Check if username exists
    existing_user = get_user_by_username(username)
    if existing_user and existing_user.get("id"):
        return jsonify({"error": "Username already exists"}), 409
    
    # Check if email exists
    existing_email = get_user_by_email(email)
    if existing_email and existing_email.get("id"):
        return jsonify({"error": "Email already registered"}), 409
    
    try:
        # Create user with profile data
        pwd_hash = generate_password_hash(password)
        user = create_user(username=username, password_hash=pwd_hash, email=email)
        user_id = user.get("id")
        
        # Update profile data
        from database.models import update_user_profile
        update_user_profile(
            user_id=user_id,
            full_name=full_name,
            mobile=mobile,
            gender=gender,
            dob=dob,
            disability=disability
        )
        
        # Send welcome email (non-blocking)
        try:
            email_service.send_welcome_email(email, username)
        except Exception as e:
            # Log but don't fail signup
            pass
        
        return jsonify({
            "success": True,
            "message": "Account created successfully",
            "user": {
                "id": user.get("id"),
                "username": user.get("username"),
                "email": user.get("email"),
                "full_name": full_name
            }
        }), 201
    
    except Exception as e:
        return jsonify({"error": f"Signup failed: {str(e)}"}), 500


@auth_bp.route("/login", methods=["POST"])
def login():
    """Enhanced login - redirects to preview page."""
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()
    
    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400
    
    user = get_user_by_username(username)
    if not user or not check_password_hash(user.get("password_hash") or "", password):
        return jsonify({"error": "Invalid credentials"}), 401
    
    session["user_id"] = user.get("id")
    session["username"] = user.get("username")
    
    # Always show preview page after login
    return jsonify({
        "success": True,
        "user": {
            "id": user.get("id"),
            "username": user.get("username"),
            "full_name": user.get("full_name")
        },
        "redirect": "/preview"  # Always redirect to preview
    }), 200


@auth_bp.route("/logout", methods=["POST"])
def logout():
    """Logout and clear session."""
    session.clear()
    return jsonify({"success": True}), 200


# ============================================================================
# PASSWORD RESET ROUTES
# ============================================================================

@auth_bp.route("/forgot-password", methods=["POST"])
def forgot_password():
    """Request password reset - sends email with reset link and verification code."""
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip()
    
    if not email:
        return jsonify({"error": "Email is required"}), 400
    
    user = get_user_by_email(email)
    if not user:
        # Don't reveal if email exists
        return jsonify({
            "success": True,
            "message": "If an account exists with this email, a password reset link will be sent"
        }), 200
    
    try:
        # Generate 6-digit verification code
        verification_code = f"{secrets.randbelow(1000000):06d}"

        # Generate reset token
        serializer = URLSafeTimedSerializer(os.getenv("SECRET_KEY", "dev-secret-key"))
        token = serializer.dumps(
            {
                "user_id": user.get("id"),
                "email": email,
                "verification_code": verification_code,
            },
            salt="password-reset-salt",
        )
        
        # Save token to database with expiration (15 minutes)
        expires_at = (datetime.utcnow() + timedelta(minutes=15)).isoformat()
        create_password_reset_token(user.get("id"), token, expires_at)
        
        # Create reset link
        reset_link = f"{request.host_url.rstrip('/')}/auth/password-reset?token={token}"
        
        # Send email
        success, msg = email_service.send_password_reset_email(
            email,
            user.get("username"),
            reset_link,
            verification_code=verification_code,
            expires_in_minutes=15
        )
        
        if success:
            return jsonify({
                "success": True,
                "message": "Password reset link and verification code sent to your email"
            }), 200
        else:
            return jsonify({
                "success": True,
                "message": "Password reset requested (email service unavailable)"
            }), 200
    
    except Exception as e:
        return jsonify({"error": f"Error processing reset: {str(e)}"}), 500


@auth_bp.route("/reset-password", methods=["POST"])
def reset_password():
    """Reset password using valid token and email verification code."""
    data = request.get_json(silent=True) or {}
    token = (data.get("token") or "").strip()
    verification_code = (data.get("verification_code") or "").strip()
    new_password = data.get("new_password", "")
    confirm_password = data.get("confirm_password", "")
    
    if not token:
        return jsonify({"error": "Token is required"}), 400

    if not verification_code:
        return jsonify({"error": "Verification code is required"}), 400

    if not verification_code.isdigit() or len(verification_code) != 6:
        return jsonify({"error": "Verification code must be a 6-digit number"}), 400
    
    if not new_password or not confirm_password:
        return jsonify({"error": "Password and confirmation are required"}), 400
    
    # Validate password
    is_valid, msg = validate_password(new_password)
    if not is_valid:
        return jsonify({"error": msg}), 400
    
    if new_password != confirm_password:
        return jsonify({"error": "Passwords do not match"}), 400
    
    try:
        # Get token from database
        reset_token = get_password_reset_token_by_string(token)
        if not reset_token or not reset_token.get("id"):
            return jsonify({"error": "Invalid or expired reset link"}), 400
        
        # Check expiration
        expires_at = datetime.fromisoformat(reset_token.get("expires_at"))
        if datetime.utcnow() > expires_at:
            return jsonify({"error": "Password reset link has expired"}), 400

        # Validate token signature and embedded verification code
        serializer = URLSafeTimedSerializer(os.getenv("SECRET_KEY", "dev-secret-key"))
        token_data = serializer.loads(token, salt="password-reset-salt")
        expected_code = str(token_data.get("verification_code") or "")
        if verification_code != expected_code:
            return jsonify({"error": "Invalid verification code"}), 400
        
        # Update password
        user_id = reset_token.get("user_id")
        pwd_hash = generate_password_hash(new_password)
        update_user_password(user_id, pwd_hash)
        
        # Mark token as used
        mark_password_reset_token_used(reset_token.get("id"))
        
        # Invalidate all other reset tokens for this user
        invalidate_password_reset_tokens(user_id)
        
        return jsonify({
            "success": True,
            "message": "Password reset successfully. Please login with your new password."
        }), 200
    
    except (BadSignature, SignatureExpired):
        return jsonify({"error": "Invalid or expired reset link"}), 400
    except Exception as e:
        return jsonify({"error": f"Error resetting password: {str(e)}"}), 500


@auth_bp.route("/validate-reset-token", methods=["GET"])
def validate_reset_token():
    """Validate if reset token is still valid."""
    token = request.args.get("token", "").strip()
    
    if not token:
        return jsonify({"valid": False, "error": "Token is required"}), 400
    
    try:
        reset_token = get_password_reset_token_by_string(token)
        if not reset_token or not reset_token.get("id"):
            return jsonify({"valid": False, "error": "Invalid token"}), 400
        
        expires_at = datetime.fromisoformat(reset_token.get("expires_at"))
        if datetime.utcnow() > expires_at:
            return jsonify({"valid": False, "error": "Token has expired"}), 400
        
        return jsonify({"valid": True}), 200
    
    except Exception as e:
        return jsonify({"valid": False, "error": "Token validation failed"}), 400


@auth_bp.route("/check-username", methods=["POST"])
def check_username():
    """Check if username is available."""
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    
    if not username or len(username) < 3:
        return jsonify({"available": False}), 200
    
    user = get_user_by_username(username)
    return jsonify({"available": not bool(user and user.get("id"))}), 200


@auth_bp.route("/check-email", methods=["POST"])
def check_email():
    """Check if email is available."""
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip()
    
    if not email or not validate_email(email):
        return jsonify({"available": False}), 200
    
    user = get_user_by_email(email)
    return jsonify({"available": not bool(user and user.get("id"))}), 200
