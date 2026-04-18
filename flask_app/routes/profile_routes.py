"""
Profile Routes for SignAI
Handles user profile viewing and editing
"""

from flask import Blueprint, request, jsonify, session, render_template
from werkzeug.security import generate_password_hash, check_password_hash
import os
import sys
import re

# Ensure imports work
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

from database.models import (
    get_user_by_id, update_user_profile, update_user_password
)
from flask_app.routes.auth import login_required, validate_password, validate_mobile

profile_bp = Blueprint("profile", __name__, url_prefix="/profile")


# ============================================================================
# PROFILE ROUTES
# ============================================================================

@profile_bp.route("/view", methods=["GET"])
@login_required
def profile_page():
    """Render the profile page UI."""
    return render_template("profile.html")

@profile_bp.route("", methods=["GET"])
@login_required
def get_profile():
    """Get current user's profile."""
    user_id = session.get("user_id")
    user = get_user_by_id(user_id)
    
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    return jsonify({
        "success": True,
        "profile": {
            "id": user.get("id"),
            "username": user.get("username"),
            "email": user.get("email"),
            "full_name": user.get("full_name"),
            "dob": user.get("dob"),
            "gender": user.get("gender"),
            "mobile": user.get("mobile"),
            "disability": user.get("disability"),
            "created_at": user.get("created_at"),
            "updated_at": user.get("updated_at"),
        }
    }), 200


@profile_bp.route("", methods=["PUT"])
@login_required
def update_profile():
    """Update user profile (editable fields only)."""
    user_id = session.get("user_id")
    user = get_user_by_id(user_id)
    
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    data = request.get_json(silent=True) or {}
    
    # Validate updatable fields
    full_name = data.get("full_name")
    mobile = data.get("mobile")
    gender = data.get("gender")
    disability = data.get("disability")
    
    # Validate full_name if provided
    if full_name is not None:
        full_name = str(full_name).strip()
        if full_name and len(full_name) < 2:
            return jsonify({"error": "Full name must be at least 2 characters"}), 400
    
    # Validate mobile if provided
    if mobile is not None:
        mobile = str(mobile).strip()
        if mobile and not validate_mobile(mobile):
            return jsonify({"error": "Mobile number must be 10 digits"}), 400
    
    # Validate gender if provided
    if gender is not None:
        gender = str(gender).strip()
        valid_genders = ["Male", "Female", "Other", "Prefer not to say"]
        if gender and gender not in valid_genders:
            return jsonify({"error": f"Gender must be one of: {', '.join(valid_genders)}"}), 400
    
    try:
        updated_user = update_user_profile(
            user_id=user_id,
            full_name=full_name,
            mobile=mobile,
            gender=gender,
            disability=disability
        )
        
        return jsonify({
            "success": True,
            "message": "Profile updated successfully",
            "profile": {
                "id": updated_user.get("id"),
                "username": updated_user.get("username"),
                "email": updated_user.get("email"),
                "full_name": updated_user.get("full_name"),
                "dob": updated_user.get("dob"),
                "gender": updated_user.get("gender"),
                "mobile": updated_user.get("mobile"),
                "disability": updated_user.get("disability"),
                "created_at": updated_user.get("created_at"),
                "updated_at": updated_user.get("updated_at"),
            }
        }), 200
    
    except Exception as e:
        return jsonify({"error": f"Failed to update profile: {str(e)}"}), 500


@profile_bp.route("/change-password", methods=["POST"])
@login_required
def change_password():
    """Change user password."""
    user_id = session.get("user_id")
    user = get_user_by_id(user_id)
    
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    data = request.get_json(silent=True) or {}
    current_password = data.get("current_password", "")
    new_password = data.get("new_password", "")
    confirm_password = data.get("confirm_password", "")
    
    if not current_password:
        return jsonify({"error": "Current password is required"}), 400
    
    # Verify current password
    if not check_password_hash(user.get("password_hash", ""), current_password):
        return jsonify({"error": "Current password is incorrect"}), 401
    
    if not new_password or not confirm_password:
        return jsonify({"error": "New password and confirmation are required"}), 400
    
    # Validate new password
    is_valid, msg = validate_password(new_password)
    if not is_valid:
        return jsonify({"error": msg}), 400
    
    if new_password != confirm_password:
        return jsonify({"error": "New passwords do not match"}), 400
    
    # Don't allow using same password
    if check_password_hash(user.get("password_hash", ""), new_password):
        return jsonify({"error": "New password must be different from current password"}), 400
    
    try:
        pwd_hash = generate_password_hash(new_password)
        update_user_password(user_id, pwd_hash)
        
        return jsonify({
            "success": True,
            "message": "Password changed successfully"
        }), 200
    
    except Exception as e:
        return jsonify({"error": f"Failed to change password: {str(e)}"}), 500
