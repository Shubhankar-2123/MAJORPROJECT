from flask import Blueprint, request, jsonify, session
from database.models import create_feedback, list_feedback_for_user

feedback_bp = Blueprint("feedback", __name__, url_prefix="/feedback")

@feedback_bp.route("/submit", methods=["POST"])
def submit_feedback():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401
    data = request.get_json(silent=True) or {}
    prediction_id = data.get("prediction_id")
    correction_text = data.get("correction_text", "").strip()
    original_text = data.get("original_text", "")
    if not correction_text:
        return jsonify({"error": "Correction text required"}), 400
    feedback = create_feedback(user_id, prediction_id, correction_text, original_text)
    return jsonify({"success": True, "feedback": feedback})

@feedback_bp.route("/history", methods=["GET"])
def feedback_history():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401
    feedbacks = list_feedback_for_user(user_id, limit=100)
    return jsonify({"success": True, "feedbacks": feedbacks})
