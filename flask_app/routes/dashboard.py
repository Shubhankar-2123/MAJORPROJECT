from flask import Blueprint, request, jsonify, session, render_template, redirect, url_for
from database.models import list_predictions_for_user

dashboard_bp = Blueprint("dashboard", __name__, url_prefix="/dashboard")

@dashboard_bp.route("/", methods=["GET"])
def dashboard_page():
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("login_page"))
    return render_template("dashboard.html")

@dashboard_bp.route("/history", methods=["GET"])
def history():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401
    predictions = list_predictions_for_user(user_id, limit=50)
    return jsonify({"success": True, "predictions": predictions})

# Add more dashboard endpoints as needed
