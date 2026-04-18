from flask import Blueprint, request, jsonify, session
from datetime import datetime
from flask_app.routes.auth import login_required
from database.models import list_predictions_for_user

predictions_bp = Blueprint("predictions", __name__, url_prefix="/api/predictions")

@predictions_bp.route("/history", methods=["GET"])
@login_required
def history():
    """Get prediction history for the logged-in user."""
    user_id = session.get("user_id")
    limit = request.args.get("limit", 50, type=int)
    predictions = list_predictions_for_user(user_id, limit=limit)

    for pred in predictions:
        if pred.get("created_at"):
            try:
                dt = datetime.fromisoformat(pred["created_at"].replace("Z", "+00:00"))
                pred["created_at_display"] = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pred["created_at_display"] = pred["created_at"]

    return jsonify({
        "success": True,
        "count": len(predictions),
        "predictions": predictions
    })
