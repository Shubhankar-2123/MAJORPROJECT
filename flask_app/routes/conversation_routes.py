import re
from flask import Blueprint, request, jsonify, session
from flask_app.routes.auth import login_required

from database.conversation_models import (
    create_conversation,
    list_conversations_for_user,
    get_conversation,
    list_messages_for_conversation,
    create_message,
    update_conversation_title,
    rename_conversation,
    toggle_pin,
    toggle_archive,
    apply_auto_title,
    touch_conversation,
    delete_conversation,
)

conversation_bp = Blueprint("conversations", __name__)


def _extract_title(text: str, max_words: int = 7, max_len: int = 40) -> str:
    """Generate a short title from message text."""
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", (text or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return "New Chat"
    words = cleaned.split(" ")[:max_words]
    title = " ".join(words)
    if len(title) > max_len:
        title = title[:max_len].rstrip()
    return title


@conversation_bp.route("/conversations", methods=["POST"])
@conversation_bp.route("/api/conversations", methods=["POST"])
@login_required
def create_new_conversation():
    """Create a new conversation. Optionally with initial title."""
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    user_id = session.get("user_id")
    conversation = create_conversation(user_id, title or "New Chat")
    return jsonify({"success": True, "conversation": conversation})


@conversation_bp.route("/conversations", methods=["GET"])
@conversation_bp.route("/api/conversations", methods=["GET"])
@login_required
def list_conversations():
    """List all conversations for logged-in user, sorted by updated_at DESC."""
    user_id = session.get("user_id")
    limit = request.args.get("limit", 100, type=int)
    include_archived = request.args.get("archived", "0") == "1"
    conversations = list_conversations_for_user(user_id, limit=limit, include_archived=include_archived)
    return jsonify({"success": True, "conversations": conversations})


@conversation_bp.route("/conversations/<int:conversation_id>", methods=["GET"])
@conversation_bp.route("/api/conversations/<int:conversation_id>", methods=["GET"])
@login_required
def get_conversation_messages(conversation_id: int):
    """Retrieve conversation and all its messages."""
    convo = get_conversation(conversation_id)
    if not convo:
        return jsonify({"error": "Conversation not found"}), 404
    if convo.get("user_id") != session.get("user_id"):
        return jsonify({"error": "Access denied"}), 403
    messages = list_messages_for_conversation(conversation_id)
    return jsonify({"success": True, "conversation": convo, "messages": messages})


@conversation_bp.route("/conversations/<int:conversation_id>/messages", methods=["POST"])
@conversation_bp.route("/api/conversations/<int:conversation_id>/messages", methods=["POST"])
@login_required
def add_message(conversation_id: int):
    """Add a message to a conversation (text or video)."""
    convo = get_conversation(conversation_id)
    if not convo:
        return jsonify({"error": "Conversation not found"}), 404
    if convo.get("user_id") != session.get("user_id"):
        return jsonify({"error": "Access denied"}), 403

    data = request.get_json(silent=True) or {}
    sender = (data.get("sender") or "user").strip()
    message_type = (data.get("message_type") or "text").strip()
    text_content = (data.get("text_content") or "").strip() if data.get("text_content") is not None else None
    video_path = (data.get("video_path") or "").strip() if data.get("video_path") is not None else None
    prediction_text = data.get("prediction")
    confidence_val = data.get("confidence")

    try:
        message = create_message(
            conversation_id=conversation_id,
            sender=sender,
            message_type=message_type,
            text_content=text_content,
            video_path=video_path,
            prediction=prediction_text,
            confidence=confidence_val,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if sender == "user" and message_type == "text":
        auto_title = _extract_title(text_content or "")
        apply_auto_title(conversation_id, auto_title)

    touch_conversation(conversation_id)
    return jsonify({"success": True, "message": message})


@conversation_bp.route("/conversations/<int:conversation_id>/messages/video", methods=["POST"])
@conversation_bp.route("/api/conversations/<int:conversation_id>/messages/video", methods=["POST"])
@login_required
def add_video_message(conversation_id: int):
    """Backwards-compatible endpoint for adding a video message."""
    convo = get_conversation(conversation_id)
    if not convo:
        return jsonify({"error": "Conversation not found"}), 404
    if convo.get("user_id") != session.get("user_id"):
        return jsonify({"error": "Access denied"}), 403

    data = request.get_json(silent=True) or {}
    video_path = (data.get("video_path") or "").strip()
    prediction_text = data.get("prediction")
    confidence_val = data.get("confidence")

    if not video_path:
        return jsonify({"error": "Video path is required"}), 400

    try:
        user_message = create_message(
            conversation_id=conversation_id,
            sender="user",
            message_type="video",
            video_path=video_path,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    system_message = None
    if prediction_text:
        system_message = create_message(
            conversation_id=conversation_id,
            sender="system",
            message_type="text",
            text_content=prediction_text,
            prediction=prediction_text,
            confidence=confidence_val,
        )

    if not convo.get("title") or convo.get("title") == "New Chat":
        title = prediction_text or "Video Upload"
        update_conversation_title(conversation_id, title)

    touch_conversation(conversation_id)
    return jsonify({
        "success": True,
        "user_message": user_message,
        "system_message": system_message,
    })


@conversation_bp.route("/conversations/<int:conversation_id>", methods=["DELETE"])
@conversation_bp.route("/api/conversations/<int:conversation_id>", methods=["DELETE"])
@login_required
def delete_conversation_route(conversation_id: int):
    """Delete a conversation (only if owner)."""
    convo = get_conversation(conversation_id)
    if not convo:
        return jsonify({"error": "Conversation not found"}), 404
    if convo.get("user_id") != session.get("user_id"):
        return jsonify({"error": "Access denied"}), 403
    delete_conversation(conversation_id)
    return jsonify({"success": True})


@conversation_bp.route("/conversations/<int:conversation_id>/rename", methods=["PATCH"])
@conversation_bp.route("/api/conversations/<int:conversation_id>/rename", methods=["PATCH"])
@login_required
def rename_conversation_route(conversation_id: int):
    convo = get_conversation(conversation_id)
    if not convo:
        return jsonify({"error": "Conversation not found"}), 404
    if convo.get("user_id") != session.get("user_id"):
        return jsonify({"error": "Access denied"}), 403

    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"error": "Title is required"}), 400
    updated = rename_conversation(conversation_id, title)
    return jsonify({"success": True, "conversation": updated})


@conversation_bp.route("/conversations/<int:conversation_id>/pin", methods=["POST"])
@conversation_bp.route("/api/conversations/<int:conversation_id>/pin", methods=["POST"])
@login_required
def toggle_pin_route(conversation_id: int):
    convo = get_conversation(conversation_id)
    if not convo:
        return jsonify({"error": "Conversation not found"}), 404
    if convo.get("user_id") != session.get("user_id"):
        return jsonify({"error": "Access denied"}), 403
    updated = toggle_pin(conversation_id)
    return jsonify({"success": True, "conversation": updated})


@conversation_bp.route("/conversations/<int:conversation_id>/archive", methods=["POST"])
@conversation_bp.route("/api/conversations/<int:conversation_id>/archive", methods=["POST"])
@login_required
def toggle_archive_route(conversation_id: int):
    convo = get_conversation(conversation_id)
    if not convo:
        return jsonify({"error": "Conversation not found"}), 404
    if convo.get("user_id") != session.get("user_id"):
        return jsonify({"error": "Access denied"}), 403
    updated = toggle_archive(conversation_id)
    return jsonify({"success": True, "conversation": updated})
