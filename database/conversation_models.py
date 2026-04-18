from typing import Any, Dict, List, Optional
import sqlite3

from .sqlite import get_connection


def row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return dict(row) if row is not None else {}


def create_conversation(user_id: int, title: Optional[str] = None, db_path: Optional[str] = None) -> Dict[str, Any]:
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            """
            INSERT INTO conversations (user_id, title, is_pinned, is_archived, auto_title)
            VALUES (?, ?, 0, 0, 1)
            """,
            (user_id, title or "New Chat"),
        )
        conn.commit()
        return get_conversation(cur.lastrowid, db_path)
    finally:
        conn.close()


def get_conversation(conversation_id: int, db_path: Optional[str] = None) -> Dict[str, Any]:
    conn = get_connection(db_path)
    try:
        cur = conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
        row = cur.fetchone()
        return row_to_dict(row)
    finally:
        conn.close()


def list_conversations_for_user(
    user_id: int,
    limit: int = 100,
    include_archived: bool = False,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    conn = get_connection(db_path)
    try:
        if include_archived:
            cur = conn.execute(
                """
                SELECT * FROM conversations
                WHERE user_id = ? AND is_archived = 1
                ORDER BY is_pinned DESC, updated_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
        else:
            cur = conn.execute(
                """
                SELECT * FROM conversations
                WHERE user_id = ? AND is_archived = 0
                ORDER BY is_pinned DESC, updated_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
        rows = cur.fetchall()
        return [row_to_dict(r) for r in rows]
    finally:
        conn.close()


def update_conversation_title(conversation_id: int, title: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE conversations SET title = ?, auto_title = 0 WHERE id = ?",
            (title, conversation_id),
        )
        conn.commit()
        return get_conversation(conversation_id, db_path)
    finally:
        conn.close()


def touch_conversation(conversation_id: int, db_path: Optional[str] = None) -> None:
    conn = get_connection(db_path)
    try:
        conn.execute("UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (conversation_id,))
        conn.commit()
    finally:
        conn.close()


def rename_conversation(conversation_id: int, title: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    return update_conversation_title(conversation_id, title, db_path)


def toggle_pin(conversation_id: int, db_path: Optional[str] = None) -> Dict[str, Any]:
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE conversations SET is_pinned = CASE WHEN is_pinned = 1 THEN 0 ELSE 1 END WHERE id = ?",
            (conversation_id,),
        )
        conn.commit()
        return get_conversation(conversation_id, db_path)
    finally:
        conn.close()


def toggle_archive(conversation_id: int, db_path: Optional[str] = None) -> Dict[str, Any]:
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE conversations SET is_archived = CASE WHEN is_archived = 1 THEN 0 ELSE 1 END WHERE id = ?",
            (conversation_id,),
        )
        conn.commit()
        return get_conversation(conversation_id, db_path)
    finally:
        conn.close()


def apply_auto_title(conversation_id: int, title: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Apply auto title only if auto_title is still enabled."""
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            UPDATE conversations
            SET title = ?, auto_title = 0
            WHERE id = ? AND auto_title = 1
            """,
            (title, conversation_id),
        )
        conn.commit()
        return get_conversation(conversation_id, db_path)
    finally:
        conn.close()


def delete_conversation(conversation_id: int, db_path: Optional[str] = None) -> None:
    conn = get_connection(db_path)
    try:
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
    finally:
        conn.close()


def create_message(
    conversation_id: int,
    sender: str,
    message_type: str,
    text_content: Optional[str] = None,
    video_path: Optional[str] = None,
    prediction: Optional[str] = None,
    confidence: Optional[float] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    if sender not in {"user", "system"}:
        raise ValueError("Invalid sender")
    if message_type not in {"text", "video"}:
        raise ValueError("Invalid message_type")
    if message_type == "text" and not (text_content or ""):
        raise ValueError("text_content is required for text messages")
    if message_type == "video" and not (video_path or ""):
        raise ValueError("video_path is required for video messages")

    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            """
            INSERT INTO messages (
                conversation_id,
                sender,
                message_type,
                text_content,
                video_path,
                prediction,
                confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                sender,
                message_type,
                text_content,
                video_path,
                prediction,
                confidence,
            ),
        )
        conn.commit()
        return get_message(cur.lastrowid, db_path)
    finally:
        conn.close()


def get_message(message_id: int, db_path: Optional[str] = None) -> Dict[str, Any]:
    conn = get_connection(db_path)
    try:
        cur = conn.execute("SELECT * FROM messages WHERE id = ?", (message_id,))
        row = cur.fetchone()
        return row_to_dict(row)
    finally:
        conn.close()


def list_messages_for_conversation(
    conversation_id: int,
    limit: int = 200,
    offset: int = 0,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            """
            SELECT * FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC
            LIMIT ? OFFSET ?
            """,
            (conversation_id, limit, offset),
        )
        rows = cur.fetchall()
        return [row_to_dict(r) for r in rows]
    finally:
        conn.close()
