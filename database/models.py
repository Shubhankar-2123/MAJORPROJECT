from typing import Any, Dict, List, Optional, Tuple
import sqlite3

from .sqlite import get_connection


def row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return dict(row) if row is not None else {}


# User operations

def create_user(username: str, password_hash: str, email: Optional[str] = None, db_path: Optional[str] = None) -> Dict[str, Any]:
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, password_hash),
        )
        conn.commit()
        user_id = cur.lastrowid
        return get_user_by_id(user_id, db_path)
    finally:
        conn.close()


def get_user_by_username(username: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    conn = get_connection(db_path)
    try:
        cur = conn.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        return row_to_dict(row)
    finally:
        conn.close()


def get_user_by_id(user_id: int, db_path: Optional[str] = None) -> Dict[str, Any]:
    conn = get_connection(db_path)
    try:
        cur = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        return row_to_dict(row)
    finally:
        conn.close()


def get_user_welcome_status(user_id: int, db_path: Optional[str] = None) -> bool:
    conn = get_connection(db_path)
    try:
        cur = conn.execute("SELECT has_seen_welcome FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        if row is None:
            return False
        return bool(row["has_seen_welcome"]) if "has_seen_welcome" in row.keys() else False
    finally:
        conn.close()


def mark_user_welcome_seen(user_id: int, db_path: Optional[str] = None) -> None:
    conn = get_connection(db_path)
    try:
        conn.execute("UPDATE users SET has_seen_welcome = 1 WHERE id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()


# Prediction operations

def create_prediction(
    user_id: Optional[int],
    input_type: str,
    input_path: Optional[str],
    input_text: Optional[str],
    predicted_text: Optional[str],
    translated_text: Optional[str],
    confidence: Optional[float],
    model_used: Optional[str],
    tts_audio_path: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            """
            INSERT INTO predictions (
                user_id, input_type, input_path, input_text, predicted_text,
                translated_text, confidence, model_used, tts_audio_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id, input_type, input_path, input_text, predicted_text,
                translated_text, confidence, model_used, tts_audio_path,
            ),
        )
        conn.commit()
        pred_id = cur.lastrowid
        return get_prediction(pred_id, db_path)
    finally:
        conn.close()


def get_prediction(prediction_id: int, db_path: Optional[str] = None) -> Dict[str, Any]:
    conn = get_connection(db_path)
    try:
        cur = conn.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
        row = cur.fetchone()
        return row_to_dict(row)
    finally:
        conn.close()


def list_predictions_for_user(user_id: int, limit: int = 50, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            "SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        )
        rows = cur.fetchall()
        return [row_to_dict(r) for r in rows]
    finally:
        conn.close()


# Feedback operations

def create_feedback(
    user_id: Optional[int],
    prediction_id: Optional[int],
    correction_text: str,
    original_text: Optional[str] = None,
    processed: bool = False,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO feedback (user_id, prediction_id, original_text, correction_text, processed) VALUES (?, ?, ?, ?, ?)",
            (user_id, prediction_id, original_text, correction_text, int(processed)),
        )
        conn.commit()
        fb_id = cur.lastrowid
        return get_feedback_by_id(fb_id, db_path)
    finally:
        conn.close()


def get_feedback_by_id(feedback_id: int, db_path: Optional[str] = None) -> Dict[str, Any]:
    conn = get_connection(db_path)
    try:
        cur = conn.execute("SELECT * FROM feedback WHERE id = ?", (feedback_id,))
        row = cur.fetchone()
        return row_to_dict(row)
    finally:
        conn.close()


def list_feedback_for_user(user_id: int, limit: int = 100, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            "SELECT * FROM feedback WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        )
        rows = cur.fetchall()
        return [row_to_dict(r) for r in rows]
    finally:
        conn.close()


def mark_feedback_processed(feedback_id: int, processed: bool = True, db_path: Optional[str] = None) -> Dict[str, Any]:
    conn = get_connection(db_path)
    try:
        conn.execute("UPDATE feedback SET processed = ? WHERE id = ?", (int(processed), feedback_id))
        conn.commit()
        return get_feedback_by_id(feedback_id, db_path)
    finally:
        conn.close()


# Performance operations

def log_performance(
    user_id: Optional[int],
    model_used: Optional[str],
    inference_time_ms: float,
    accuracy: Optional[float] = None,
    confidence: Optional[float] = None,
    notes: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            """
            INSERT INTO performance (user_id, model_used, inference_time_ms, accuracy, confidence, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, model_used, inference_time_ms, accuracy, confidence, notes),
        )
        conn.commit()
        perf_id = cur.lastrowid
        return get_performance_by_id(perf_id, db_path)
    finally:
        conn.close()


def get_performance_by_id(performance_id: int, db_path: Optional[str] = None) -> Dict[str, Any]:
    conn = get_connection(db_path)
    try:
        cur = conn.execute("SELECT * FROM performance WHERE id = ?", (performance_id,))
        row = cur.fetchone()
        return row_to_dict(row)
    finally:
        conn.close()


def get_performance_stats(user_id: Optional[int] = None, db_path: Optional[str] = None) -> Dict[str, Any]:
    """Returns aggregated performance stats: count, avg_inference_time_ms, avg_confidence, avg_accuracy."""
    conn = get_connection(db_path)
    try:
        if user_id is None:
            cur = conn.execute(
                "SELECT COUNT(*) AS total_runs, AVG(inference_time_ms) AS avg_inference_time_ms, AVG(confidence) AS avg_confidence, AVG(accuracy) AS avg_accuracy FROM performance"
            )
        else:
            cur = conn.execute(
                "SELECT COUNT(*) AS total_runs, AVG(inference_time_ms) AS avg_inference_time_ms, AVG(confidence) AS avg_confidence, AVG(accuracy) AS avg_accuracy FROM performance WHERE user_id = ?",
                (user_id,),
            )
        row = cur.fetchone()
        return row_to_dict(row)
    finally:
        conn.close()


def get_feedback_stats(user_id: Optional[int] = None, db_path: Optional[str] = None) -> Dict[str, Any]:
    """Returns feedback stats: total, processed_count, pending_count."""
    conn = get_connection(db_path)
    try:
        if user_id is None:
            cur = conn.execute(
                "SELECT COUNT(*) AS total, SUM(processed) AS processed_count, (COUNT(*) - SUM(processed)) AS pending_count FROM feedback"
            )
        else:
            cur = conn.execute(
                "SELECT COUNT(*) AS total, SUM(processed) AS processed_count, (COUNT(*) - SUM(processed)) AS pending_count FROM feedback WHERE user_id = ?",
                (user_id,),
            )
        row = cur.fetchone()
        stats = row_to_dict(row)
        # Handle None sums when there are zero rows
        stats["processed_count"] = int(stats.get("processed_count") or 0)
        stats["pending_count"] = int(stats.get("pending_count") or 0)
        stats["total"] = int(stats.get("total") or 0)
        return stats
    finally:
        conn.close()


def get_text_cache(cache_key: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            "SELECT cache_key, simplified_json, source, created_at FROM text_cache WHERE cache_key = ?",
            (cache_key,),
        )
        row = cur.fetchone()
        return row_to_dict(row)
    finally:
        conn.close()


def upsert_text_cache(cache_key: str, simplified_json: str, source: str, created_at: str, db_path: Optional[str] = None) -> None:
    conn = get_connection(db_path)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO text_cache (cache_key, simplified_json, source, created_at) VALUES (?, ?, ?, ?)",
            (cache_key, simplified_json, source, created_at),
        )
        conn.commit()
    finally:
        conn.close()


# User Profile operations

def get_user_by_email(email: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    """Get user by email address."""
    conn = get_connection(db_path)
    try:
        cur = conn.execute("SELECT * FROM users WHERE email = ?", (email,))
        row = cur.fetchone()
        return row_to_dict(row)
    finally:
        conn.close()


def update_user_profile(
    user_id: int,
    full_name: Optional[str] = None,
    dob: Optional[str] = None,
    mobile: Optional[str] = None,
    gender: Optional[str] = None,
    disability: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Update user profile information."""
    conn = get_connection(db_path)
    try:
        updates = []
        params = []
        if full_name is not None:
            updates.append("full_name = ?")
            params.append(full_name)
        if dob is not None:
            updates.append("dob = ?")
            params.append(dob)
        if mobile is not None:
            updates.append("mobile = ?")
            params.append(mobile)
        if gender is not None:
            updates.append("gender = ?")
            params.append(gender)
        if disability is not None:
            updates.append("disability = ?")
            params.append(disability)
        
        if updates:
            updates.append("updated_at = CURRENT_TIMESTAMP")
            params.append(user_id)
            query = "UPDATE users SET " + ", ".join(updates) + " WHERE id = ?"
            conn.execute(query, params)
            conn.commit()
        
        return get_user_by_id(user_id, db_path)
    finally:
        conn.close()


def update_user_password(user_id: int, password_hash: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    """Update user password hash."""
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE users SET password_hash = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (password_hash, user_id),
        )
        conn.commit()
        return get_user_by_id(user_id, db_path)
    finally:
        conn.close()


# Password Reset operations

def create_password_reset_token(user_id: int, token: str, expires_at: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    """Create a password reset token."""
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO password_reset_tokens (user_id, token, expires_at) VALUES (?, ?, ?)",
            (user_id, token, expires_at),
        )
        conn.commit()
        token_id = cur.lastrowid
        return get_password_reset_token(token_id, db_path)
    finally:
        conn.close()


def get_password_reset_token(token_id: int, db_path: Optional[str] = None) -> Dict[str, Any]:
    """Get password reset token by ID."""
    conn = get_connection(db_path)
    try:
        cur = conn.execute("SELECT * FROM password_reset_tokens WHERE id = ?", (token_id,))
        row = cur.fetchone()
        return row_to_dict(row)
    finally:
        conn.close()


def get_password_reset_token_by_string(token: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    """Get password reset token by token string."""
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            "SELECT * FROM password_reset_tokens WHERE token = ? AND used = 0",
            (token,),
        )
        row = cur.fetchone()
        return row_to_dict(row)
    finally:
        conn.close()


def mark_password_reset_token_used(token_id: int, db_path: Optional[str] = None) -> Dict[str, Any]:
    """Mark a password reset token as used."""
    conn = get_connection(db_path)
    try:
        conn.execute("UPDATE password_reset_tokens SET used = 1 WHERE id = ?", (token_id,))
        conn.commit()
        return get_password_reset_token(token_id, db_path)
    finally:
        conn.close()


def invalidate_password_reset_tokens(user_id: int, db_path: Optional[str] = None) -> int:
    """Invalidate all unused password reset tokens for a user."""
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            "UPDATE password_reset_tokens SET used = 1 WHERE user_id = ? AND used = 0",
            (user_id,),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()