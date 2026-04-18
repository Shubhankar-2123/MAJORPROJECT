"""
Custom Signs Database Models
Provides database operations for user custom sign videos.
"""

from typing import Any, Dict, List, Optional
import sqlite3
from .sqlite import get_connection


def row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a sqlite3.Row to a dictionary."""
    return dict(row) if row is not None else {}


# Custom Signs operations

def create_custom_sign(
    user_id: int,
    word: str,
    category: str = "words",
    video_path: Optional[str] = None,
    image_path: Optional[str] = None,
    verified: bool = False,
    confidence: Optional[float] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Create or update a custom sign (video or image) for a user."""
    conn = get_connection(db_path)
    try:
        # Use INSERT OR REPLACE to handle duplicates
        cur = conn.execute(
            """
            INSERT INTO custom_signs 
            (user_id, word, category, video_path, image_path, verified, confidence, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id, word) 
            DO UPDATE SET 
                category = excluded.category,
                video_path = excluded.video_path, 
                image_path = excluded.image_path,
                verified = excluded.verified,
                confidence = excluded.confidence,
                updated_at = CURRENT_TIMESTAMP
            """,
            (user_id, word, category, video_path, image_path, int(verified), confidence),
        )
        conn.commit()
        sign_id = cur.lastrowid
        # Get the created/updated record
        return get_custom_sign_by_id(sign_id, db_path)
    finally:
        conn.close()


def get_custom_sign(user_id: int, word: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    """Get a custom sign for a specific user and word."""
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            "SELECT * FROM custom_signs WHERE user_id = ? AND word = ?",
            (user_id, word),
        )
        row = cur.fetchone()
        return row_to_dict(row)
    finally:
        conn.close()


def get_custom_sign_by_id(sign_id: int, db_path: Optional[str] = None) -> Dict[str, Any]:
    """Get a custom sign by ID."""
    conn = get_connection(db_path)
    try:
        cur = conn.execute("SELECT * FROM custom_signs WHERE id = ?", (sign_id,))
        row = cur.fetchone()
        return row_to_dict(row)
    finally:
        conn.close()


def list_custom_signs_for_user(user_id: int, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all custom signs for a user."""
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            "SELECT * FROM custom_signs WHERE user_id = ? ORDER BY word ASC",
            (user_id,),
        )
        rows = cur.fetchall()
        return [row_to_dict(r) for r in rows]
    finally:
        conn.close()


def delete_custom_sign(sign_id: int, user_id: int, db_path: Optional[str] = None) -> bool:
    """Delete a custom sign. Returns True if deleted, False if not found or unauthorized."""
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            "DELETE FROM custom_signs WHERE id = ? AND user_id = ?",
            (sign_id, user_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def get_custom_signs_count(user_id: int, db_path: Optional[str] = None) -> int:
    """Get count of custom signs for a user."""
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            "SELECT COUNT(*) as count FROM custom_signs WHERE user_id = ?",
            (user_id,),
        )
        row = cur.fetchone()
        return row["count"] if row else 0
    finally:
        conn.close()


def list_custom_signs_by_category(
    user_id: int,
    category: str,
    db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """List custom signs for a user filtered by category."""
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            "SELECT * FROM custom_signs WHERE user_id = ? AND category = ? ORDER BY word ASC",
            (user_id, category),
        )
        rows = cur.fetchall()
        return [row_to_dict(r) for r in rows]
    finally:
        conn.close()


def get_custom_signs_by_category(
    category: str,
    db_path: Optional[str] = None
) -> int:
    """Count how many custom signs exist for a category."""
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            "SELECT COUNT(*) as count FROM custom_signs WHERE category = ?",
            (category,),
        )
        row = cur.fetchone()
        return row["count"] if row else 0
    finally:
        conn.close()


def update_custom_sign_verification(
    sign_id: int,
    verified: bool,
    confidence: float,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """Update verification status and confidence of a custom sign."""
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            UPDATE custom_signs 
            SET verified = ?, confidence = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (int(verified), confidence, sign_id),
        )
        conn.commit()
        return get_custom_sign_by_id(sign_id, db_path)
    finally:
        conn.close()
