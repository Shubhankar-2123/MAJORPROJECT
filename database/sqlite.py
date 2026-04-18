import os
import sqlite3
from typing import Optional

DEFAULT_DB_PATH = None

def get_db_path(explicit_path: Optional[str] = None) -> str:
    global DEFAULT_DB_PATH
    if explicit_path:
        return explicit_path
    if DEFAULT_DB_PATH:
        return DEFAULT_DB_PATH
    # Default to data/app.db under project root
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base, "data")
    os.makedirs(data_dir, exist_ok=True)
    DEFAULT_DB_PATH = os.path.join(data_dir, "app.db")
    return DEFAULT_DB_PATH


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = get_db_path(db_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize the SQLite database schema (idempotent)."""
    from . import schema
    conn = get_connection(db_path)
    try:
        schema.create_all(conn)
        schema.migrate(conn)
    finally:
        conn.close()