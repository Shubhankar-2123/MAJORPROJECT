import sqlite3

# Schema creation for users, predictions, feedback, and performance tables

USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    has_seen_welcome INTEGER NOT NULL DEFAULT 0,
    full_name TEXT,
    dob TEXT,
    gender TEXT,
    mobile TEXT,
    disability TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

PREDICTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    input_type TEXT NOT NULL,
    input_path TEXT,
    input_text TEXT,
    predicted_text TEXT,
    translated_text TEXT,
    confidence REAL,
    model_used TEXT,
    tts_audio_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE SET NULL
);
"""

PREDICTIONS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_predictions_user_id ON predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
"""

FEEDBACK_TABLE = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    prediction_id INTEGER,
    original_text TEXT,
    correction_text TEXT NOT NULL,
    processed INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE SET NULL,
    FOREIGN KEY(prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
);
"""

FEEDBACK_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_feedback_user_id ON feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_prediction_id ON feedback(prediction_id);
CREATE INDEX IF NOT EXISTS idx_feedback_processed ON feedback(processed);
"""

PERFORMANCE_TABLE = """
CREATE TABLE IF NOT EXISTS performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    model_used TEXT,
    inference_time_ms REAL NOT NULL,
    accuracy REAL,
    confidence REAL,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE SET NULL
);
"""

PERFORMANCE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_performance_user_id ON performance(user_id);
CREATE INDEX IF NOT EXISTS idx_performance_created_at ON performance(created_at);
CREATE INDEX IF NOT EXISTS idx_performance_model_used ON performance(model_used);
"""

CONVERSATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    title TEXT,
    is_pinned INTEGER DEFAULT 0,
    is_archived INTEGER DEFAULT 0,
    auto_title INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);
"""

CONVERSATIONS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at);
"""

MESSAGES_TABLE = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    sender TEXT NOT NULL CHECK (sender IN ('user', 'system')),
    message_type TEXT NOT NULL CHECK (message_type IN ('text', 'video')),
    text_content TEXT,
    video_path TEXT,
    prediction TEXT,
    confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
"""

MESSAGES_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
"""

TEXT_CACHE_TABLE = """
CREATE TABLE IF NOT EXISTS text_cache (
    cache_key TEXT PRIMARY KEY,
    simplified_json TEXT NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

PASSWORD_RESET_TABLE = """
CREATE TABLE IF NOT EXISTS password_reset_tokens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    token TEXT NOT NULL UNIQUE,
    expires_at TIMESTAMP NOT NULL,
    used INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);
"""

PASSWORD_RESET_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_password_reset_user_id ON password_reset_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_password_reset_token ON password_reset_tokens(token);
CREATE INDEX IF NOT EXISTS idx_password_reset_expires ON password_reset_tokens(expires_at);
"""

CUSTOM_SIGNS_TABLE = """
CREATE TABLE IF NOT EXISTS custom_signs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    word TEXT NOT NULL,
    category TEXT DEFAULT 'words',
    video_path TEXT,
    image_path TEXT,
    verified INTEGER DEFAULT 0,
    confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE(user_id, word)
);
"""

CUSTOM_SIGNS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_custom_signs_user ON custom_signs(user_id);
CREATE INDEX IF NOT EXISTS idx_custom_signs_user_word ON custom_signs(user_id, word);
"""

DROP_ALL = """
DROP TABLE IF EXISTS custom_signs;
DROP TABLE IF EXISTS performance;
DROP TABLE IF EXISTS feedback;
DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS messages;
DROP TABLE IF EXISTS conversations;
DROP TABLE IF EXISTS password_reset_tokens;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS text_cache;
"""


def create_all(conn: sqlite3.Connection) -> None:
    """Create all tables and indexes in the connected SQLite database."""
    conn.executescript(USERS_TABLE)
    conn.executescript(PREDICTIONS_TABLE)
    conn.executescript(PREDICTIONS_INDEXES)
    conn.executescript(FEEDBACK_TABLE)
    conn.executescript(FEEDBACK_INDEXES)
    conn.executescript(PERFORMANCE_TABLE)
    conn.executescript(PERFORMANCE_INDEXES)
    conn.executescript(TEXT_CACHE_TABLE)
    conn.executescript(PASSWORD_RESET_TABLE)
    conn.executescript(PASSWORD_RESET_INDEXES)
    conn.executescript(CONVERSATIONS_TABLE)
    conn.executescript(CONVERSATIONS_INDEXES)
    conn.executescript(MESSAGES_TABLE)
    conn.executescript(MESSAGES_INDEXES)
    conn.executescript(CUSTOM_SIGNS_TABLE)
    conn.executescript(CUSTOM_SIGNS_INDEXES)
    conn.commit()


def migrate(conn: sqlite3.Connection) -> None:
    """Apply lightweight schema migrations for existing databases."""
    # Add has_seen_welcome column to users if missing
    try:
        conn.execute("ALTER TABLE users ADD COLUMN has_seen_welcome INTEGER NOT NULL DEFAULT 0")
    except Exception:
        pass
    
    # Add profile fields to users if missing
    try:
        users_cols = conn.execute("PRAGMA table_info(users)").fetchall()
        users_names = {c[1] for c in users_cols} if users_cols else set()
        if "full_name" not in users_names:
            conn.execute("ALTER TABLE users ADD COLUMN full_name TEXT")
        if "dob" not in users_names:
            conn.execute("ALTER TABLE users ADD COLUMN dob TEXT")
        if "gender" not in users_names:
            conn.execute("ALTER TABLE users ADD COLUMN gender TEXT")
        if "mobile" not in users_names:
            conn.execute("ALTER TABLE users ADD COLUMN mobile TEXT")
        if "disability" not in users_names:
            conn.execute("ALTER TABLE users ADD COLUMN disability TEXT")
        if "updated_at" not in users_names:
            conn.execute("ALTER TABLE users ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    except Exception:
        pass
    
    # Add UNIQUE constraint to email if not already unique
    try:
        conn.execute("ALTER TABLE users ADD CONSTRAINT unique_email UNIQUE (email)")
    except Exception:
        pass
    
    # Ensure text_cache table exists
    try:
        conn.executescript(PASSWORD_RESET_TABLE)
        conn.executescript(PASSWORD_RESET_INDEXES)
    except Exception:
        pass
    
    # Ensure password reset table exists
    try:
        conn.executescript(PASSWORD_RESET_TABLE)
        conn.executescript(PASSWORD_RESET_INDEXES)
    except Exception:
        pass
    
    # Ensure text_cache table exists
    try:
        conn.executescript(TEXT_CACHE_TABLE)
    except Exception:
        pass
    # Ensure conversations table exists
    try:
        conn.executescript(CONVERSATIONS_TABLE)
        conn.executescript(CONVERSATIONS_INDEXES)
    except Exception:
        pass

    # Add new conversation columns if missing
    try:
        convo_cols = conn.execute("PRAGMA table_info(conversations)").fetchall()
        convo_names = {c[1] for c in convo_cols} if convo_cols else set()
        if "is_pinned" not in convo_names:
            conn.execute("ALTER TABLE conversations ADD COLUMN is_pinned INTEGER DEFAULT 0")
        if "is_archived" not in convo_names:
            conn.execute("ALTER TABLE conversations ADD COLUMN is_archived INTEGER DEFAULT 0")
        if "auto_title" not in convo_names:
            conn.execute("ALTER TABLE conversations ADD COLUMN auto_title INTEGER DEFAULT 1")
    except Exception:
        pass

    # Ensure messages table exists with multimodal schema
    try:
        cols = conn.execute("PRAGMA table_info(messages)").fetchall()
        col_names = {c[1] for c in cols} if cols else set()
        has_message_type = "message_type" in col_names
    except Exception:
        col_names = set()
        has_message_type = False

    if not col_names:
        try:
            conn.executescript(MESSAGES_TABLE)
            conn.executescript(MESSAGES_INDEXES)
        except Exception:
            pass
    elif not has_message_type:
        # Migrate legacy messages table
        try:
            conn.execute("ALTER TABLE messages RENAME TO messages_old")
            conn.executescript(MESSAGES_TABLE)
            conn.execute(
                """
                INSERT INTO messages (
                    conversation_id,
                    sender,
                    message_type,
                    text_content,
                    video_path,
                    prediction,
                    confidence,
                    created_at
                )
                SELECT
                    conversation_id,
                    sender,
                    'text',
                    message_text,
                    NULL,
                    prediction,
                    confidence,
                    created_at
                FROM messages_old
                """
            )
            conn.execute("DROP TABLE messages_old")
            conn.executescript(MESSAGES_INDEXES)
        except Exception:
            pass
    else:
        try:
            conn.executescript(MESSAGES_INDEXES)
        except Exception:
            pass
    
    # Ensure custom_signs table exists
    try:
        conn.executescript(CUSTOM_SIGNS_TABLE)
        conn.executescript(CUSTOM_SIGNS_INDEXES)
    except Exception:
        pass
    
    # Migrate custom_signs table to add new columns
    try:
        custom_cols = conn.execute("PRAGMA table_info(custom_signs)").fetchall()
        custom_names = {c[1] for c in custom_cols} if custom_cols else set()
        if "category" not in custom_names:
            conn.execute("ALTER TABLE custom_signs ADD COLUMN category TEXT DEFAULT 'words'")
        if "image_path" not in custom_names:
            conn.execute("ALTER TABLE custom_signs ADD COLUMN image_path TEXT")
        if "verified" not in custom_names:
            conn.execute("ALTER TABLE custom_signs ADD COLUMN verified INTEGER DEFAULT 0")
        if "confidence" not in custom_names:
            conn.execute("ALTER TABLE custom_signs ADD COLUMN confidence REAL")
        # Make video_path nullable
        try:
            conn.execute("ALTER TABLE custom_signs MODIFY video_path TEXT")
        except Exception:
            pass  # SQLite doesn't support MODIFY, will skip
    except Exception:
        pass
    
    conn.commit()


def drop_all(conn: sqlite3.Connection) -> None:
    """Drop all application tables (useful for full resets)."""
    conn.executescript(DROP_ALL)
    conn.commit()