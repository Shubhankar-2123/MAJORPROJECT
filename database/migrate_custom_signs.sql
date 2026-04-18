-- ========================================
-- CUSTOM SIGNS FEATURE MIGRATION SCRIPT
-- ========================================
-- This migration adds user-specific custom sign video support
-- Allows users to upload their own sign videos for specific words
-- Does NOT affect ML models or default dataset

-- Create custom_signs table
CREATE TABLE IF NOT EXISTS custom_signs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    word TEXT NOT NULL,
    video_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE(user_id, word)
);

-- Create index for fast user-based lookups
CREATE INDEX IF NOT EXISTS idx_custom_signs_user 
ON custom_signs(user_id);

-- Create composite index for word lookups per user
CREATE INDEX IF NOT EXISTS idx_custom_signs_user_word 
ON custom_signs(user_id, word);

-- ========================================
-- NOTES:
-- ========================================
-- 1. UNIQUE(user_id, word) ensures one custom video per word per user
-- 2. ON DELETE CASCADE ensures cleanup when user is deleted
-- 3. video_path stores relative path: uploads/custom_signs/user_<id>/<word>.mp4
-- 4. This table is completely isolated from predictions, feedback, and ML models
-- 5. Default dataset videos remain untouched in data/Frames_Word_Level
-- ========================================
