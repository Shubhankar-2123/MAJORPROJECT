import os
import json
from typing import Dict, Any, Optional, Tuple
from urllib.parse import quote
from rapidfuzz import process


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/")


class TextToSignService:
    def __init__(self, base_data_dir: Optional[str] = None, uploads_dir: Optional[str] = None):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        # Search for all Frames_Word_Level_* directories or single Frames_Word_Level
        data_dir = os.path.join(project_root, "data")
        self.frames_base_dirs = []
        
        if base_data_dir:
            self.frames_base_dirs = [base_data_dir]
        else:
            # Check for Frames_Word_Level_1, _2, _3, etc. and Frames_Word_Level
            for item in os.listdir(data_dir) if os.path.isdir(data_dir) else []:
                if item.startswith("Frames_Word_Level"):
                    full_path = os.path.join(data_dir, item)
                    if os.path.isdir(full_path):
                        self.frames_base_dirs.append(full_path)
            # Sort to ensure consistent order: Frames_Word_Level first, then _1, _2, etc.
            self.frames_base_dirs.sort()
        
        # Legacy directories (kept only as last resort)
        self.legacy_primary_dir = os.path.join(project_root, "data", "text_to_sign")
        self.legacy_fallback_dir = os.path.join(project_root, "isl_project - Copy", "dataset")
        
        # Uploads directory for custom signs
        self.uploads_dir = uploads_dir or os.path.join(project_root, "uploads")

        # Will be used for serving; for Frames_Word_Level this is the base
        self._video_base = None

        # Build dataset index from preferred source
        self.dataset = self._load_dataset()
        self._stopwords = {
            "are", "is", "am", "the", "a", "an", "of", "to", "in", "on", "and", "or",
        }

    def _load_dataset(self) -> Dict[str, Any]:
        """Build a dataset mapping from words to video relpaths.

        Priority:
        1) data/Frames_Word_Level*: scan for videos recursively, ignore images.
        2) Legacy JSON datasets (data/text_to_sign/sign_dataset.json or isl_project - Copy/dataset/sign_dataset.json)
        """
        # 1) Try Frames_Word_Level* scanning
        if self.frames_base_dirs:
            index: Dict[str, Any] = {}
            video_exts = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")

            def _normalize_text(text: str) -> str:
                import re as _re
                text = (text or "").lower().strip()
                text = _re.sub(r"[^a-z0-9\s]", " ", text)
                text = _re.sub(r"\s+", " ", text).strip()
                return text

            # Scan all Frames_Word_Level_* directories
            for frames_dir in self.frames_base_dirs:
                for root, _dirs, files in os.walk(frames_dir):
                    for name in files:
                        if not name.lower().endswith(video_exts):
                            # ignore images and other non-video files
                            continue
                        abs_path = os.path.join(root, name)
                        rel = os.path.relpath(abs_path, frames_dir)
                        rel = _normalize_path(rel)
                        folder_phrase = os.path.basename(os.path.dirname(abs_path))
                        stem = os.path.splitext(name)[0]
                        candidates = {
                            folder_phrase,
                            stem,
                            stem.replace("_", " "),
                            folder_phrase.replace("_", " "),
                        }
                        for cand in candidates:
                            key = _normalize_text(cand)
                            if not key:
                                continue
                            # Only set first occurrence to keep a stable mapping
                            if key not in index:
                                # Store both relative path and which base dir it's from
                                index[key] = {"path": rel, "base_dir": frames_dir}
            if index:
                self._video_base = self.frames_base_dirs[0] if self.frames_base_dirs else None
                return index

        # 2) Legacy JSON datasets
        for base in [self.legacy_primary_dir, self.legacy_fallback_dir]:
            json_path = os.path.join(base, "sign_dataset.json")
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._video_base = os.path.join(base, "videos")
                return data
        return {}

    def is_ready(self) -> bool:
        return bool(self.dataset)

    def find_word(self, user_input: str) -> Tuple[Optional[str], Optional[str]]:
        """Returns (canonical_word, error_message). If canonical_word is None, error contains info."""
        if not user_input:
            return None, "Empty input"
        user_input = user_input.lower().strip()

        # Skip common stopwords (no error)
        if user_input in self._stopwords:
            return None, "skip"

        # Direct match
        if user_input in self.dataset:
            return user_input, None

        # Synonym match
        for word, details in self.dataset.items():
            synonyms = [s.lower() for s in details.get("synonyms", [])]
            if user_input in synonyms:
                return word, None

        # Fuzzy match (conservative): only auto-map on very high confidence and longer queries
        # This reduces collisions like 'how' -> 'who' when the former is missing.
        closest, score, _ = process.extractOne(user_input, self.dataset.keys())
        if score and score >= 99 and len(user_input) >= 4:
            return closest, None
        if score and score >= 95:
            return None, f"'{user_input}' not found. Did you mean '{closest}'?"
        return None, f"'{user_input}' not found."

    def resolve_word_video(self, word: str) -> Optional[Dict[str, str]]:
        entry = self.dataset.get(word)
        if not entry:
            return None
        # Accept various schemas
        rel = None
        base_dir_override = None
        if isinstance(entry, dict):
            rel = entry.get("path") or entry.get("video_path") or entry.get("file")
            base_dir_override = entry.get("base_dir")
        elif isinstance(entry, str):
            rel = entry
        if not rel:
            return None
        rel = _normalize_path(rel)
        # If we built from Frames_Word_Level*, use the stored base_dir or _video_base
        base_to_use = base_dir_override or self._video_base
        if base_to_use and os.path.isdir(base_to_use):
            abs_path = os.path.join(base_to_use, rel)
            if os.path.exists(abs_path):
                url = f"/frames_video/{quote(rel)}"
                filename = os.path.basename(rel)
                return {
                    "word": word,
                    "filename": filename,
                    "abs_path": abs_path,
                    "url": url,
                }

        # Legacy behavior (filename under videos/)
        filename = os.path.basename(rel)
        abs_path = os.path.join(base_to_use, filename) if (base_to_use and not os.path.isabs(rel)) else rel
        if not abs_path or not os.path.exists(abs_path):
            candidate = os.path.join(os.path.dirname(base_to_use or ""), rel)
            if os.path.exists(candidate):
                abs_path = candidate
            else:
                return None
        url = f"/tts_video/{quote(filename)}"
        return {
            "word": word,
            "filename": filename,
            "abs_path": abs_path,
            "url": url,
        }

    def resolve_word_video_with_custom(
        self, 
        word: str, 
        user_id: Optional[int] = None
    ) -> Optional[Dict[str, str]]:
        """
        Resolve word video with custom sign fallback logic.
        
        Priority:
        1. User's custom sign (if user_id provided)
        2. Default dataset video
        
        Args:
            word: Word to look up
            user_id: Optional user ID for custom sign lookup
            
        Returns:
            Dictionary with word, filename, abs_path, url
            None if not found
        """
        # 1. Check for custom sign
        if user_id is not None:
            try:
                from database.custom_signs_models import get_custom_sign
                custom = get_custom_sign(user_id, word)
                if custom and custom.get("video_path"):
                    video_path = custom["video_path"]
                    abs_path = os.path.join(self.uploads_dir, video_path)
                    if os.path.exists(abs_path):
                        filename = os.path.basename(video_path)
                        url = f"/custom_sign_video/{user_id}/{quote(word)}"
                        return {
                            "word": word,
                            "filename": filename,
                            "abs_path": abs_path,
                            "url": url,
                            "is_custom": True,
                        }
            except Exception:
                # If custom sign lookup fails, fall through to default
                pass
        
        # 2. Fall back to default dataset
        result = self.resolve_word_video(word)
        if result:
            result["is_custom"] = False
        return result
