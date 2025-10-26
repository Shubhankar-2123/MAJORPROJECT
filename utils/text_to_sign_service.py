import os
import json
from typing import Dict, Any, Optional, Tuple
from urllib.parse import quote
from rapidfuzz import process


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/")


class TextToSignService:
    def __init__(self, base_data_dir: Optional[str] = None):
        # Prefer data/Frames_Word_Level; fallback to legacy locations if unavailable
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.frames_base_dir = base_data_dir or os.path.join(project_root, "data", "Frames_Word_Level")
        # Legacy directories (kept only as last resort)
        self.legacy_primary_dir = os.path.join(project_root, "data", "text_to_sign")
        self.legacy_fallback_dir = os.path.join(project_root, "isl_project - Copy", "dataset")

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
        1) data/Frames_Word_Level: scan for videos recursively, ignore images.
        2) Legacy JSON datasets (data/text_to_sign/sign_dataset.json or isl_project - Copy/dataset/sign_dataset.json)
        """
        # 1) Try Frames_Word_Level scanning
        if os.path.isdir(self.frames_base_dir):
            index: Dict[str, Any] = {}
            video_exts = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")

            def _normalize_text(text: str) -> str:
                import re as _re
                text = (text or "").lower().strip()
                text = _re.sub(r"[^a-z0-9\s]", " ", text)
                text = _re.sub(r"\s+", " ", text).strip()
                return text

            for root, _dirs, files in os.walk(self.frames_base_dir):
                for name in files:
                    if not name.lower().endswith(video_exts):
                        # ignore images and other non-video files
                        continue
                    abs_path = os.path.join(root, name)
                    rel = os.path.relpath(abs_path, self.frames_base_dir)
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
                            index[key] = {"path": rel}
            if index:
                self._video_base = self.frames_base_dir
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
        if isinstance(entry, dict):
            rel = entry.get("path") or entry.get("video_path") or entry.get("file")
        elif isinstance(entry, str):
            rel = entry
        if not rel:
            return None
        rel = _normalize_path(rel)
        # If we built from Frames_Word_Level, rel is a RELATIVE path under frames base
        if self._video_base and os.path.isdir(self._video_base) and os.path.commonpath([self._video_base]) == self._video_base:
            url = f"/frames_video/{quote(rel)}"
            filename = os.path.basename(rel)
            abs_path = os.path.join(self._video_base, rel)
            return {
                "word": word,
                "filename": filename,
                "abs_path": abs_path,
                "url": url,
            }

        # Legacy behavior (filename under videos/)
        filename = os.path.basename(rel)
        abs_path = os.path.join(self._video_base, filename) if (self._video_base and not os.path.isabs(rel)) else rel
        if not abs_path or not os.path.exists(abs_path):
            candidate = os.path.join(os.path.dirname(self._video_base or ""), rel)
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


