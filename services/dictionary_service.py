import os
from typing import Dict, List, Optional
from urllib.parse import quote

VIDEO_EXTS = (".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm")
WEB_VIDEO_EXTS = (".mp4", ".webm", ".m4v")


def _normalize_text(text: str) -> str:
    return (text or "").strip()


def _first_video_in_dir(root: str) -> Optional[str]:
    def _priority(name: str):
        ext = os.path.splitext(name)[1].lower()
        if ext in WEB_VIDEO_EXTS:
            return (0, WEB_VIDEO_EXTS.index(ext), name.lower())
        return (1, ext, name.lower())

    video_names = [n for n in os.listdir(root) if n.lower().endswith(VIDEO_EXTS)]
    for name in sorted(video_names, key=_priority):
        return os.path.join(root, name)
    return None


def _safe_relpath(base: str, abs_path: str) -> Optional[str]:
    try:
        rel = os.path.relpath(abs_path, base)
        if rel.startswith(".."):
            return None
        return rel.replace("\\", "/")
    except Exception:
        return None


def _is_sentence_label(label: str) -> bool:
    normalized = (label or "").replace("_", " ").strip()
    return len([part for part in normalized.split() if part]) > 1


def build_dictionary_entries(
    dynamic_dirs: List[str],
    frame_word_dirs: Optional[List[str]],
    user_signs_dir: Optional[str],
) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    seen = set()

    # Dynamic folders: sentences only.
    for base in dynamic_dirs:
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            folder = os.path.basename(root)
            if not folder:
                continue
            video = _first_video_in_dir(root)
            if not video:
                continue
            if not _is_sentence_label(folder):
                continue
            kind = "sentence"
            rel = _safe_relpath(base, video)
            if not rel:
                continue
            key = f"dyn::{kind}::{folder.lower()}"
            if key in seen:
                continue
            seen.add(key)
            entries.append({
                "word": _normalize_text(folder),
                "type": kind,
                "url": f"/dyn_video/{quote(rel)}",
                "source": "system",
            })

    # Frames_Word_Level* folders: words only.
    for base in frame_word_dirs or []:
        if not os.path.isdir(base):
            continue
        for root, _dirs, files in os.walk(base):
            folder = os.path.basename(root)
            for name in sorted(files):
                if not name.lower().endswith(VIDEO_EXTS):
                    continue
                stem = os.path.splitext(name)[0].replace("_", " ").strip()
                candidate = folder.replace("_", " ").strip() if folder else stem
                if _is_sentence_label(candidate):
                    continue
                label = candidate or stem
                rel = _safe_relpath(base, os.path.join(root, name))
                if not rel:
                    continue
                key = f"frames::word::{label.lower()}"
                if key in seen:
                    continue
                seen.add(key)
                entries.append({
                    "word": _normalize_text(label),
                    "type": "word",
                    "url": f"/frames_video/{quote(rel)}",
                    "source": "system",
                })
                break

    # User custom signs
    if user_signs_dir and os.path.isdir(user_signs_dir):
        for root, _dirs, files in os.walk(user_signs_dir):
            for name in sorted(files):
                if not name.lower().endswith(VIDEO_EXTS):
                    continue
                stem = os.path.splitext(name)[0].replace("_", " ")
                rel = _safe_relpath(user_signs_dir, os.path.join(root, name))
                if not rel:
                    continue
                key = f"user::word::{stem.lower()}::{rel.lower()}"
                if key in seen:
                    continue
                seen.add(key)
                entries.append({
                    "word": _normalize_text(stem),
                    "type": "word",
                    "url": f"/user_signs/{quote(rel)}",
                    "source": "user",
                })

    return entries
