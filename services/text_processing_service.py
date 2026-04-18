import hashlib
import json
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

try:
    import requests  # type: ignore
    REQUESTS_AVAILABLE = True
except Exception:
    requests = None  # type: ignore
    REQUESTS_AVAILABLE = False

from database.sqlite import get_connection


DEFAULT_STOPWORDS = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "of", "to", "in", "on", "for", "with", "at", "by", "from",
    "as", "it", "this", "that", "these", "those", "i", "you", "he", "she", "we",
    "they", "me", "my", "your", "his", "her", "our", "their", "us", "them",
}


def _normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _hash_key(text: str, enable_llm: bool) -> str:
    raw = f"{text}::llm={int(enable_llm)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class TextProcessingService:
    def __init__(
        self,
        enable_llm: bool = False,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_sec: int = 10,
        cache_ttl_sec: int = 86400,
        stopwords: Optional[set] = None,
        db_path: Optional[str] = None,
        min_match_ratio: float = 0.5,
    ) -> None:
        self.enable_llm = bool(enable_llm)
        self.api_url = (api_url or "").strip() or None
        self.api_key = (api_key or "").strip() or None
        self.model = (model or "").strip() or None
        self.timeout_sec = max(1, int(timeout_sec))
        self.cache_ttl_sec = max(60, int(cache_ttl_sec))
        self.stopwords = stopwords or DEFAULT_STOPWORDS
        self.db_path = db_path
        self.min_match_ratio = max(0.0, min(float(min_match_ratio), 1.0))

    def simplify_rule_based(self, text: str) -> List[str]:
        cleaned = _normalize_text(text)
        if not cleaned:
            return []
        tokens = [t for t in cleaned.split(" ") if t and t not in self.stopwords]
        return tokens

    def simplify_with_llm(self, text: str) -> Optional[List[str]]:
        if not self.enable_llm:
            return None
        if not (self.api_url and self.api_key and REQUESTS_AVAILABLE):
            return None

        prompt = (
            "Simplify this sentence into core meaningful words suitable for sign language. "
            "Return only comma-separated words.\n\n"
            f"Sentence: {text}"
        )

        payload: Dict[str, str] = {"prompt": prompt}
        if self.model:
            payload["model"] = self.model

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(self.api_url, headers=headers, json=payload, timeout=self.timeout_sec)
            if resp.status_code >= 400:
                return None
            data = resp.json() if resp.content else {}
            text_out = (data.get("text") or data.get("response") or "").strip()
            if not text_out:
                return None
            tokens = [t.strip().lower() for t in text_out.split(",") if t.strip()]
            return tokens or None
        except Exception:
            return None

    def _get_cache(self, cache_key: str) -> Optional[Dict[str, str]]:
        conn = get_connection(self.db_path)
        try:
            row = conn.execute(
                "SELECT simplified_json, source, created_at FROM text_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if not row:
                return None
            created_at = row["created_at"]
            if created_at:
                try:
                    dt = datetime.fromisoformat(created_at)
                    if datetime.utcnow() - dt > timedelta(seconds=self.cache_ttl_sec):
                        return None
                except Exception:
                    pass
            return {
                "simplified_json": row["simplified_json"],
                "source": row["source"],
            }
        finally:
            conn.close()

    def _set_cache(self, cache_key: str, tokens: List[str], source: str) -> None:
        conn = get_connection(self.db_path)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO text_cache (cache_key, simplified_json, source, created_at) VALUES (?, ?, ?, ?)",
                (cache_key, json.dumps(tokens), source, datetime.utcnow().isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    def _is_insufficient(self, tokens: List[str], available_words: Optional[set]) -> bool:
        if not tokens:
            return True
        if not available_words:
            return True
        matched = [t for t in tokens if t in available_words]
        if not matched:
            return True
        ratio = len(matched) / max(len(tokens), 1)
        return ratio < self.min_match_ratio

    def hybrid_simplify(self, text: str, available_words: Optional[set] = None) -> Dict[str, object]:
        cleaned = _normalize_text(text)
        if not cleaned:
            return {"tokens": [], "source": "empty"}

        cache_key = _hash_key(cleaned, self.enable_llm)
        cached = self._get_cache(cache_key)
        if cached:
            try:
                tokens = json.loads(cached["simplified_json"] or "[]")
            except Exception:
                tokens = []
            return {"tokens": tokens, "source": cached.get("source", "cache")}

        tokens = self.simplify_rule_based(cleaned)
        source = "rule_based"

        if self.enable_llm and self._is_insufficient(tokens, available_words):
            llm_tokens = self.simplify_with_llm(text)
            if llm_tokens:
                tokens = llm_tokens
                source = "llm"

        self._set_cache(cache_key, tokens, source)
        return {"tokens": tokens, "source": source}
