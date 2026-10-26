"""Disk cache for search queries and results."""

from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
from typing import List, Optional
from app.tools.search.base import SearchResult

CACHE_DIR = Path(".cache/search")


class SearchCache:
    def __init__(self, ttl_hours: int = 24):
        self.ttl_seconds = ttl_hours * 3600
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, query: str, backend: str) -> Path:
        key_str = f"{backend}:{query.strip().lower()}"
        digest = hashlib.sha256(key_str.encode("utf-8")).hexdigest()
        return CACHE_DIR / f"{digest}.json"

    def get(self, query: str, backend: str) -> Optional[List[SearchResult]]:
        path = self._get_cache_key(query, backend)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cached_at = data.get("timestamp", 0)
            if time.time() - cached_at > self.ttl_seconds:
                return None
            return [SearchResult(**item) for item in data.get("results", [])]
        except Exception:
            return None

    def set(self, query: str, backend: str, results: List[SearchResult]) -> None:
        path = self._get_cache_key(query, backend)
        try:
            data = {
                "query": query,
                "backend": backend,
                "timestamp": time.time(),
                "results": [r.model_dump() for r in results],
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass
