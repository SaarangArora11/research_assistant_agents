"""Search Router with ordered fallback chain, circuit breaker cooldowns, and deduplication."""

from __future__ import annotations
import asyncio
import time
import urllib.parse
from typing import Dict, List, Optional
from app.core.config import Settings, load_settings
from app.tools.search.base import BaseSearchEngine, SearchResult
from app.tools.search.cache import SearchCache
from app.tools.search.engines import (
    ArxivSearch,
    BraveSearch,
    DuckDuckGoSearch,
    ExaSearch,
    SearXNGSearch,
    SemanticScholarSearch,
    SerperSearch,
    TavilySearch,
    WikipediaSearch,
)


def _normalize_url(url: str) -> str:
    """Normalize URL for deduplication (strip fragment and trailing slash)."""
    parsed = urllib.parse.urlparse(url)
    clean_path = parsed.path.rstrip("/")
    return f"{parsed.netloc}{clean_path}?{parsed.query}".rstrip("?").lower()


class SearchRouter:
    """Intelligent search router that tries backends in order, with cooldowns and fallbacks."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or load_settings()
        self.cache = SearchCache(ttl_hours=self.settings.search_cache_ttl_hours)
        self.cooldowns: Dict[str, float] = {}  # backend -> timestamp until cooled
        self.cooldown_duration = 300.0  # 5 minutes on error
        self._init_engines()

    def _init_engines(self) -> None:
        self.engines: Dict[str, BaseSearchEngine] = {
            "ddg": DuckDuckGoSearch(),
            "searxng": SearXNGSearch(base_url=self.settings.searxng_url),
            "brave": BraveSearch(api_key=self.settings.brave_api_key),
            "tavily": TavilySearch(api_key=self.settings.tavily_api_key),
            "serper": SerperSearch(api_key=self.settings.serper_api_key),
            "exa": ExaSearch(api_key=self.settings.exa_api_key),
            "wikipedia": WikipediaSearch(),
            "arxiv": ArxivSearch(),
            "semantic_scholar": SemanticScholarSearch(api_key=self.settings.semantic_scholar_api_key),
        }

    def _is_cooling_down(self, name: str) -> bool:
        until = self.cooldowns.get(name, 0)
        return time.time() < until

    def _set_cooldown(self, name: str) -> None:
        self.cooldowns[name] = time.time() + self.cooldown_duration

    async def search(
        self,
        query: str,
        max_results: int = 5,
        prefer_academic: bool = False,
    ) -> List[SearchResult]:
        """Execute search with fallback chain across available engines."""
        query = query.strip()
        if not query:
            return []

        # Determine priority list
        order = list(self.settings.search_backends_order)

        # If academic intent detected, elevate arXiv and Semantic Scholar
        academic_keywords = {"arxiv", "paper", "study", "citation", "algorithm", "dataset", "benchmark"}
        if prefer_academic or any(kw in query.lower() for kw in academic_keywords):
            order = ["arxiv", "semantic_scholar"] + [k for k in order if k not in ("arxiv", "semantic_scholar")]

        # Filter engines that are configured and not currently in cooldown
        available = [
            k for k in order
            if k in self.engines and self.engines[k].is_configured() and not self._is_cooling_down(k)
        ]

        if not available:
            # If all are cooling down, reset cooldowns and try again
            self.cooldowns.clear()
            available = [k for k in order if k in self.engines and self.engines[k].is_configured()]

        # Check cache on first preferred backend
        if available:
            primary_backend = available[0]
            cached = self.cache.get(query, primary_backend)
            if cached:
                return cached[:max_results]

        # Single or Multi search
        if self.settings.multi_search and len(available) >= 2:
            return await self._multi_search(available[:2], query, max_results)

        # Fallback loop
        for name in available:
            engine = self.engines[name]
            try:
                results = await engine.search(query, max_results=max_results)
                if results:
                    self.cache.set(query, name, results)
                    return results[:max_results]
                else:
                    # Empty results: try next backend
                    continue
            except Exception as e:
                print(f"Search backend '{name}' failed for query '{query}': {e}. Cooling down.")
                self._set_cooldown(name)
                continue

        # Last resort fallback: Wikipedia
        try:
            wiki_results = await self.engines["wikipedia"].search(query, max_results=max_results)
            if wiki_results:
                return wiki_results
        except Exception:
            pass

        return []

    async def _multi_search(self, backends: List[str], query: str, max_results: int) -> List[SearchResult]:
        """Query multiple backends in parallel and merge results."""
        tasks = [self.engines[b].search(query, max_results=max_results) for b in backends]
        res_lists = await asyncio.gather(*tasks, return_exceptions=True)

        merged: List[SearchResult] = []
        seen_urls = set()

        for idx, res in enumerate(res_lists):
            if isinstance(res, list):
                for item in res:
                    norm = _normalize_url(item.url)
                    if norm not in seen_urls:
                        seen_urls.add(norm)
                        merged.append(item)
            elif isinstance(res, Exception):
                self._set_cooldown(backends[idx])

        return merged[:max_results]
