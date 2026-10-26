"""Base interfaces and data structures for search engines."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel


class SearchResult(BaseModel):
    """Standardized search result."""
    url: str
    title: str
    snippet: str
    backend: str
    published_date: Optional[str] = None
    score: Optional[float] = None


class BaseSearchEngine(ABC):
    """Abstract search engine provider."""

    name: str = "base"

    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Perform search and return standardized results."""
        pass

    def is_configured(self) -> bool:
        """Check if required API keys or endpoints are configured."""
        return True
