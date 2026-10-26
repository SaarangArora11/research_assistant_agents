"""Tests for search engines and fetch router."""

import pytest
from app.tools.fetch.router import FetchRouter
from app.tools.search.engines import WikipediaSearch
from app.tools.search.router import SearchRouter, _normalize_url


def test_url_normalization():
    u1 = "https://example.com/page/"
    u2 = "https://EXAMPLE.com/page?ref=1"
    u3 = "https://example.com/page#section"
    assert _normalize_url(u1) == "example.com/page"
    assert _normalize_url(u3) == "example.com/page"
    assert _normalize_url(u2) == "example.com/page?ref=1"


@pytest.mark.asyncio
async def test_wikipedia_search():
    wiki = WikipediaSearch()
    results = await wiki.search("Alan Turing", max_results=2)
    assert len(results) > 0
    assert "Turing" in results[0].title
    assert results[0].backend == "wikipedia"


@pytest.mark.asyncio
async def test_search_router_fallback():
    router = SearchRouter()
    # Query with prefer_academic=True
    results = await router.search("deep learning attention mechanism", max_results=3, prefer_academic=True)
    assert len(results) > 0
    assert results[0].url.startswith("http")


@pytest.mark.asyncio
async def test_fetch_router():
    fetcher = FetchRouter(max_chars=2000, timeout=10.0)
    # Fetch an accessible public URL
    text = await fetcher.fetch_url("https://en.wikipedia.org/wiki/Artificial_intelligence")
    assert text is not None
    assert len(text) > 100
    assert "intelligence" in text.lower()
