"""Implementations of various search engine backends."""

from __future__ import annotations
import asyncio
import re
import urllib.parse
import xml.etree.ElementTree as ET
from typing import List, Optional
import httpx
from app.tools.search.base import BaseSearchEngine, SearchResult


class DuckDuckGoSearch(BaseSearchEngine):
    name = "ddg"

    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        from duckduckgo_search import DDGS

        def _sync_search():
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(
                        SearchResult(
                            url=r.get("href", ""),
                            title=r.get("title", ""),
                            snippet=r.get("body", ""),
                            backend="ddg",
                        )
                    )
            return results

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_search)


class SearXNGSearch(BaseSearchEngine):
    name = "searxng"

    def __init__(self, base_url: str = ""):
        self.base_url = base_url.rstrip("/")

    def is_configured(self) -> bool:
        return bool(self.base_url)

    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        if not self.base_url:
            return []
        url = f"{self.base_url}/search"
        params = {"q": query, "format": "json"}
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("results", [])[:max_results]:
                results.append(
                    SearchResult(
                        url=item.get("url", ""),
                        title=item.get("title", ""),
                        snippet=item.get("content", ""),
                        backend="searxng",
                    )
                )
            return results


class BraveSearch(BaseSearchEngine):
    name = "brave"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        if not self.api_key:
            return []
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {"X-Subscription-Token": self.api_key, "Accept": "application/json"}
        params = {"q": query, "count": max_results}
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
            results = []
            web_results = data.get("web", {}).get("results", [])
            for item in web_results[:max_results]:
                results.append(
                    SearchResult(
                        url=item.get("url", ""),
                        title=item.get("title", ""),
                        snippet=item.get("description", ""),
                        backend="brave",
                    )
                )
            return results


class TavilySearch(BaseSearchEngine):
    name = "tavily"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        if not self.api_key:
            return []
        url = "https://api.tavily.com/search"
        payload = {"api_key": self.api_key, "query": query, "max_results": max_results}
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("results", [])[:max_results]:
                results.append(
                    SearchResult(
                        url=item.get("url", ""),
                        title=item.get("title", ""),
                        snippet=item.get("content", ""),
                        backend="tavily",
                    )
                )
            return results


class SerperSearch(BaseSearchEngine):
    name = "serper"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        if not self.api_key:
            return []
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        payload = {"q": query, "num": max_results}
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("organic", [])[:max_results]:
                results.append(
                    SearchResult(
                        url=item.get("link", ""),
                        title=item.get("title", ""),
                        snippet=item.get("snippet", ""),
                        backend="serper",
                    )
                )
            return results


class ExaSearch(BaseSearchEngine):
    name = "exa"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        if not self.api_key:
            return []
        url = "https://api.exa.ai/search"
        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        payload = {"query": query, "numResults": max_results, "useAutoprompt": True}
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("results", [])[:max_results]:
                results.append(
                    SearchResult(
                        url=item.get("url", ""),
                        title=item.get("title", ""),
                        snippet=item.get("text", "")[:300] if item.get("text") else "",
                        backend="exa",
                    )
                )
            return results


class WikipediaSearch(BaseSearchEngine):
    name = "wikipedia"

    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        url = "https://en.wikipedia.org/w/api.php"
        headers = {
            "User-Agent": "SynapseResearchAssistant/1.0 (https://github.com/research-assistant-agents; mailto:contact@example.com)",
            "Accept": "application/json",
        }
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": max_results,
        }
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("query", {}).get("search", []):
                clean_snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
                title = item.get("title", "")
                page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                results.append(
                    SearchResult(
                        url=page_url,
                        title=title,
                        snippet=clean_snippet,
                        backend="wikipedia",
                    )
                )
            return results


class ArxivSearch(BaseSearchEngine):
    name = "arxiv"

    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        # Remove special characters from query for arXiv syntax
        clean_query = re.sub(r"[^\w\s]", " ", query).strip()
        url = f"https://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(clean_query)}&max_results={max_results}"
        headers = {
            "User-Agent": "SynapseResearchAssistant/1.0 (https://github.com/research-assistant-agents)",
        }
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            results = []
            for entry in root.findall("atom:entry", ns):
                title = entry.findtext("atom:title", "", ns).strip()
                summary = entry.findtext("atom:summary", "", ns).strip()
                link = entry.findtext("atom:id", "", ns).strip()
                results.append(
                    SearchResult(
                        url=link,
                        title=title.replace("\n", " "),
                        snippet=summary.replace("\n", " ")[:350],
                        backend="arxiv",
                    )
                )
            return results


class SemanticScholarSearch(BaseSearchEngine):
    name = "semantic_scholar"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key

    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {"query": query, "limit": max_results, "fields": "title,abstract,url,year"}
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for paper in data.get("data", [])[:max_results]:
                paper_url = paper.get("url") or f"https://www.semanticscholar.org/paper/{paper.get('paperId')}"
                abstract = paper.get("abstract") or ""
                year = paper.get("year", "")
                title = paper.get("title", "")
                snippet = f"({year}) {abstract}" if year else abstract
                results.append(
                    SearchResult(
                        url=paper_url,
                        title=title,
                        snippet=snippet[:350],
                        backend="semantic_scholar",
                    )
                )
            return results
