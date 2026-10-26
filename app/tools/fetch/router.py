"""Web page and document fetcher with trafilatura, PDF parsing, and Jina Reader fallback."""

from __future__ import annotations
import io
import re
from typing import Optional
import httpx
from pypdf import PdfReader
import trafilatura

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


class FetchRouter:
    """Fetches and extracts clean readable text from web pages and PDF documents."""

    def __init__(self, max_chars: int = 12000, timeout: float = 12.0):
        self.max_chars = max_chars
        self.timeout = timeout

    async def fetch_url(self, url: str) -> Optional[str]:
        """Fetch URL and extract clean text or markdown, using multiple strategies."""
        url = url.strip()
        if not url.startswith("http://") and not url.startswith("https://"):
            return None

        # Check if URL is a PDF
        if url.lower().endswith(".pdf") or "arxiv.org/pdf" in url:
            pdf_text = await self._fetch_pdf(url)
            if pdf_text:
                return pdf_text[: self.max_chars]

        # 1. Primary Strategy: direct HTTP fetch + trafilatura extraction
        try:
            headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200:
                    content_type = resp.headers.get("content-type", "").lower()
                    if "application/pdf" in content_type:
                        pdf_text = self._parse_pdf_bytes(resp.content)
                        if pdf_text:
                            return pdf_text[: self.max_chars]

                    extracted = trafilatura.extract(
                        resp.text,
                        include_links=False,
                        include_images=False,
                        output_format="txt",
                    )
                    if extracted and len(extracted.strip()) > 100:
                        return extracted.strip()[: self.max_chars]
        except Exception:
            pass

        # 2. Fallback Strategy: Jina Reader API (free, renders clean markdown for challenging pages)
        try:
            jina_url = f"https://r.jina.ai/{url}"
            headers = {"User-Agent": DEFAULT_USER_AGENT}
            async with httpx.AsyncClient(timeout=self.timeout + 3.0) as client:
                resp = await client.get(jina_url, headers=headers)
                if resp.status_code == 200 and len(resp.text.strip()) > 100:
                    # Clean up Jina header prefixes if present
                    text = resp.text.strip()
                    return text[: self.max_chars]
        except Exception:
            pass

        return None

    async def _fetch_pdf(self, url: str) -> Optional[str]:
        """Fetch and extract text from a PDF url."""
        try:
            headers = {"User-Agent": DEFAULT_USER_AGENT}
            async with httpx.AsyncClient(timeout=self.timeout + 5.0, follow_redirects=True) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200:
                    return self._parse_pdf_bytes(resp.content)
        except Exception:
            pass
        return None

    def _parse_pdf_bytes(self, content: bytes) -> Optional[str]:
        """Extract text from PDF byte stream."""
        try:
            reader = PdfReader(io.BytesIO(content))
            pages_text = []
            for i, page in enumerate(reader.pages[:10]):  # First 10 pages max
                txt = page.extract_text()
                if txt:
                    pages_text.append(txt)
            combined = "\n\n".join(pages_text).strip()
            return re.sub(r"\s+", " ", combined) if combined else None
        except Exception:
            return None


# Global singleton instance
fetch_router = FetchRouter()
