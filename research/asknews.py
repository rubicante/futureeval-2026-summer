"""AskNewsSearcher — news article retrieval via AskNews SDK."""

from __future__ import annotations

import asyncio
import logging
import os
import time

from asknews_sdk import AsyncAskNewsSDK

from research.result import ResearchResult

logger = logging.getLogger(__name__)


class _TokenBucket:
    """Async-safe token bucket rate limiter.

    Models AskNews Pro plan: 0.5 tokens/s (1 req/2 s), burst of 2.
    Shared across all searcher instances so parallel questions don't
    exceed the plan limit.
    """

    def __init__(self, rate: float, burst: int) -> None:
        self._rate = rate
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available, then consume it."""
        async with self._lock:
            now = time.monotonic()
            self._tokens = min(
                float(self._burst),
                self._tokens + (now - self._last_refill) * self._rate,
            )
            self._last_refill = now
            if self._tokens < 1:
                deficit = 1 - self._tokens
                self._tokens = 0.0
                self._last_refill = now + deficit / self._rate
            else:
                deficit = 0.0
                self._tokens -= 1.0
        if deficit > 0:
            await asyncio.sleep(deficit / self._rate)


class AskNewsSearcher:
    _bucket = _TokenBucket(rate=0.5, burst=2)

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    @classmethod
    def from_env(cls) -> AskNewsSearcher:
        key = os.getenv("ASKNEWS_API_KEY")
        if not key:
            raise RuntimeError("ASKNEWS_API_KEY not set")
        return cls(key)

    async def search(self, query: str) -> list[ResearchResult]:
        logger.debug("AskNews search: %s", query[:120])
        ask = AsyncAskNewsSDK(api_key=self.api_key, scopes={"news"})
        await self._bucket.acquire()
        hot = await ask.news.search_news(
            query=query,
            n_articles=6,
            return_type="both",
            strategy="latest news",
        )
        await self._bucket.acquire()
        historical = await ask.news.search_news(
            query=query,
            n_articles=10,
            return_type="both",
            strategy="news knowledge",
        )

        results: list[ResearchResult] = []
        seen: set[str] = set()
        for articles in (hot.as_dicts, historical.as_dicts):
            for a in articles or []:
                article_url = str(a.article_url)
                if article_url in seen:
                    continue
                seen.add(article_url)
                pub = a.pub_date.strftime("%B %d, %Y %I:%M %p")
                results.append(
                    ResearchResult(
                        title=a.eng_title,
                        url=article_url,
                        text=(
                            f"{a.summary}\n"
                            f"Original language: {a.language}\n"
                            f"Publish date: {pub}\n"
                            f"Source: {a.source_id}"
                        ),
                    )
                )
        result_list = sorted(results, key=lambda r: r.score or 0, reverse=True)
        logger.debug("AskNews: %d results for query: %s", len(result_list), query[:120])
        return result_list
