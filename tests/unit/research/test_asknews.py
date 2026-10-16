"""Unit tests for AskNewsSearcher rate limiting."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

from research.asknews import AskNewsSearcher, _TokenBucket


class TestTokenBucket:
    def test_initial_burst_pass_through(self) -> None:
        """First *burst* acquires should complete with negligible delay."""
        bucket = _TokenBucket(rate=0.5, burst=2)

        async def run() -> float:
            t0 = time.monotonic()
            await bucket.acquire()
            await bucket.acquire()
            return time.monotonic() - t0

        elapsed = asyncio.run(run())
        assert elapsed < 0.05, f"burst should be instant, took {elapsed:.3f}s"

    def test_third_acquire_waits(self) -> None:
        """Third acquire after burst exhaustion should wait ~2s."""
        bucket = _TokenBucket(rate=0.5, burst=2)

        async def run() -> float:
            await bucket.acquire()  # uses burst
            await bucket.acquire()  # uses burst
            t0 = time.monotonic()
            await bucket.acquire()  # should wait
            return time.monotonic() - t0

        elapsed = asyncio.run(run())
        assert 1.8 <= elapsed <= 2.4, f"expected ~2s delay, took {elapsed:.3f}s"

    def test_refill_with_slower_rate(self) -> None:
        """Partial refill after a sleep reduces the acquire wait proportionally."""
        bucket = _TokenBucket(rate=1.0, burst=1)  # 1 token/s

        async def run() -> float:
            await bucket.acquire()  # exhaust burst
            t0 = time.monotonic()
            await asyncio.sleep(0.6)  # refill 0.6 tokens
            await bucket.acquire()  # needs 0.4s more
            return time.monotonic() - t0

        elapsed = asyncio.run(run())
        assert 0.9 <= elapsed <= 1.2, f"expected ~1.0s (0.6 sleep + 0.4 wait), took {elapsed:.3f}s"

    def test_concurrent_callers_serialised(self) -> None:
        """Three concurrent callers — total time accumulates per caller."""
        bucket = _TokenBucket(rate=1.0, burst=0)  # 1 token/s, no burst

        async def worker() -> None:
            await bucket.acquire()

        async def run() -> float:
            t0 = time.monotonic()
            await asyncio.gather(worker(), worker(), worker())
            return time.monotonic() - t0

        elapsed = asyncio.run(run())
        # 3 acquires with 0 burst — each reserves next token: 1s, 2s, 3s sleeps all in parallel
        assert 2.8 <= elapsed <= 3.4, (
            f"expected ~3s for 3 concurrent (no burst), took {elapsed:.3f}s"
        )


class TestAskNewsSearcherRateLimiting:
    def test_search_acquires_bucket_twice(self) -> None:
        """Verify search() calls acquire before each API invocation."""
        import research.asknews as module

        # Replace the bucket with a spy
        acquire_calls: list[None] = []
        original_bucket = module.AskNewsSearcher._bucket

        class SpyBucket:
            async def acquire(self) -> None:
                acquire_calls.append(None)

        try:
            module.AskNewsSearcher._bucket = SpyBucket()  # type: ignore[assignment]

            # Mock the SDK + search_news
            with patch.object(module, "AsyncAskNewsSDK") as mock_sdk_cls:
                mock_sdk = mock_sdk_cls.return_value
                mock_result = AsyncMock()
                mock_result.as_dicts = []
                mock_sdk.news.search_news = AsyncMock(return_value=mock_result)

                searcher = AskNewsSearcher(api_key="test-key")
                result = asyncio.run(searcher.search("test query"))

                assert len(acquire_calls) == 2, f"expected 2 acquires, got {len(acquire_calls)}"
                assert isinstance(result, list)
        finally:
            module.AskNewsSearcher._bucket = original_bucket
