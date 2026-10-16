"""Unit tests for MetaculusClient pacing."""

from __future__ import annotations

import asyncio

import pytest

from metaculus.client import MetaculusClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("METACULUS_TOKEN", "test-token")
    c = MetaculusClient()
    yield c
    asyncio.run(c.close())


class TestDetailPacing:
    def test_detail_fetches_are_paced(self, client, monkeypatch):
        """Consecutive get_post_details calls start ≥ _DETAIL_PACING apart,
        regardless of caller — the pacing lives in the client."""
        monkeypatch.setattr(MetaculusClient, "_DETAIL_PACING", 0.05)
        starts: list[float] = []

        async def fake_get_json(url, params=None, retries=3):
            starts.append(asyncio.get_running_loop().time())
            return {}

        monkeypatch.setattr(client, "_get_json", fake_get_json)

        async def run():
            await client.get_post_details(1)
            await client.get_post_details(2)
            await asyncio.gather(client.get_post_details(3), client.get_post_details(4))

        asyncio.run(run())
        assert len(starts) == 4
        ordered = sorted(starts)
        gaps = [b - a for a, b in zip(ordered, ordered[1:], strict=False)]
        assert all(gap >= 0.045 for gap in gaps)  # small tolerance for timer slop

    def test_first_fetch_not_delayed(self, client, monkeypatch):
        monkeypatch.setattr(MetaculusClient, "_DETAIL_PACING", 10.0)

        async def fake_get_json(url, params=None, retries=3):
            return {}

        monkeypatch.setattr(client, "_get_json", fake_get_json)

        async def run():
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            await client.get_post_details(1)
            return loop.time() - t0

        assert asyncio.run(run()) < 1.0
