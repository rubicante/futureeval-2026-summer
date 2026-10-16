"""Unit tests for prediction-market anchoring (research/market_anchor.py)."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from research.market_anchor import (
    AnchorMatch,
    AnchorVerdicts,
    Candidate,
    MarketAnchorClient,
    anchor_modes,
    anchor_report,
    search_terms,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _question(
    title="Will Reform UK win the Makerfield parliamentary by-election on June 18, 2026?",
    body="Resolves YES if the Reform UK candidate wins the by-election.",
):
    return SimpleNamespace(title=title, body=body, resolve_time="2026-06-19T00:00:00Z")


class TestAnchorModes:
    def test_empty(self):
        assert anchor_modes(None) == frozenset()
        assert anchor_modes("") == frozenset()

    def test_flags(self):
        assert anchor_modes("markets") == frozenset({"markets"})
        assert anchor_modes("markets, polls") == frozenset({"markets", "polls"})


class TestSearchTerms:
    def test_strips_stopwords_and_dates(self):
        queries = search_terms(
            "Will Reform UK win the Makerfield parliamentary by-election on June 18, 2026?"
        )
        assert queries
        assert "Will" not in queries[0].split()
        assert "Reform" in queries[0] and "Makerfield" in queries[0]

    def test_strips_parentheticals(self):
        queries = search_terms("How much will X exceed Y in these two weeks (Jun 15 - Jun 26)?")
        assert all("Jun 15" not in q for q in queries)

    def test_empty_title(self):
        assert search_terms("") == []

    def test_proper_noun_query_added(self):
        queries = search_terms("Will an Israel–Hezbollah ceasefire enter into effect before September 1, 2026?")
        assert len(queries) >= 1
        assert any("Israel" in q for q in queries)


def _transport(handler):
    return httpx.MockTransport(handler)


class TestPolymarketSearch:
    async def test_parses_and_filters_closed(self):
        payload = {
            "events": [
                {
                    "closed": False,
                    "title": "Makerfield by-election Winner",
                    "markets": [
                        {
                            "id": "111",
                            "closed": False,
                            "question": "Will Robert Kenyon win?",
                            "outcomePrices": '["0.23", "0.77"]',
                            "clobTokenIds": '["tok-yes", "tok-no"]',
                            "volume": "50000.5",
                            "description": "Resolves YES if Kenyon wins.",
                        },
                        {"id": "112", "closed": True, "question": "closed market"},
                    ],
                },
                {"closed": True, "title": "old event", "markets": [{"id": "99"}]},
            ]
        }

        def handler(request):
            return httpx.Response(200, json=payload)

        client = MarketAnchorClient()
        client._client = httpx.AsyncClient(transport=_transport(handler))
        try:
            out = await client.polymarket_search("Makerfield")
        finally:
            await client.close()
        assert len(out) == 1
        c = out[0]
        assert c.market_id == "pm:111"
        assert c.yes_price == pytest.approx(0.23)
        assert c.yes_token == "tok-yes"
        assert c.volume == pytest.approx(50000.5)


class TestManifoldSearch:
    async def test_filters_thin_resolved_nonbinary(self):
        payload = [
            {"id": "a", "question": "good", "outcomeType": "BINARY",
             "probability": 0.4, "uniqueBettorCount": 25, "isResolved": False},
            {"id": "b", "question": "thin", "outcomeType": "BINARY",
             "probability": 0.5, "uniqueBettorCount": 3, "isResolved": False},
            {"id": "c", "question": "resolved", "outcomeType": "BINARY",
             "probability": 0.9, "uniqueBettorCount": 50, "isResolved": True},
            {"id": "d", "question": "mc", "outcomeType": "MULTIPLE_CHOICE",
             "uniqueBettorCount": 50, "isResolved": False},
        ]

        def handler(request):
            return httpx.Response(200, json=payload)

        client = MarketAnchorClient()
        client._client = httpx.AsyncClient(transport=_transport(handler))
        try:
            out = await client.manifold_search("test")
        finally:
            await client.close()
        assert [c.market_id for c in out] == ["mf:a"]
        assert out[0].n_bettors == 25


class TestPolymarketPrice:
    async def test_median_of_trailing_window(self):
        import datetime as dt

        now = dt.datetime.now(dt.UTC).timestamp()
        history = {"history": [
            {"t": now - 3600 * 30, "p": 0.9},   # outside 24h
            {"t": now - 3600 * 20, "p": 0.30},
            {"t": now - 3600 * 10, "p": 0.40},
            {"t": now - 3600 * 1, "p": 0.35},
        ]}

        def handler(request):
            return httpx.Response(200, json=history)

        client = MarketAnchorClient()
        client._client = httpx.AsyncClient(transport=_transport(handler))
        try:
            price = await client.polymarket_price("tok")
        finally:
            await client.close()
        assert price == pytest.approx(0.35)

    async def test_falls_back_to_72h_then_none(self):
        import datetime as dt

        now = dt.datetime.now(dt.UTC).timestamp()

        def handler_72h(request):
            return httpx.Response(200, json={"history": [{"t": now - 3600 * 48, "p": 0.6}]})

        client = MarketAnchorClient()
        client._client = httpx.AsyncClient(transport=_transport(handler_72h))
        try:
            assert await client.polymarket_price("tok") == pytest.approx(0.6)
        finally:
            await client.close()

        def handler_empty(request):
            return httpx.Response(200, json={"history": []})

        client = MarketAnchorClient()
        client._client = httpx.AsyncClient(transport=_transport(handler_empty))
        try:
            assert await client.polymarket_price("tok") is None
        finally:
            await client.close()


def _candidate(market_id="pm:1", question="Will Kenyon win?", price=0.23, token=None):
    return Candidate(
        source="polymarket", market_id=market_id, question=question,
        yes_price=price, yes_token=token, volume=41000.0, n_bettors=None,
        description="Resolves YES if Kenyon wins the 2026 by-election.",
    )


class _FakeGateway:
    def __init__(self, verdicts):
        self._verdicts = verdicts
        self.prompts: list[str] = []

    async def extract_structured(self, request, output_type):
        from research.market_anchor import SearchTerms

        self.prompts.append(request.prompt)
        if output_type is SearchTerms:
            return SearchTerms(queries=["Kenyon Makerfield"])
        return self._verdicts


def _patch_searches(monkeypatch, candidates, manifold=()):
    async def fake_pm(self, query, *, include_closed=False):
        return list(candidates)

    async def fake_mf(self, query, *, include_closed=False):
        return list(manifold)

    monkeypatch.setattr(MarketAnchorClient, "polymarket_search", fake_pm)
    monkeypatch.setattr(MarketAnchorClient, "manifold_search", fake_mf)


class TestAnchorReport:
    async def test_exact_match_produces_block_and_meta(self, monkeypatch):
        _patch_searches(monkeypatch, [_candidate()])
        gateway = _FakeGateway(AnchorVerdicts(matches=[
            AnchorMatch(market_id="pm:1", congruence="exact", caveat=""),
        ]))
        result = await anchor_report(_question(), gateway)
        assert result is not None
        block, meta = result
        assert "Prediction-market anchors" in block
        assert "YES 0.23" in block
        assert meta[0]["market_id"] == "pm:1"
        assert meta[0]["congruence"] == "exact"

    async def test_unrelated_dropped(self, monkeypatch):
        _patch_searches(monkeypatch, [_candidate()])
        gateway = _FakeGateway(AnchorVerdicts(matches=[
            AnchorMatch(market_id="pm:1", congruence="unrelated", caveat="different race"),
        ]))
        assert await anchor_report(_question(), gateway) is None

    async def test_informative_caveat_in_block(self, monkeypatch):
        _patch_searches(monkeypatch, [_candidate()])
        gateway = _FakeGateway(AnchorVerdicts(matches=[
            AnchorMatch(market_id="pm:1", congruence="informative",
                        caveat="deadline Jun 30, question asks Sep 1"),
        ]))
        block, meta = await anchor_report(_question(), gateway)
        assert "caveat: deadline Jun 30" in block

    async def test_no_candidates_returns_none(self, monkeypatch):
        _patch_searches(monkeypatch, [])
        gateway = _FakeGateway(AnchorVerdicts(matches=[]))
        assert await anchor_report(_question(), gateway) is None
        assert len(gateway.prompts) == 1  # terms call only, no gate call

    async def test_gate_error_returns_none(self, monkeypatch):
        _patch_searches(monkeypatch, [_candidate()])

        class Broken:
            async def extract_structured(self, request, output_type):
                raise RuntimeError("llm down")

        assert await anchor_report(_question(), Broken()) is None

    async def test_financial_series_question_skipped(self, monkeypatch):
        _patch_searches(monkeypatch, [_candidate()])
        gateway = _FakeGateway(AnchorVerdicts(matches=[
            AnchorMatch(market_id="pm:1", congruence="exact"),
        ]))
        q = _question(body="See https://fred.stlouisfed.org/series/DGS10 for data.")
        assert await anchor_report(q, gateway) is None
        assert gateway.prompts == []

    async def test_overlap_ranking_beats_volume(self, monkeypatch):
        from research.market_anchor import MAX_GATE_CANDIDATES

        junk = [
            _candidate(market_id=f"pm:junk{i}", question="Will Trump insult someone today?")
            for i in range(MAX_GATE_CANDIDATES + 2)
        ]
        right = Candidate(
            source="polymarket", market_id="pm:right",
            question="Will Reform UK win the Makerfield by-election?",
            yes_price=0.2, yes_token=None, volume=100.0, n_bettors=None,
            description="",
        )
        _patch_searches(monkeypatch, [*junk, right])
        gateway = _FakeGateway(AnchorVerdicts(matches=[
            AnchorMatch(market_id="pm:right", congruence="exact"),
        ]))
        block, meta = await anchor_report(_question(), gateway)
        assert meta[0]["market_id"] == "pm:right"
        # The gate prompt saw the overlapping market despite the junk flood.
        assert "pm:right" in gateway.prompts[-1]

    async def test_llm_terms_failure_falls_back_to_heuristic(self, monkeypatch):
        from research.market_anchor import llm_search_terms

        class Broken:
            async def extract_structured(self, request, output_type):
                raise RuntimeError("llm down")

        queries = await llm_search_terms(Broken(), "Will Reform UK win the Makerfield by-election?")
        assert queries and any("Makerfield" in q for q in queries)

    async def test_search_failure_tolerated(self, monkeypatch):
        async def broken_pm(self, query, *, include_closed=False):
            raise httpx.ConnectError("down")

        async def fake_mf(self, query, *, include_closed=False):
            return [Candidate(source="manifold", market_id="mf:9", question="q?",
                              yes_price=0.4, yes_token=None, volume=None,
                              n_bettors=30, description="")]

        monkeypatch.setattr(MarketAnchorClient, "polymarket_search", broken_pm)
        monkeypatch.setattr(MarketAnchorClient, "manifold_search", fake_mf)
        gateway = _FakeGateway(AnchorVerdicts(matches=[
            AnchorMatch(market_id="mf:9", congruence="exact"),
        ]))
        block, meta = await anchor_report(_question(), gateway)
        assert "manifold" in block
        assert "30 bettors (play-money)" in block
