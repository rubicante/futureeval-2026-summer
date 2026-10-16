"""Unit tests for Tier-2 quant assist (research/quant_assist.py)."""

from __future__ import annotations

import datetime as dt

import numpy as np

import research.quant_assist as qa
from common.models import ForecastQuestion, QuestionType
from research.market_data import Bar, Point


class TestModes:
    def test_empty(self):
        assert qa.quant_modes(None) == frozenset()
        assert qa.quant_modes("") == frozenset()

    def test_single_and_combined(self):
        assert qa.quant_modes("series") == {"series"}
        assert qa.quant_modes("shadow,assist") == {"shadow", "assist"}
        assert qa.quant_modes(" shadow , assist ") == {"shadow", "assist"}


class TestDetect:
    def test_fred_and_yahoo(self):
        body = (
            "Resolves by [FRED](https://fred.stlouisfed.org/series/DGS10) and "
            "[Yahoo](https://finance.yahoo.com/quote/NQ%3DF/history/)."
        )
        assert qa.detect_series(body) == [("fred", "DGS10"), ("yahoo", "NQ=F")]

    def test_dedup_and_cap(self):
        body = " ".join(
            f"https://fred.stlouisfed.org/series/S{i}" for i in (1, 1, 2, 3, 4)
        )
        detected = qa.detect_series(body)
        assert len(detected) == qa.MAX_SERIES
        assert detected[0] == ("fred", "S1")

    def test_no_links(self):
        assert qa.detect_series("Will X happen?") == []


def _question(body: str) -> ForecastQuestion:
    return ForecastQuestion(
        title="What will X be?",
        body=body,
        question_type=QuestionType.NUMERIC,
        resolve_time=(dt.date.today() + dt.timedelta(days=30)).isoformat() + "T00:00:00Z",
    )


class FakeMarketData:
    def __init__(self, *a, **k):
        rng = np.random.default_rng(2)
        start = dt.date(2024, 1, 1)
        self._values = 4.0 + np.cumsum(rng.normal(0, 0.03, 600))
        self._dates = [start + dt.timedelta(days=i) for i in range(600)]

    async def fred_series(self, series_id, *, as_of=None):
        return [Point(d, float(v)) for d, v in zip(self._dates, self._values, strict=True)]

    async def yahoo_daily(self, symbol, *, years=5, as_of=None):
        return [
            Bar(d, float(v), float(v) * 1.01, float(v) * 0.99, float(v))
            for d, v in zip(self._dates, self._values, strict=True)
        ]

    async def close(self):
        pass


class TestReport:
    async def test_report_contents(self, monkeypatch):
        monkeypatch.setattr(qa, "MarketData", FakeMarketData)
        q = _question(
            "https://fred.stlouisfed.org/series/DGS10 and "
            "https://finance.yahoo.com/quote/NVDA/"
        )
        report = await qa.quant_assist_report(q)
        assert report is not None
        assert "FRED DGS10" in report
        assert "Yahoo NVDA close" in report
        assert "random-walk" in report
        assert "10%" in report and "90%" in report

    async def test_no_sources_returns_none(self, monkeypatch):
        monkeypatch.setattr(qa, "MarketData", FakeMarketData)
        assert await qa.quant_assist_report(_question("no links here")) is None

    async def test_fetch_failure_returns_none(self, monkeypatch):
        class BrokenMarketData(FakeMarketData):
            async def fred_series(self, series_id, *, as_of=None):
                raise RuntimeError("boom")

        monkeypatch.setattr(qa, "MarketData", BrokenMarketData)
        q = _question("https://fred.stlouisfed.org/series/DGS10")
        assert await qa.quant_assist_report(q) is None
