"""Unit tests for the quant series router, model core, and pipeline glue."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from common.models import ForecastQuestion, NumericScaling, QuestionType
from forecasting.quant import (
    _parse_period,
    _standardized_spread_windows,
    _to_cdf,
    match_quant_series,
    mixture_percentiles,
    quant_strategy_result,
)

YAHOO_BODY = "resolves according to finance.yahoo.com data"
FRED_BODY = "published on fred.stlouisfed.org"
CBOE_BODY = "CSV published by cboe.com"


class TestRouter:
    def test_matches_all_known_series(self):
        cases = [
            (
                "How much will Nasdaq-100 Futures total price returns exceed "
                "S&P 500 Futures in these biweekly periods of Q2 2026? (Jun 15 - Jun 26)",
                YAHOO_BODY, "nq_es", "spread",
            ),
            (
                "How much will Nvidia's stock price returns exceed Apple's in "
                "these biweekly periods of Q2 2026? (Jun 1 - Jun 12)",
                YAHOO_BODY, "nvda_aapl", "spread",
            ),
            (
                "What will be the ending value of the UST 10Y Yield for these "
                "biweekly periods of Q2 2026? (Jun 15 - Jun 26)",
                FRED_BODY, "ust10y", "level",
            ),
            (
                "What will be the maximum intraday value of the VIX over these "
                "biweekly periods in Q2 2026? (Jun 15 - Jun 26)",
                CBOE_BODY, "vix_max", "vix_max",
            ),
        ]
        for title, body, name, kind in cases:
            spec = match_quant_series(title, body)
            assert spec is not None, title
            assert spec.name == name
            assert spec.kind == kind
            assert spec.period_start == dt.date(2026, 6, 15) or spec.period_start == dt.date(
                2026, 6, 1
            )

    def test_rejects_missing_source_token(self):
        title = (
            "What will be the ending value of the UST 10Y Yield for these "
            "biweekly periods of Q2 2026? (Jun 15 - Jun 26)"
        )
        assert match_quant_series(title, "resolves by some other source") is None

    def test_rejects_unparseable_period(self):
        title = "What will be the ending value of the UST 10Y Yield in Q3 2026?"
        assert match_quant_series(title, FRED_BODY) is None

    def test_unknown_title_no_match(self):
        assert match_quant_series("Will X happen by 2027? (Jun 1 - Jun 12)", YAHOO_BODY) is None

    def test_period_year_wrap(self):
        period = _parse_period("Something of Q4 2026? (Dec 28 - Jan 8)")
        assert period == (dt.date(2026, 12, 28), dt.date(2027, 1, 8))


class TestModelCore:
    def test_mixture_percentiles_center_and_symmetry(self):
        rng = np.random.default_rng(7)
        z = rng.standard_normal(20_000)
        pct = mixture_percentiles(z, scale=2.0, center=10.0)
        assert pct[50.0] == pytest.approx(10.0, abs=0.1)
        # 10–90 band ≈ 2 × 1.2816 × scale, slightly widened by the mixture
        band = pct[90.0] - pct[10.0]
        assert 4.9 < band < 6.5
        values = [pct[p] for p in sorted(pct)]
        assert values == sorted(values)

    def test_mixture_fattens_tails_only(self):
        rng = np.random.default_rng(7)
        z = rng.standard_normal(20_000)
        mixed = mixture_percentiles(z, 1.0, 0.0)
        pure = mixture_percentiles(z, 1.0, 0.0, wide_weight=0.0)
        # p99 pushed out by the wide component; median barely moves
        assert mixed[99.0] > pure[99.0] * 1.2
        assert abs(mixed[50.0] - pure[50.0]) < 0.05

    def test_standardized_windows_unit_scale(self):
        rng = np.random.default_rng(11)
        n = 3000
        a = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        b = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        z, scale = _standardized_spread_windows(a, b, horizon=10)
        assert len(z) > 2500
        assert 0.8 < np.std(z) < 1.25  # standardization ≈ unit variance
        assert scale > 0


class TestPipelineGlue:
    SCALING = NumericScaling(
        range_min=-5.0,
        range_max=5.0,
        open_lower_bound=True,
        open_upper_bound=True,
        cdf_size=201,
    )

    def test_to_cdf_valid(self):
        rng = np.random.default_rng(3)
        pct = mixture_percentiles(rng.standard_normal(5000), scale=1.0, center=0.0)
        cdf = _to_cdf(pct, self.SCALING)
        assert len(cdf) == 201
        assert all(b - a >= 0 for a, b in zip(cdf, cdf[1:], strict=False))
        assert 0.0 <= cdf[0] < 0.3 and 0.7 < cdf[-1] <= 1.0

    async def test_quant_strategy_result(self, monkeypatch):
        import forecasting.quant as quant_mod
        from research.market_data import Bar

        rng = np.random.default_rng(5)
        start = dt.date(2018, 1, 1)

        def bars(seed_scale):
            closes = 100 * np.exp(np.cumsum(rng.normal(0, seed_scale, 2200)))
            out, d = [], start
            for c in closes:
                d = d + dt.timedelta(days=1)
                out.append(Bar(d, c, c * 1.01, c * 0.99, c))
            return out

        class FakeMarketData:
            def __init__(self, *a, **k):
                self._bars = {"NQ=F": bars(0.012), "ES=F": bars(0.009), "^VXN": bars(0.02)}

            async def yahoo_daily(self, symbol, *, years=15, as_of=None):
                return self._bars[symbol]

            async def close(self):
                pass

        monkeypatch.setattr(quant_mod, "MarketData", FakeMarketData)

        question = ForecastQuestion(
            title=(
                "How much will Nasdaq-100 Futures total price returns exceed "
                "S&P 500 Futures in these biweekly periods of Q3 2026? (Jul 13 - Jul 24)"
            ),
            body=YAHOO_BODY,
            question_type=QuestionType.NUMERIC,
            scaling=self.SCALING,
        )
        spec = match_quant_series(question.title, question.body)
        assert spec is not None
        result = await quant_strategy_result(question, spec)
        assert len(result.forecast) == 201
        assert result.sample_models == ["quant"]
        assert result.ctx["_quant"]["series"] == "nq_es"
        assert "Quant series nq_es" in result.comment


class TestSanityGate:
    """Widen-only gate on the Tier-1 quant path (plan §4)."""

    def _question(self):
        return ForecastQuestion(
            title=(
                "How much will Nasdaq-100 Futures total price returns exceed "
                "S&P 500 Futures in these biweekly periods of Q3 2026? (Jul 13 - Jul 24)"
            ),
            body=YAHOO_BODY,
            question_type=QuestionType.NUMERIC,
            scaling=TestPipelineGlue.SCALING,
        )

    def _fake_md(self, monkeypatch):
        import forecasting.quant as quant_mod
        from research.market_data import Bar

        rng = np.random.default_rng(5)
        start = dt.date(2018, 1, 1)

        def bars(scale):
            closes = 100 * np.exp(np.cumsum(rng.normal(0, scale, 2200)))
            return [
                Bar(start + dt.timedelta(days=i), c, c * 1.01, c * 0.99, c)
                for i, c in enumerate(closes)
            ]

        fixed_bars = {"NQ=F": bars(0.012), "ES=F": bars(0.009), "^VXN": bars(0.02)}

        class FakeMarketData:
            def __init__(self, *a, **k):
                self._bars = fixed_bars

            async def yahoo_daily(self, symbol, *, years=15, as_of=None):
                return self._bars[symbol]

            async def close(self):
                pass

        monkeypatch.setattr(quant_mod, "MarketData", FakeMarketData)

    def _gateway(self, action, factor=1.0):
        from forecasting.quant import GateVerdict

        class FakeGateway:
            async def extract_structured(self, request, output_type):
                return GateVerdict(action=action, widen_factor=factor, reason="test")

        return FakeGateway()

    @staticmethod
    def _band(result):
        pct = {float(k): v for k, v in result.ctx["_quant"]["percentiles"].items()}
        return pct[90.0] - pct[10.0]

    async def test_none_matches_ungated(self, monkeypatch):
        self._fake_md(monkeypatch)
        q = self._question()
        spec = match_quant_series(q.title, q.body)
        ungated = await quant_strategy_result(q, spec)
        gated = await quant_strategy_result(q, spec, gateway=self._gateway("none"))
        assert self._band(gated) == pytest.approx(self._band(ungated))

    async def test_widen_widens_and_clamps(self, monkeypatch):
        self._fake_md(monkeypatch)
        q = self._question()
        spec = match_quant_series(q.title, q.body)
        ungated = await quant_strategy_result(q, spec)
        widened = await quant_strategy_result(q, spec, gateway=self._gateway("widen", 1.5))
        clamped = await quant_strategy_result(q, spec, gateway=self._gateway("widen", 9.0))
        assert self._band(widened) == pytest.approx(self._band(ungated) * 1.5, rel=0.05)
        assert self._band(clamped) == pytest.approx(self._band(ungated) * 2.0, rel=0.05)
        assert "gate widened" in widened.comment

    async def test_flag_raises(self, monkeypatch):
        from forecasting.quant import QuantGateFlagged

        self._fake_md(monkeypatch)
        q = self._question()
        spec = match_quant_series(q.title, q.body)
        with pytest.raises(QuantGateFlagged):
            await quant_strategy_result(q, spec, gateway=self._gateway("flag"))

    async def test_gate_error_proceeds_ungated(self, monkeypatch):
        self._fake_md(monkeypatch)

        class BrokenGateway:
            async def extract_structured(self, request, output_type):
                raise RuntimeError("llm down")

        q = self._question()
        spec = match_quant_series(q.title, q.body)
        result = await quant_strategy_result(q, spec, gateway=BrokenGateway())
        assert len(result.forecast) == 201
