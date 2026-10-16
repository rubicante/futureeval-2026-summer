"""Quantitative models for recurring financial-series questions.

Design (docs/financial-series-plan.md): replicate the resolution arithmetic
exactly on the resolution-source feed, then model the residual randomness
with empirical quantiles of vol-standardized historical windows. No
parametric tails (fat tails come from the data itself), no directional
drift, and every forecast is blended with a wide background distribution
in CDF space (the "humility mixture").

The router is a strict allowlist: a question forecasts through this path
only when its title matches a known series pattern, its body names the
expected resolution source, and its period dates parse. Anything else
falls through to the LLM ensemble untouched.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from common.models import build_question_context
from forecasting.distributions import generate_continuous_cdf
from forecasting.strategies import StrategyResult
from llm.gateway import Gateway, LLMRequest
from research.market_data import MarketData

logger = logging.getLogger(__name__)

__all__ = [
    "QuantGateFlagged",
    "QuantSpec",
    "match_quant_series",
    "quant_percentiles",
    "quant_strategy_result",
    "quant_shadow_info",
]

PERCENTILE_LEVELS = (1, 2.5, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 97.5, 99)

# Humility mixture (plan §3d): CDF = (1-w)·model + w·(model widened 3×).
MIX_WIDE_WEIGHT = 0.10
MIX_WIDE_FACTOR = 3.0

EWMA_LAMBDA = 0.94
# The reference close P0 is not fixed yet at forecast time (question closes
# hours before the market does) — count a fraction of a day of extra variance.
P0_INTRADAY_FRACTION = 0.3

# Implied-vol regime rescale, dampened and clamped (plan §3a.3).
IV_CLAMP = (0.75, 1.5)


@dataclass(frozen=True)
class QuantSpec:
    kind: Literal["spread", "level", "vix_max"]
    name: str
    period_start: dt.date
    period_end: dt.date
    # spread: (yahoo_a, yahoo_b); level: fred series id; vix_max: unused
    symbols: tuple[str, ...]
    iv_symbol: str | None = None


# (regex on title, kind, name, symbols, iv symbol, required source token in body)
_SERIES_PATTERNS: list[tuple[str, str, str, tuple[str, ...], str | None, str]] = [
    (
        r"Nasdaq-100 Futures total price returns exceed S&P 500 Futures",
        "spread", "nq_es", ("NQ=F", "ES=F"), "^VXN", "finance.yahoo.com",
    ),
    (
        r"Gold Futures total price returns exceed S&P 500 Futures",
        "spread", "gc_es", ("GC=F", "ES=F"), "^GVZ", "finance.yahoo.com",
    ),
    (
        r"Crude Oil Futures total price returns exceed S&P 500 Futures",
        "spread", "cl_es", ("CL=F", "ES=F"), "^OVX", "finance.yahoo.com",
    ),
    (
        r"Nvidia's stock price returns exceed Apple's",
        "spread", "nvda_aapl", ("NVDA", "AAPL"), None, "finance.yahoo.com",
    ),
    (
        r"Nvidia's stock price returns exceed Microsoft's",
        "spread", "nvda_msft", ("NVDA", "MSFT"), None, "finance.yahoo.com",
    ),
    (
        r"ending value of the UST 10Y Yield",
        "level", "ust10y", ("DGS10",), None, "fred.stlouisfed.org",
    ),
    (
        r"ending value of the ICE BofA US High Yield Option-Adjusted Spread",
        "level", "hy_oas", ("BAMLH0A0HYM2",), None, "fred.stlouisfed.org",
    ),
    (
        r"maximum intraday value of the VIX",
        "vix_max", "vix_max", (), None, "cboe.com",
    ),
]

_PERIOD_RE = re.compile(
    r"\((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* (\d{1,2})\s*[-–]\s*"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* (\d{1,2})\)"
)
_YEAR_RE = re.compile(r"\b(20\d{2})\b")

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
)}


def match_quant_series(title: str, body: str) -> QuantSpec | None:
    """Strict allowlist match; returns None unless every check passes."""
    for pattern, kind, name, symbols, iv_symbol, source_token in _SERIES_PATTERNS:
        if not re.search(pattern, title):
            continue
        if source_token not in body:
            logger.warning(
                "quant router: %r matched series %s but body lacks source token %r "
                "— resolution wording may have changed, falling back to LLM path",
                title[:60], name, source_token,
            )
            return None
        period = _parse_period(title)
        if period is None:
            logger.warning("quant router: %r matched %s but period parse failed", title[:60], name)
            return None
        start, end = period
        return QuantSpec(
            kind=kind,  # type: ignore[arg-type]
            name=name,
            period_start=start,
            period_end=end,
            symbols=symbols,
            iv_symbol=iv_symbol,
        )
    return None


def _parse_period(title: str) -> tuple[dt.date, dt.date] | None:
    match = _PERIOD_RE.search(title)
    year_match = _YEAR_RE.search(title)
    if not match or not year_match:
        return None
    year = int(year_match.group(1))
    start = dt.date(year, _MONTHS[match.group(1)], int(match.group(2)))
    end_year = year + 1 if _MONTHS[match.group(3)] < _MONTHS[match.group(1)] else year
    end = dt.date(end_year, _MONTHS[match.group(3)], int(match.group(4)))
    if end <= start:
        return None
    return start, end


# ── model core ───────────────────────────────────────────────────────


def ewma_vol(diffs: np.ndarray, lam: float = EWMA_LAMBDA) -> float:
    """EWMA standard deviation of a daily-difference series."""
    weights = lam ** np.arange(len(diffs) - 1, -1, -1)
    return float(np.sqrt(np.sum(weights * diffs**2) / np.sum(weights)))


def _ewma_vol_series(diffs: np.ndarray, lam: float = EWMA_LAMBDA) -> np.ndarray:
    """EWMA vol at each index, using only data up to that index."""
    out = np.empty(len(diffs))
    var = float(np.mean(diffs[: min(20, len(diffs))] ** 2))  # warm start
    for i, d in enumerate(diffs):
        var = lam * var + (1 - lam) * d * d
        out[i] = np.sqrt(var)
    return out


def mixture_percentiles(
    z_samples: np.ndarray,
    scale: float,
    center: float,
    levels: tuple[float, ...] = PERCENTILE_LEVELS,
    wide_weight: float = MIX_WIDE_WEIGHT,
    wide_factor: float = MIX_WIDE_FACTOR,
) -> dict[float, float]:
    """Percentiles of ``center + scale·Z`` blended with a widened copy.

    The blend is a true CDF mixture: F = (1-w)·F(z) + w·F(z·wide_factor),
    evaluated on a value grid and inverted at *levels*.
    """
    z = np.sort(z_samples)
    n = len(z)
    ecdf_p = (np.arange(1, n + 1) - 0.5) / n
    grid = np.linspace(z[0] * wide_factor * 1.2, z[-1] * wide_factor * 1.2, 4001)
    f_model = np.interp(grid, z, ecdf_p, left=0.0, right=1.0)
    f_wide = np.interp(grid, z * wide_factor, ecdf_p, left=0.0, right=1.0)
    f_mix = (1 - wide_weight) * f_model + wide_weight * f_wide
    zq = np.interp(np.array(levels) / 100.0, f_mix, grid)
    return {float(p): float(center + scale * q) for p, q in zip(levels, zq, strict=True)}


def _standardized_spread_windows(
    closes_a: np.ndarray, closes_b: np.ndarray, horizon: int
) -> tuple[np.ndarray, float]:
    """(z-samples of horizon-day spread returns, current scaled vol).

    Windows are the exact resolved quantity — 100·(A₁/A₀−1) − 100·(B₁/B₀−1)
    over *horizon* trading days — standardized by the EWMA vol of daily
    spread returns at each window's start (vol clustering) times √horizon.
    """
    ret_a = 100 * np.diff(closes_a) / closes_a[:-1]
    ret_b = 100 * np.diff(closes_b) / closes_b[:-1]
    daily_spread = ret_a - ret_b
    vol_series = _ewma_vol_series(daily_spread)

    window = (
        100 * (closes_a[horizon:] / closes_a[:-horizon] - 1)
        - 100 * (closes_b[horizon:] / closes_b[:-horizon] - 1)
    )
    # window i spans closes[i]..closes[i+horizon]; vol known at its start
    start_vol = vol_series[: len(window)]
    valid = start_vol > 1e-9
    z = window[valid] / (start_vol[valid] * np.sqrt(horizon))

    current_vol = float(vol_series[-1]) * np.sqrt(horizon + P0_INTRADAY_FRACTION)
    return z, current_vol


async def quant_percentiles(
    spec: QuantSpec, md: MarketData, *, as_of: dt.date | None = None, scale_mult: float = 1.0
) -> tuple[dict[float, float], str]:
    """Percentile forecast + human-readable rationale for a matched series.

    ``scale_mult`` widens the distribution (sanity gate, §4) — it may only
    ever be ≥ 1.
    """
    scale_mult = max(1.0, scale_mult)
    if spec.kind == "spread":
        return await _spread_forecast(spec, md, as_of, scale_mult)
    if spec.kind == "level":
        return await _level_forecast(spec, md, as_of, scale_mult)
    return await _vix_max_forecast(spec, md, as_of, scale_mult)


async def _spread_forecast(
    spec: QuantSpec, md: MarketData, as_of: dt.date | None, scale_mult: float = 1.0
) -> tuple[dict[float, float], str]:
    bars_a = await md.yahoo_daily(spec.symbols[0], as_of=as_of)
    bars_b = await md.yahoo_daily(spec.symbols[1], as_of=as_of)
    by_date_b = {b.date: b.close for b in bars_b}
    dates = [b.date for b in bars_a if b.date in by_date_b]
    closes_a = np.array([b.close for b in bars_a if b.date in by_date_b])
    closes_b = np.array([by_date_b[d] for d in dates])

    horizon = int(np.busday_count(spec.period_start, spec.period_end)) + 1
    z, scale = _standardized_spread_windows(closes_a, closes_b, horizon)

    iv_note = ""
    if spec.iv_symbol is not None:
        try:
            iv_bars = await md.yahoo_daily(spec.iv_symbol, years=2, as_of=as_of)
            iv_now = iv_bars[-1].close
            iv_mean = float(np.mean([b.close for b in iv_bars[-252:]]))
            factor = float(np.clip(np.sqrt(iv_now / iv_mean), *IV_CLAMP))
            scale *= factor
            iv_note = f", IV rescale ×{factor:.2f} ({spec.iv_symbol} {iv_now:.1f} vs 1y mean {iv_mean:.1f})"
        except Exception:  # IV overlay is an enhancement, never a blocker
            logger.warning("quant: IV overlay failed for %s", spec.iv_symbol, exc_info=True)

    # Zero-drift center (plan §3a.2): remove the empirical median so the
    # historical shape contributes tails/skew but no directional bet.
    z = z - np.median(z)
    percentiles = mixture_percentiles(z, scale * scale_mult, center=0.0)
    rationale = (
        f"Quant series {spec.name}: {len(z)} overlapping {horizon}d windows, "
        f"EWMA spread vol → scale {scale:.2f}pp{iv_note}; zero-drift center, "
        f"{int(MIX_WIDE_WEIGHT * 100)}% humility mixture. "
        f"10–90%: {percentiles[10.0]:.2f} .. {percentiles[90.0]:.2f}"
    )
    return percentiles, rationale


async def _level_forecast(
    spec: QuantSpec, md: MarketData, as_of: dt.date | None, scale_mult: float = 1.0
) -> tuple[dict[float, float], str]:
    points = await md.fred_series(spec.symbols[0], as_of=as_of)
    values = np.array([p.value for p in points])
    last_date, last_value = points[-1].date, float(values[-1])

    horizon = max(int(np.busday_count(last_date, spec.period_end)), 1)
    diffs = np.diff(values)
    vol_series = _ewma_vol_series(diffs)
    window = values[horizon:] - values[:-horizon]
    start_vol = vol_series[: len(window)]
    valid = start_vol > 1e-9
    z = window[valid] / (start_vol[valid] * np.sqrt(horizon))
    z = z - np.median(z)

    scale = float(vol_series[-1]) * np.sqrt(horizon) * scale_mult
    percentiles = mixture_percentiles(z, scale, center=last_value)
    rationale = (
        f"Quant series {spec.name}: random walk from {spec.symbols[0]}={last_value} "
        f"({last_date}), {horizon} business days to period end, EWMA daily vol "
        f"{vol_series[-1]:.4f} → scale {scale:.3f}; 10–90%: "
        f"{percentiles[10.0]:.2f} .. {percentiles[90.0]:.2f}"
    )
    return percentiles, rationale


async def _vix_max_forecast(
    spec: QuantSpec, md: MarketData, as_of: dt.date | None, scale_mult: float = 1.0
) -> tuple[dict[float, float], str]:
    bars = await md.cboe_vix_history(as_of=as_of)
    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    current = float(closes[-1])

    window_days = int(np.busday_count(spec.period_start, spec.period_end)) + 1
    # Rolling max of HIGH over each window; ratio to the close before it.
    n = len(bars) - window_days
    log_ratios = np.array(
        [np.log(highs[i + 1 : i + 1 + window_days].max() / closes[i]) for i in range(n)]
    )
    starts = closes[:n]

    # Condition on the starting level: max/start ratios are regime-dependent
    # (plan §3c). Take the nearest-level half of history, min 300 windows.
    order = np.argsort(np.abs(np.log(starts / current)))
    keep = order[: max(300, n // 2)]
    z = np.sort(log_ratios[keep])
    z = z - np.median(z)
    median_ratio = float(np.exp(np.median(log_ratios[keep])))

    log_percentiles = mixture_percentiles(
        z, scale_mult, center=np.log(current * median_ratio)
    )
    percentiles = {p: float(np.exp(v)) for p, v in log_percentiles.items()}
    rationale = (
        f"Quant series vix_max: VIX close {current:.2f}, {window_days}d window, "
        f"{len(keep)} level-conditioned historical windows (median max/start "
        f"{median_ratio:.2f}); 10–90%: {percentiles[10.0]:.1f} .. {percentiles[90.0]:.1f}"
    )
    return percentiles, rationale


# ── pipeline integration ─────────────────────────────────────────────


def _to_cdf(percentiles: dict[float, float], scaling) -> list[float]:
    # Extreme mixture tails can land far outside the question range, which
    # NumericDistribution rejects (>2× range beyond a bound). Clip to one
    # range-width beyond the bounds — the CDF mass past an open bound is
    # identical either way.
    width = scaling.range_max - scaling.range_min
    lo, hi = scaling.range_min - 0.9 * width, scaling.range_max + 0.9 * width
    percentiles = {p: min(max(v, lo), hi) for p, v in percentiles.items()}

    # Values must be strictly increasing for NumericDistribution; nudge any
    # numerically-flat neighbors (clipping and scale ≈ 0 both cause ties).
    items = sorted(percentiles.items())
    values = [v for _, v in items]
    span = max(abs(values[-1] - values[0]), 1e-6)
    for i in range(1, len(values)):
        if values[i] <= values[i - 1]:
            values[i] = values[i - 1] + span * 1e-6
    return generate_continuous_cdf(
        {p: v for (p, _), v in zip(items, values, strict=True)},
        scaling.open_upper_bound,
        scaling.open_lower_bound,
        scaling.range_max,
        scaling.range_min,
        scaling.zero_point,
        scaling.cdf_size,
    )


class QuantGateFlagged(Exception):
    """Sanity gate flagged the question — caller falls back to the LLM path."""


class GateVerdict(BaseModel):
    """Structured verdict of the widen-only sanity gate (plan §4)."""

    action: str = Field(description="One of: none, widen, flag")
    widen_factor: float = Field(
        default=1.0, description="Only for action=widen: multiply the distribution width, 1.0-2.0"
    )
    reason: str = Field(default="", description="One short sentence")


_GATE_PROMPT = """\
You are a risk check for a statistical forecaster. It predicts the question
below purely from historical volatility, assuming no unusual scheduled or
breaking events inside the period. Your ONLY job: decide whether history is
representative for this specific window.

Question: {title}
Period: {start} to {end}. Today: {today}.
Statistical forecast (10%/50%/90%): {p10:.4g} / {p50:.4g} / {p90:.4g}

Consider: scheduled events inside the period (earnings of the named
companies, FOMC/CPI dates, elections, OPEC meetings) and any ongoing regime
break (war, crisis, halted market). Then answer with exactly one action:
- "none"  — history is representative (the common case)
- "widen" — a scheduled event adds variance; give widen_factor 1.0-2.0
- "flag"  — the statistical premise itself is broken (rare)

You may NOT shift the center or narrow the distribution."""


async def _sanity_gate(
    gateway: Gateway, spec: QuantSpec, percentiles: dict[float, float], title: str
) -> GateVerdict:
    prompt = _GATE_PROMPT.format(
        title=title,
        start=spec.period_start.isoformat(),
        end=spec.period_end.isoformat(),
        today=dt.date.today().isoformat(),
        p10=percentiles[10.0],
        p50=percentiles[50.0],
        p90=percentiles[90.0],
    )
    return await gateway.extract_structured(
        LLMRequest(prompt=prompt, purpose="decompose"), GateVerdict
    )


async def quant_strategy_result(question, spec: QuantSpec, gateway: Gateway | None = None):
    """Full StrategyResult for a Tier-1 matched numeric question.

    Deterministic, research-free path: resolution-source data → percentiles
    → constrained CDF. Returns the same shape as an LLM strategy so
    captures, submission, and replay work unchanged. When a *gateway* is
    supplied, a cheap widen-only LLM sanity gate runs on the result: it may
    widen the distribution (≤2×) or flag it (→ ``QuantGateFlagged``), never
    shift or sharpen it.
    """
    if question.scaling is None:
        raise ValueError("quant path requires a numeric question with scaling")
    gate_note = ""
    md = MarketData()
    try:
        percentiles, rationale = await quant_percentiles(spec, md)
        if gateway is not None:
            try:
                verdict = await _sanity_gate(gateway, spec, percentiles, question.title)
            except Exception:
                logger.warning("quant gate errored for %s; proceeding ungated", spec.name,
                               exc_info=True)
                verdict = GateVerdict(action="none", reason="gate error")
            if verdict.action == "flag":
                raise QuantGateFlagged(f"{spec.name}: {verdict.reason}")
            if verdict.action == "widen":
                factor = float(min(max(verdict.widen_factor, 1.0), 2.0))
                percentiles, rationale = await quant_percentiles(spec, md, scale_mult=factor)
                gate_note = f" [gate widened ×{factor:.2f}: {verdict.reason}]"
    finally:
        await md.close()
    rationale += gate_note
    cdf = _to_cdf(percentiles, question.scaling)

    ctx = build_question_context(question)
    ctx["summary_report"] = rationale
    ctx["_quant"] = {
        "series": spec.name,
        "kind": spec.kind,
        "period": [spec.period_start.isoformat(), spec.period_end.isoformat()],
        "percentiles": {str(k): v for k, v in percentiles.items()},
        "gate": gate_note or "none",
    }
    return StrategyResult(
        forecast=cdf,
        comment=rationale,
        ctx=ctx,
        raw_results=[(cdf, rationale)],
        sample_models=["quant"],
        health=None,
    )


async def quant_shadow_info(question, spec: QuantSpec) -> dict | None:
    """Shadow-mode payload: percentiles only, stored in the capture ctx."""
    md = MarketData()
    try:
        percentiles, rationale = await quant_percentiles(spec, md)
    except Exception:
        logger.warning("quant shadow failed for %s", spec.name, exc_info=True)
        return None
    finally:
        await md.close()
    return {
        "series": spec.name,
        "kind": spec.kind,
        "period": [spec.period_start.isoformat(), spec.period_end.isoformat()],
        "percentiles": {str(k): v for k, v in percentiles.items()},
        "rationale": rationale,
    }
