"""Tier-2 quant assist — inject resolution-source data into research context.

For any question whose body links a fetchable numeric series (FRED or
Yahoo Finance), fetch the actual series at forecast time and prepend a
compact stats block to the research summary: latest value, trailing
changes, EWMA vol, 1y range, and a naive random-walk 10/50/90 at the
question's horizon. The ensemble then anchors on real numbers and spends
its reasoning on regime judgment instead of guessing levels.
(docs/financial-series-plan.md §5.)

Never blocks a forecast: any failure returns None and the LLM path
proceeds without the block.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import urllib.parse
from typing import Any

import numpy as np

from research.market_data import MarketData

logger = logging.getLogger(__name__)

# 10%/90% of a standard normal — the naive random-walk band.
_Z80 = 1.2816

_FRED_RE = re.compile(r"fred\.stlouisfed\.org/(?:series|graph)/([A-Za-z0-9]+)")
_YAHOO_RE = re.compile(r"finance\.yahoo\.com/quote/([A-Za-z0-9%._^=-]+)")

MAX_SERIES = 3


def quant_modes(value: Any) -> frozenset[str]:
    """Parse the ``--fc quant=`` value: comma-separated flags
    (``series``, ``shadow``, ``assist``)."""
    if not value:
        return frozenset()
    return frozenset(part.strip() for part in str(value).split(",") if part.strip())


def detect_series(body: str) -> list[tuple[str, str]]:
    """[(source, series_id/symbol), …] linked from the question body."""
    found: list[tuple[str, str]] = []
    for series_id in _FRED_RE.findall(body):
        found.append(("fred", series_id))
    for raw in _YAHOO_RE.findall(body):
        symbol = urllib.parse.unquote(raw).rstrip("/").split("/")[0]
        found.append(("yahoo", symbol))
    deduped = list(dict.fromkeys(found))
    return deduped[:MAX_SERIES]


def _ewma_vol(diffs: np.ndarray, lam: float = 0.94) -> float:
    weights = lam ** np.arange(len(diffs) - 1, -1, -1)
    return float(np.sqrt(np.sum(weights * diffs**2) / np.sum(weights)))


def _horizon_bdays(resolve_time: str | None) -> int:
    if not resolve_time:
        return 30
    try:
        resolve_date = dt.datetime.fromisoformat(resolve_time.replace("Z", "+00:00")).date()
    except ValueError:
        return 30
    return int(np.clip(np.busday_count(dt.date.today(), resolve_date), 1, 250))


def _series_line(label: str, dates: list[dt.date], values: np.ndarray, horizon: int) -> str:
    latest = float(values[-1])
    changes = []
    for name, lag in (("1m", 21), ("3m", 63), ("1y", 252)):
        if len(values) > lag:
            changes.append(f"Δ{name} {latest - float(values[-1 - lag]):+.4g}")
    year = values[-252:] if len(values) > 252 else values
    vol = _ewma_vol(np.diff(values[-500:]) if len(values) > 1 else np.array([0.0]))
    band = _Z80 * vol * np.sqrt(horizon)
    return (
        f"- {label}: latest {latest:.4g} ({dates[-1].isoformat()}); "
        + (", ".join(changes) + "; " if changes else "")
        + f"1y range {float(year.min()):.4g}–{float(year.max()):.4g}; "
        f"EWMA daily vol {vol:.4g}; naive random-walk to resolution "
        f"(~{horizon} business days): "
        f"10% {latest - band:.4g} / 50% {latest:.4g} / 90% {latest + band:.4g}"
    )


async def quant_assist_report(question: Any) -> str | None:
    """Stats block for a question with detectable resolution-source series."""
    detected = detect_series(question.body)
    if not detected:
        return None
    horizon = _horizon_bdays(question.resolve_time)
    lines: list[str] = []
    md = MarketData()
    try:
        for source, key in detected:
            try:
                if source == "fred":
                    points = await md.fred_series(key)
                    dates = [p.date for p in points]
                    values = np.array([p.value for p in points])
                    label = f"FRED {key}"
                else:
                    bars = await md.yahoo_daily(key, years=5)
                    dates = [b.date for b in bars]
                    values = np.array([b.close for b in bars])
                    label = f"Yahoo {key} close"
                if len(values) < 30:
                    continue
                lines.append(_series_line(label, dates, values, horizon))
            except Exception:
                logger.warning("quant assist: fetch failed for %s %s", source, key, exc_info=True)
    finally:
        await md.close()
    if not lines:
        return None
    return (
        "[Resolution-source data, fetched "
        f"{dt.date.today().isoformat()} — anchor numeric estimates on these "
        "values; spend reasoning on regime changes and events, not on "
        "guessing current levels]\n" + "\n".join(lines)
    )
