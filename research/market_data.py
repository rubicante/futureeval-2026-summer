"""Market data layer for quantitative questions.

Fetches the exact public series Metaculus resolution criteria name:
Yahoo Finance daily bars (futures/stocks/vol indices), FRED daily series,
and the CBOE VIX history CSV. See docs/financial-series-plan.md §1.

Design:
- One fetch per (source, symbol) per day, cached as JSON under
  ``runs/market_cache/`` — every question in a run reuses the same pull.
- ``as_of`` truncation gives point-in-time views for backtests. These
  feeds are historical closes, so truncation reconstructs what was
  knowable on a date (FRED revisions on DGS10/HY OAS are negligible).
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

logger = logging.getLogger(__name__)

CACHE_DIR = Path("runs/market_cache")

# Yahoo's public chart API rejects default httpx UA.
_YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
_YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_CBOE_VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"


@dataclass(frozen=True)
class Bar:
    date: dt.date
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class Point:
    date: dt.date
    value: float


class MarketData:
    """Cached access to the resolution-source feeds."""

    def __init__(self, cache_dir: Path = CACHE_DIR, timeout: int = 30) -> None:
        self._cache_dir = cache_dir
        self._client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)

    async def close(self) -> None:
        await self._client.aclose()

    # ── public API ───────────────────────────────────────────────────

    async def yahoo_daily(
        self, symbol: str, *, years: int = 15, as_of: dt.date | None = None
    ) -> list[Bar]:
        """Daily OHLC bars from Yahoo Finance (e.g. ``NQ=F``, ``NVDA``, ``^VXN``)."""
        rows = await self._cached(f"yahoo_{symbol}", self._fetch_yahoo, symbol, years)
        bars = [Bar(dt.date.fromisoformat(r[0]), r[1], r[2], r[3], r[4]) for r in rows]
        return _truncate(bars, as_of)

    async def fred_series(self, series_id: str, *, as_of: dt.date | None = None) -> list[Point]:
        """Daily values from FRED (e.g. ``DGS10``, ``BAMLH0A0HYM2``). No API key."""
        rows = await self._cached(f"fred_{series_id}", self._fetch_fred, series_id)
        points = [Point(dt.date.fromisoformat(r[0]), r[1]) for r in rows]
        return _truncate(points, as_of)

    async def cboe_vix_history(self, *, as_of: dt.date | None = None) -> list[Bar]:
        """Full VIX OHLC history (1990→) from the CBOE CSV that resolves VIX questions."""
        rows = await self._cached("cboe_VIX", self._fetch_cboe_vix)
        bars = [Bar(dt.date.fromisoformat(r[0]), r[1], r[2], r[3], r[4]) for r in rows]
        return _truncate(bars, as_of)

    # ── fetchers (return JSON-serializable rows) ─────────────────────

    async def _fetch_yahoo(self, symbol: str, years: int) -> list[list]:
        response = await self._client.get(
            _YAHOO_URL.format(symbol=symbol),
            params={"range": f"{years}y", "interval": "1d"},
            headers=_YAHOO_HEADERS,
        )
        response.raise_for_status()
        result = response.json()["chart"]["result"][0]
        timestamps = result.get("timestamp") or []
        quote = result["indicators"]["quote"][0]
        tz = result["meta"].get("exchangeTimezoneName") or "America/New_York"
        rows: list[list] = []
        for i, ts in enumerate(timestamps):
            o, h, lo, c = (quote[k][i] for k in ("open", "high", "low", "close"))
            if None in (o, h, lo, c):
                continue  # holiday/partial rows
            date = dt.datetime.fromtimestamp(ts, dt.UTC).astimezone(
                _zoneinfo(tz)
            ).date()
            rows.append([date.isoformat(), o, h, lo, c])
        return rows

    async def _fetch_fred(self, series_id: str) -> list[list]:
        # Without cosd, fredgraph.csv defaults some series to a ~3y window.
        response = await self._client.get(
            _FRED_URL, params={"id": series_id, "cosd": "1990-01-01"}
        )
        response.raise_for_status()
        rows: list[list] = []
        for record in csv.DictReader(io.StringIO(response.text)):
            # header is DATE,<series_id> (older exports use observation_date)
            date = record.get("DATE") or record.get("observation_date")
            raw = record.get(series_id)
            if not date or raw is None or raw in ("", "."):
                continue
            rows.append([date, float(raw)])
        return rows

    async def _fetch_cboe_vix(self) -> list[list]:
        response = await self._client.get(_CBOE_VIX_URL)
        response.raise_for_status()
        rows: list[list] = []
        for record in csv.DictReader(io.StringIO(response.text)):
            date = dt.datetime.strptime(record["DATE"], "%m/%d/%Y").date()
            rows.append(
                [
                    date.isoformat(),
                    float(record["OPEN"]),
                    float(record["HIGH"]),
                    float(record["LOW"]),
                    float(record["CLOSE"]),
                ]
            )
        return rows

    # ── cache ────────────────────────────────────────────────────────

    async def _cached(self, key: str, fetch, *args) -> list[list]:
        safe_key = key.replace("=", "_").replace("^", "idx_").replace("/", "_")
        path = self._cache_dir / f"{safe_key}_{dt.date.today().isoformat()}.json"
        if path.exists():
            return json.loads(path.read_text())
        rows = await fetch(*args)
        if not rows:
            raise RuntimeError(f"market data fetch for {key!r} returned no rows")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows))
        logger.info("market_data: fetched %s (%d rows)", key, len(rows))
        return rows


def _truncate(rows: list, as_of: dt.date | None) -> list:
    if as_of is None:
        return rows
    return [r for r in rows if r.date <= as_of]


def _zoneinfo(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except KeyError:
        return ZoneInfo("America/New_York")
