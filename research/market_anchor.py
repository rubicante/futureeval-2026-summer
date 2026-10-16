"""Prediction-market anchoring — inject market prices into research context.

For political/geopolitical/public-figure questions, real-money prediction
markets are a well-calibrated crowd forecast the bot pack demonstrably
underuses. This module finds matching markets (Polymarket, Manifold),
verifies match congruence with one cheap LLM gate call, and prepends a
price block to the research summary. (docs/market-anchoring-plan.md.)

Never blocks a forecast: any failure returns None and the LLM path
proceeds without the block.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import statistics
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel, Field

from llm.gateway import Gateway, LLMRequest
from research.quant_assist import detect_series

logger = logging.getLogger(__name__)

_GAMMA_SEARCH_URL = "https://gamma-api.polymarket.com/public-search"
_CLOB_HISTORY_URL = "https://clob.polymarket.com/prices-history"
_MANIFOLD_SEARCH_URL = "https://api.manifold.markets/v0/search-markets"

# Gate prompt size / block size limits.
MAX_GATE_CANDIDATES = 8
MAX_BLOCK_ANCHORS = 4
# Manifold markets with fewer bettors than this are noise, not signal.
MANIFOLD_MIN_BETTORS = 10
# Blend/report price is a median over this trailing window, not last trade.
PRICE_WINDOW_HOURS = 24


def anchor_modes(value: Any) -> frozenset[str]:
    """Parse the ``--rc anchor=`` value: comma-separated flags (``markets``)."""
    if not value:
        return frozenset()
    return frozenset(part.strip() for part in str(value).split(",") if part.strip())


@dataclass(frozen=True)
class Candidate:
    """One market found by search, before gate verification."""

    source: str  # "polymarket" | "manifold"
    market_id: str
    question: str
    yes_price: float | None
    yes_token: str | None  # Polymarket CLOB token for price history
    volume: float | None
    n_bettors: int | None
    description: str


class AnchorMatch(BaseModel):
    """Gate verdict for one candidate market."""

    market_id: str = Field(description="The candidate's market_id, verbatim")
    congruence: str = Field(
        description="One of: exact (same event, same resolution), "
        "informative (related — different deadline, subset, or proxy), "
        "unrelated"
    )
    caveat: str = Field(
        default="",
        description="One short line naming any mismatch: deadline, year, "
        "different resolver, subset of outcomes. Empty only for exact.",
    )


class AnchorVerdicts(BaseModel):
    matches: list[AnchorMatch] = Field(default_factory=list)


_MATCH_PROMPT = """\
You match forecasting questions to prediction markets. Below is a question
and candidate markets found by keyword search. For each candidate, judge
whether betting prices on it are evidence about THIS question's outcome.

Be strict about resolution congruence: check the event, the deadline/date
(a different year or deadline is at best "informative", never "exact"),
and what exactly resolves YES. Most candidates are "unrelated" — keyword
overlap alone means nothing.

Question: {title}
Resolves: {resolve_time}
Resolution criteria: {criteria}

Candidates:
{candidates}

Return a verdict for every candidate by market_id.
"""


class MarketAnchorClient:
    """Thin async client over the public Polymarket and Manifold APIs."""

    def __init__(self, timeout: int = 15) -> None:
        self._client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def polymarket_search(self, query: str, *, include_closed: bool = False) -> list[Candidate]:
        response = await self._client.get(_GAMMA_SEARCH_URL, params={"q": query})
        response.raise_for_status()
        out: list[Candidate] = []
        for event in response.json().get("events") or []:
            if event.get("closed") and not include_closed:
                continue
            for market in event.get("markets") or []:
                if market.get("closed") and not include_closed:
                    continue
                prices = _json_list(market.get("outcomePrices"))
                tokens = _json_list(market.get("clobTokenIds"))
                out.append(
                    Candidate(
                        source="polymarket",
                        market_id=f"pm:{market.get('id')}",
                        question=market.get("question") or event.get("title") or "",
                        yes_price=_maybe_float(prices[0]) if prices else None,
                        yes_token=tokens[0] if tokens else None,
                        volume=_maybe_float(market.get("volume")),
                        n_bettors=None,
                        description=(market.get("description") or "")[:300],
                    )
                )
        return out

    async def manifold_search(self, query: str, *, include_closed: bool = False) -> list[Candidate]:
        response = await self._client.get(
            _MANIFOLD_SEARCH_URL, params={"term": query, "limit": 8}
        )
        response.raise_for_status()
        out: list[Candidate] = []
        for market in response.json():
            if market.get("isResolved") and not include_closed:
                continue
            if market.get("outcomeType") != "BINARY":
                continue
            if (market.get("uniqueBettorCount") or 0) < MANIFOLD_MIN_BETTORS:
                continue
            out.append(
                Candidate(
                    source="manifold",
                    market_id=f"mf:{market.get('id')}",
                    question=market.get("question") or "",
                    yes_price=_maybe_float(market.get("probability")),
                    yes_token=None,
                    volume=None,
                    n_bettors=market.get("uniqueBettorCount"),
                    description=(market.get("textDescription") or "")[:300],
                )
            )
        return out

    async def polymarket_price(
        self, yes_token: str, *, as_of: dt.datetime | None = None
    ) -> float | None:
        """Median YES price over the 24 h ending at *as_of* (default: now).

        A trailing median resists spikes and thin-book manipulation; with
        ``as_of`` it reconstructs the anchor a live run would have seen.
        """
        params: dict[str, Any] = (
            {"market": yes_token, "interval": "1d", "fidelity": "60"}
            if as_of is None
            else {"market": yes_token, "interval": "max", "fidelity": "720"}
        )
        response = await self._client.get(_CLOB_HISTORY_URL, params=params)
        response.raise_for_status()
        history = response.json().get("history") or []
        end = (as_of or dt.datetime.now(dt.UTC)).timestamp()
        for window_h in (PRICE_WINDOW_HOURS, 72):
            window = [p["p"] for p in history if end - window_h * 3600 <= p["t"] <= end]
            if window:
                return float(statistics.median(window))
        return None


# ── question → search terms ──────────────────────────────────────────

_STOPWORDS = frozenset(["will", "the", "a", "an", "of", "in", "on", "at", "by", "for", "to", "be", "is", "are", "was", "were", "do", "does", "what", "which", "who", "how", "many", "much", "before", "after", "between", "during", "any", "these", "this", "that", "its", "it"])
_PAREN_RE = re.compile(r"\([^)]*\)")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'&-]+|\d{4}")


def search_terms(title: str) -> list[str]:
    """1–2 keyword queries from a question title (heuristic fallback).

    The primary path is :func:`llm_search_terms`; this fires only when
    that call fails. Comparators and resolution details (dates, "above",
    percentages) hurt strict market search, so keep topic words only.
    """
    cleaned = _PAREN_RE.sub(" ", title)
    tokens = [t for t in _TOKEN_RE.findall(cleaned) if t.lower() not in _STOPWORDS]
    if not tokens:
        return []
    queries = [" ".join(tokens[:5])]
    proper = [t for t in tokens if t[0].isupper()]
    if 2 <= len(proper) < len(tokens[:5]):
        queries.append(" ".join(proper[:5]))
    return list(dict.fromkeys(queries))


class SearchTerms(BaseModel):
    """LLM-chosen market search queries for a question."""

    queries: list[str] = Field(
        description="2-3 short keyword queries (2-5 words each) a prediction "
        "market about this question would use in its title. Topic words only: "
        "no dates, no numbers, no comparators like above/below/exceed."
    )


_TERMS_PROMPT = """\
Prediction markets (Polymarket, Manifold) might have a market about the
question below. Give 2-3 short keyword search queries to find it. Use the
words a market title would use: the person/event/place, the measurable thing.
Never include dates, years, numbers, or comparison words.

Question: {title}

Example: for "Will Donald Trump's approval rating be above 40% on July 10,
2026?" good queries are "Trump approval rating" and "presidential approval".
"""


async def llm_search_terms(gateway: Gateway, title: str) -> list[str]:
    """2–3 search queries via one decompose-tier call; heuristic on failure."""
    try:
        result = await gateway.extract_structured(
            LLMRequest(prompt=_TERMS_PROMPT.format(title=title), purpose="decompose"),
            SearchTerms,
        )
        queries = [q.strip() for q in result.queries if q.strip()][:3]
        if queries:
            return queries
    except Exception:
        logger.warning("anchor search-term LLM failed; using heuristic", exc_info=True)
    return search_terms(title)


def _overlap(title_tokens: frozenset[str], market_question: str) -> int:
    """How many meaningful question-title tokens appear in the market title."""
    market_tokens = {
        t.lower() for t in _TOKEN_RE.findall(market_question)
    } - _STOPWORDS
    return len(title_tokens & market_tokens)


def _title_tokens(title: str) -> frozenset[str]:
    return frozenset(
        t.lower() for t in _TOKEN_RE.findall(_PAREN_RE.sub(" ", title))
    ) - _STOPWORDS


# ── gate ─────────────────────────────────────────────────────────────


async def _verify_matches(
    gateway: Gateway, question: Any, candidates: list[Candidate]
) -> dict[str, AnchorMatch]:
    lines = []
    for c in candidates:
        extras = []
        if c.volume:
            extras.append(f"volume ${c.volume:,.0f}")
        if c.n_bettors:
            extras.append(f"{c.n_bettors} bettors")
        extra = f" ({', '.join(extras)})" if extras else ""
        lines.append(f"- market_id={c.market_id} [{c.source}]{extra}: {c.question}\n"
                     f"  {c.description[:200]}")
    # body = description + resolution criteria + fine print, in order; the
    # criteria live toward the end, so excerpt the tail.
    body = getattr(question, "body", "") or ""
    prompt = _MATCH_PROMPT.format(
        title=question.title,
        resolve_time=(getattr(question, "resolve_time", None) or "unknown")[:10],
        criteria=body[-700:],
        candidates="\n".join(lines),
    )
    verdicts = await gateway.extract_structured(
        LLMRequest(prompt=prompt, purpose="decompose"), AnchorVerdicts
    )
    return {m.market_id: m for m in verdicts.matches}


# ── report ───────────────────────────────────────────────────────────


async def anchor_report(
    question: Any, gateway: Gateway
) -> tuple[str, list[dict[str, Any]]] | None:
    """(context block, capture metadata) for a question with matched markets.

    Returns None when no market matches — or on any error; anchoring must
    never block a forecast.
    """
    try:
        return await _anchor_report(question, gateway)
    except Exception:
        logger.warning("market anchor failed for %s", question.title[:60], exc_info=True)
        return None


async def _anchor_report(
    question: Any, gateway: Gateway
) -> tuple[str, list[dict[str, Any]]] | None:
    # Financial-series questions anchor on resolution-source data instead.
    if detect_series(getattr(question, "body", "") or ""):
        return None
    queries = await llm_search_terms(gateway, question.title)
    if not queries:
        return None

    client = MarketAnchorClient()
    try:
        candidates: list[Candidate] = []
        for query in queries:
            for search in (client.polymarket_search, client.manifold_search):
                try:
                    candidates.extend(await search(query))
                except Exception:
                    logger.warning("anchor search failed (%s)", query, exc_info=True)
        candidates = _dedup(candidates)
        # Gate slots are scarce: rank by topical overlap with the question
        # title first (volume-sorting lets popular-but-unrelated markets
        # crowd out the right one), then real money over play money.
        tokens = _title_tokens(question.title)
        candidates.sort(
            key=lambda c: (
                -_overlap(tokens, c.question),
                c.source != "polymarket",
                -(c.volume or 0),
            )
        )
        candidates = candidates[:MAX_GATE_CANDIDATES]
        if not candidates:
            return None

        verdicts = await _verify_matches(gateway, question, candidates)
        kept = [
            (c, verdicts[c.market_id])
            for c in candidates
            if c.market_id in verdicts
            and verdicts[c.market_id].congruence in ("exact", "informative")
        ]
        kept = kept[:MAX_BLOCK_ANCHORS]
        if not kept:
            return None

        lines: list[str] = []
        meta: list[dict[str, Any]] = []
        for candidate, match in kept:
            price = candidate.yes_price
            if candidate.source == "polymarket" and candidate.yes_token:
                fresh = await client.polymarket_price(candidate.yes_token)
                price = fresh if fresh is not None else price
            if price is None:
                continue
            detail = f"volume ${candidate.volume:,.0f}" if candidate.volume else ""
            if candidate.n_bettors:
                detail = f"{candidate.n_bettors} bettors (play-money)"
            caveat = f"; caveat: {match.caveat}" if match.caveat else ""
            lines.append(
                f"- [{candidate.source}] \"{candidate.question}\" "
                f"(congruence: {match.congruence}{caveat})\n"
                f"  YES {price:.2f}" + (f"   {detail}" if detail else "")
            )
            meta.append(
                {
                    "source": candidate.source,
                    "market_id": candidate.market_id,
                    "question": candidate.question,
                    "congruence": match.congruence,
                    "caveat": match.caveat,
                    "price": price,
                    "volume": candidate.volume,
                }
            )
        if not lines:
            return None
        block = (
            "[Prediction-market anchors, fetched "
            f"{dt.datetime.now(dt.UTC).strftime('%Y-%m-%dT%H:%MZ')} — real-money "
            "prices at meaningful liquidity are usually better calibrated than "
            "model priors. Anchor on them; deviate only with concrete evidence "
            "the market has not priced. Mind each caveat — an 'informative' "
            "market bounds or hints at this question, it does not resolve it]\n"
            + "\n".join(lines)
        )
        return block, meta
    finally:
        await client.close()


# ── helpers ──────────────────────────────────────────────────────────


def _dedup(candidates: list[Candidate]) -> list[Candidate]:
    return list({c.market_id: c for c in candidates}.values())


def _json_list(value: Any) -> list:
    """Gamma encodes list fields as JSON strings ('["Yes","No"]')."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
