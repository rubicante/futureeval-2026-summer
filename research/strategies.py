"""Research strategies — research pipelines that build prompt-template contexts.

Each strategy is a callable that takes a question and config and returns
the context dict consumed by prompt templates (.format(**ctx)).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import tiktoken

from common.config import BotConfig
from common.models import ForecastQuestion, build_question_context
from llm.gateway import LLMRequest
from research.asknews import AskNewsSearcher
from research.exa import ExaSearcher
from research.query_plan import (
    DEFAULT_BASE_RATE_CAP,
    DEFAULT_CLAUSE_CAP,
    DEFAULT_RECENCY_CAP,
    FollowupQueries,
    QueryPlan,
)
from research.result import ResearchResult
from research.searcher import Searcher

logger = logging.getLogger(__name__)

ResearchStrategy = Callable[[ForecastQuestion, BotConfig], Awaitable[dict[str, Any]]]

# Cap on the research report fed into forecast prompts.
MAX_REPORT_TOKENS = 4000

# Per-query result caps for the planned pipeline.
_CLAUSE_CAP = 3  # broad coverage, shallow — one good answer per criterion
_RECENCY_CAP = 5  # deep but narrow — news, dates matter
_BASE_RATE_CAP = 3  # one good reference class — token-dense, least perishable


# ── research strategy entry points ───────────────────────────────────


async def no_research(question: ForecastQuestion, _config: BotConfig) -> dict[str, Any]:
    return build_question_context(question)


async def _fallback_research(question: ForecastQuestion, config: BotConfig) -> dict[str, Any]:
    """Generic decompose-and-search via Exa, no planner.

    Used when the planner fails; forced with ``--rc plan=legacy``.
    """
    researcher = ExaSearcher.from_env()
    queries = await _decompose(question, researcher, config)
    ctx = build_question_context(question)
    ctx["summary_report"] = await _compose(researcher, queries)
    ctx["_query_plan"] = None
    return ctx


def _build_planner_prompt(question: ForecastQuestion) -> str:
    template = (Path(__file__).parent / "prompts" / "planner.txt").read_text()
    ctx = build_question_context(question)
    return template.format(**ctx)


async def planned_research(question: ForecastQuestion, config: BotConfig) -> dict[str, Any]:
    """Criteria-planned research pipeline — the default.

    1. Planner: extract structured QueryPlan via LLM.
    2. Hop 1: search all queries in parallel with per-query routing.
    3. Synthesise results grouped by plan section.
    """
    rc = config.research_config
    if rc.get("plan") == "legacy":
        return await _fallback_research(question, config)

    # ── 1. PLANNER ──────────────────────────────────────────────
    plan: QueryPlan | None = None
    try:
        planner_prompt = _build_planner_prompt(question)
        plan = await config.gateway.extract_structured(
            LLMRequest(prompt=planner_prompt, purpose="decompose"),
            QueryPlan,
        )
    except Exception:
        logger.warning("Planner failed, falling back to generic decomposition", exc_info=True)

    if plan is None:
        return await _fallback_research(question, config)

    # Cap query counts
    plan.clause_queries = plan.clause_queries[:DEFAULT_CLAUSE_CAP]
    plan.recency_queries = plan.recency_queries[:DEFAULT_RECENCY_CAP]
    plan.base_rate_queries = plan.base_rate_queries[:DEFAULT_BASE_RATE_CAP]

    # ── 2. HOP 1 — SEARCH ───────────────────────────────────────
    clause_queries = plan.clause_queries
    recency_queries = plan.recency_queries
    base_rate_queries = plan.base_rate_queries
    all_queries = clause_queries + recency_queries + base_rate_queries

    if not all_queries:
        ctx = build_question_context(question)
        ctx["summary_report"] = "No search queries generated."
        ctx["_query_plan"] = plan.model_dump()
        return ctx

    # Build searchers with routing
    searchers: dict[str, Searcher] = {"exa": ExaSearcher.from_env()}

    if rc.get("news") == "asknews":
        try:
            searchers["asknews"] = AskNewsSearcher.from_env()
        except (RuntimeError, ImportError):
            logger.warning("AskNews unavailable — recency queries fall back to Exa")
            searchers["asknews"] = searchers["exa"]

    # Build (query, searcher) pairs with routing
    pairs: list[tuple[str, Searcher]] = []
    for q in clause_queries:
        pairs.append((q, searchers["exa"]))
    for q in recency_queries:
        recency_searcher = searchers.get("asknews") or searchers["exa"]
        pairs.append((q, recency_searcher))
    for q in base_rate_queries:
        pairs.append((q, searchers["exa"]))

    async def _search_safe(query: str, searcher: Searcher) -> list[ResearchResult]:
        try:
            return await searcher.search(query)
        except Exception:
            logger.warning("Search failed for query: %s", query[:80], exc_info=True)
            return []

    start = time.perf_counter()
    batches = await asyncio.gather(*[_search_safe(q, s) for q, s in pairs])

    # Per-query caps + deduplication + group by section
    seen_urls: set[str] = set()
    clause_results: list[ResearchResult] = []
    recency_results: list[ResearchResult] = []
    base_rate_results: list[ResearchResult] = []

    n_clause = len(clause_queries)
    n_recency = len(recency_queries)

    for i, batch in enumerate(batches):
        if i < n_clause:
            cap = _CLAUSE_CAP
            target = clause_results
        elif i < n_clause + n_recency:
            cap = _RECENCY_CAP
            target = recency_results
        else:
            cap = _BASE_RATE_CAP
            target = base_rate_results
        for r in batch[:cap]:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                target.append(r)

    # ── 2.5 HOP 2 — GAP CHECK (optional) ───────────────────────
    followup_queries: list[str] | None = None

    if rc.get("hops") == 2:
        gap_input = _build_gap_input(
            question, plan, clause_results, recency_results, base_rate_results
        )
        try:
            followup = await config.gateway.extract_structured(
                LLMRequest(prompt=gap_input, purpose="decompose"),
                FollowupQueries,
            )
        except Exception:
            logger.warning("Gap check failed, skipping hop 2", exc_info=True)
            followup = None

        if followup and followup.followup_queries:
            followup_queries = followup.followup_queries[:3]
            logger.info("Hop 2: %d follow-up queries", len(followup_queries))
            hop2_results = await _run_hop(followup_queries, searchers["exa"])
            for r in hop2_results:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    clause_results.append(r)

    # ── 3. SYNTHESIS ────────────────────────────────────────────
    summary_report, ntokens = _synthesize(
        plan, clause_results, recency_results, base_rate_results
    )
    elapsed = time.perf_counter() - start
    total_results = len(clause_results) + len(recency_results) + len(base_rate_results)
    logger.info(
        "%d queries, %d results kept, %.1fs, %d tokens",
        len(all_queries),
        total_results,
        elapsed,
        ntokens,
    )

    # ── 4. RETURN ───────────────────────────────────────────────
    ctx = build_question_context(question)
    ctx["summary_report"] = summary_report
    ctx["_query_plan"] = plan.model_dump()
    ctx["_followup_queries"] = followup_queries
    return ctx


def _build_gap_input(
    question: ForecastQuestion,
    plan: QueryPlan,
    clause_results: list[ResearchResult],
    recency_results: list[ResearchResult],
    base_rate_results: list[ResearchResult],
) -> str:
    """Build the hop-2 gap-check prompt, capped at ~2000 tokens."""
    enc = tiktoken.get_encoding("cl100k_base")

    def fmt_section(name: str, results: list[ResearchResult], queries: list[str]) -> str:
        q_lines = "\n".join(f"  - {q}" for q in queries) if queries else "  - none"
        header = f"## {name}\nQueries:\n{q_lines}"
        if not results:
            return f"{header}\n[no results]"
        lines = [header]
        for j, r in enumerate(results):
            date = (r.published_date or "")[:10]
            dated = f" (published {date})" if date else ""
            snippet = r.text[:200].replace("\n", " ")
            lines.append(f"[{j + 1}] {r.title}{dated} — {snippet}")
        return "\n".join(lines)

    results_summary = "\n\n".join(
        [
            fmt_section("Clause Results", clause_results, plan.clause_queries),
            fmt_section("Recency Results", recency_results, plan.recency_queries),
            fmt_section("Base Rate Results", base_rate_results, plan.base_rate_queries),
        ]
    )

    template = (Path(__file__).parent / "prompts" / "gap_check.txt").read_text()
    ctx = build_question_context(question)
    ctx["clause_queries"] = "; ".join(plan.clause_queries) if plan.clause_queries else "none"
    ctx["recency_queries"] = "; ".join(plan.recency_queries) if plan.recency_queries else "none"
    ctx["base_rate_queries"] = (
        "; ".join(plan.base_rate_queries) if plan.base_rate_queries else "none"
    )
    ctx["results_summary"] = results_summary

    full = template.format(**ctx)
    tokens = enc.encode(full)
    if len(tokens) > 2000:
        full = enc.decode(tokens[:2000])
    return full


async def _run_hop(
    queries: list[str],
    searcher: Searcher,
    cap: int = 3,
) -> list[ResearchResult]:
    """Run a batch of queries on one searcher, deduplicate by URL."""
    if not queries:
        return []
    batches = await asyncio.gather(*[searcher.search(q) for q in queries])
    seen: set[str] = set()
    results: list[ResearchResult] = []
    for batch in batches:
        for r in batch[:cap]:
            if r.url not in seen:
                seen.add(r.url)
                results.append(r)
    return results


def _synthesize(
    plan: QueryPlan,
    clause_results: list[ResearchResult],
    recency_results: list[ResearchResult],
    base_rate_results: list[ResearchResult],
) -> tuple[str, int]:
    """Format results grouped by plan section, capped at MAX_REPORT_TOKENS."""
    enc = tiktoken.get_encoding("cl100k_base")

    sections: list[str] = []
    total_tokens = 0

    def add_section(header: str, results: list[ResearchResult], queries: list[str]) -> None:
        nonlocal total_tokens
        query_summary = "; ".join(queries) if queries else "none"
        header_text = f"## {header}\nQueries: {query_summary}\n"
        header_tokens = len(enc.encode(header_text))
        if total_tokens + header_tokens > MAX_REPORT_TOKENS:
            return
        parts = [header_text]
        total_tokens += header_tokens
        for j, r in enumerate(results):
            part = _format_one(j, r)
            part_tokens = len(enc.encode(part))
            if total_tokens + part_tokens > MAX_REPORT_TOKENS:
                break
            parts.append(part)
            total_tokens += part_tokens
        sections.append("\n".join(parts))

    add_section("Clause Results", clause_results, plan.clause_queries)
    add_section("Recency Results", recency_results, plan.recency_queries)
    add_section("Base Rate Results", base_rate_results, plan.base_rate_queries)

    report = "\n\n".join(sections) if sections else "No relevant information found."
    return report, total_tokens


# ── pipeline internals ───────────────────────────────────────────────


async def _decompose(
    question: ForecastQuestion,
    researcher: Any,
    config: Any,
) -> list[str]:
    prompt = researcher.build_query_prompt(question)
    result = await config.gateway.complete(
        LLMRequest(prompt=prompt, temperature=0.7, purpose="decompose")
    )
    lines = [line.strip() for line in result.strip().splitlines() if line.strip()]
    if not lines:
        logger.warning("Decompose LLM returned empty output, using question title as query")
        return [question.title]
    return lines


def _format_one(i: int, r: ResearchResult) -> str:
    date = (r.published_date or "")[:10]
    dated = f" (published {date})" if date else ""
    return f"[{i + 1}] {r.title}{dated}\n    URL: {r.url}\n    {r.text}\n"


async def _compose(
    researcher: Any,
    queries: str | list[str],
) -> str:
    query_list = [queries] if isinstance(queries, str) else queries
    logger.info("Using researcher: %s", type(researcher).__name__)
    for i, q in enumerate(query_list):
        logger.info("  [%d] %s", i + 1, q)

    try:
        all_results: list[ResearchResult] = []
        seen_urls: set[str] = set()

        start = time.perf_counter()
        batches = await asyncio.gather(*[researcher.search(q) for q in query_list])
        for i, (q, batch) in enumerate(zip(query_list, batches, strict=True)):
            logger.debug("  [%d] %d results — %s", i + 1, len(batch), q[:100])
        for batch in batches:
            for r in batch:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    all_results.append(r)

        # Best score first; recency breaks ties so the token cap keeps fresh sources.
        all_results.sort(key=lambda r: (r.score or 0, r.published_date or ""), reverse=True)

        if not all_results:
            return "No relevant information found."

        enc = tiktoken.get_encoding("cl100k_base")
        parts: list[str] = []
        ntokens = 0
        for i, r in enumerate(all_results):
            part = _format_one(i, r)
            part_tokens = len(enc.encode(part))
            if parts and ntokens + part_tokens > MAX_REPORT_TOKENS:
                break
            parts.append(part)
            ntokens += part_tokens
        context = "\n".join(parts)
        elapsed = time.perf_counter() - start

        logger.info(
            f"{len(query_list)} requests, {len(parts)}/{len(all_results)} results kept, "
            f"{elapsed:.1f}s, {ntokens} tokens"
        )
        return context
    except Exception as e:
        logger.warning("compose_research failed", exc_info=True)
        return f"Research failed with error: {e}. Proceeding without search results."
