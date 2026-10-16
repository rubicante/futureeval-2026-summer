"""ExaSearcher — web search via Exa API (native async SDK)."""

from __future__ import annotations

import logging
import os

from exa_py import AsyncExa
from exa_py.api import ContentsOptions, HighlightsContentsOptions

from common.models import ForecastQuestion, build_question_context
from research.result import ResearchResult

logger = logging.getLogger(__name__)

DECOMPOSITION_PROMPT = """
You are decomposing a forecasting question into search queries for Exa, a semantic search engine that retrieves passages similar in meaning to the query.

Each query you produce will be run as a separate Exa search, and the results will be concatenated into one research document. Because of this, every query must target a distinct slice of evidence — redundant or near-rephrased queries waste a search and skew the document toward one topic.

Write queries as declarative statements resembling text that would appear near the answer — e.g. 'analysts expect X to happen by mid-2026' rather than 'will X happen in 2026'. Avoid question phrasing and keyword-style fragments.

Generate 5 queries that together gather evidence which would meaningfully update a probability estimate. Draw on angles like recent developments, base rates and historical trends, expert or market expectations, primary data sources, and disconfirming or contrarian scenarios — but only include an angle if it yields a genuinely distinct, useful query. Prefer fewer, high-quality queries over padding to fill a quota.

If background or resolution criteria are sparse or missing, rely on the question title and infer the relevant entities and timeframe yourself.

Output only the query lines and nothing else — no numbering, no prefixes, no preamble, no commentary.

Question: {title}
Background: {body}
"""


class ExaSearcher:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    @classmethod
    def from_env(cls) -> ExaSearcher:
        key = os.getenv("EXA_API_KEY")
        if not key:
            raise RuntimeError("EXA_API_KEY not set")
        return cls(key)

    @staticmethod
    def build_query_prompt(question: ForecastQuestion) -> str:
        ctx = build_question_context(question)
        return DECOMPOSITION_PROMPT.format(**ctx)

    async def search(self, query: str) -> list[ResearchResult]:
        logger.debug("Exa search: %s", query[:120])
        exa = AsyncExa(api_key=self.api_key)
        response = await exa.search(
            query,
            type="neural",
            num_results=5,
            contents=ContentsOptions(
                highlights=HighlightsContentsOptions(
                    query=query, num_sentences=3, highlights_per_url=3
                ),
            ),
        )
        results: list[ResearchResult] = []
        seen: set[str] = set()
        for item in response.results:
            highlights = item.highlights or []
            if not highlights:
                continue
            text = "\n".join(highlights)
            if text in seen:
                continue
            seen.add(text)
            results.append(
                ResearchResult(
                    title=item.title or "",
                    url=item.url,
                    text=text,
                    score=getattr(item, "score", None),
                    published_date=getattr(item, "published_date", None),
                )
            )
        result_list = sorted(results, key=lambda r: r.score or 0, reverse=True)
        logger.debug("Exa: %d results for query: %s", len(result_list), query[:120])
        return result_list
