"""Unit tests for research.strategies — formatting, planned pipeline, and fallback path."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.models import ForecastQuestion, QuestionType
from research.query_plan import DEFAULT_CLAUSE_CAP, FollowupQueries, QueryPlan
from research.result import ResearchResult
from research.strategies import _format_one, planned_research


class TestResearchResult:
    def test_create_full(self):
        r = ResearchResult(
            title="Test title",
            url="https://example.com",
            text="Sample text content.",
            score=0.95,
        )
        assert r.title == "Test title"
        assert r.url == "https://example.com"
        assert r.text == "Sample text content."
        assert r.score == 0.95

    def test_create_minimal(self):
        r = ResearchResult(
            title="Minimal",
            url="https://example.com",
            text="Text",
        )
        assert r.score is None

    def test_field_access_by_name(self):
        r = ResearchResult("Title", "URL", "text", 0.5)
        assert r[0] == "Title"
        assert r[1] == "URL"
        assert r.score == 0.5


class TestFormatSearchResults:
    def test_single_result(self):
        r = ResearchResult(
            title="Article 1",
            url="https://example.com/1",
            text="Content of article 1.",
            score=0.8,
        )
        formatted = _format_one(0, r)
        assert "[1]" in formatted
        assert "Article 1" in formatted
        assert "https://example.com/1" in formatted
        assert "Content of article 1." in formatted

    def test_published_date_included(self):
        r = ResearchResult(
            "T", "https://x.com/page", "text", 0.5, published_date="2026-06-01T12:00:00Z"
        )
        formatted = _format_one(0, r)
        assert "(published 2026-06-01)" in formatted

    def test_no_date_omits_published(self):
        r = ResearchResult("T", "https://x.com/page", "text", 0.5)
        formatted = _format_one(0, r)
        assert "published" not in formatted
        assert "https://x.com/page" in formatted


# ── planned pipeline tests ──────────────────────────────────────────


def _make_gateway(extract_result=None, extract_side_effect=None, complete_result="") -> MagicMock:
    gw = MagicMock()
    gw.extract_structured = AsyncMock(return_value=extract_result, side_effect=extract_side_effect)
    gw.complete = AsyncMock(return_value=complete_result)
    return gw


def _sample_question() -> ForecastQuestion:
    return ForecastQuestion(
        title="Will X happen by 2030?",
        body="Background text.\n\nResolution: X is defined as...\n\nFine print.",
        question_type=QuestionType.BINARY,
        resolve_time="2030-01-01T00:00:00Z",
        url="https://www.metaculus.com/questions/123/",
    )


def _sample_results(count: int = 2, offset: int = 0) -> list[ResearchResult]:
    return [
        ResearchResult(
            title=f"Article {i + offset}",
            url=f"https://example.com/{i + offset}",
            text=f"Content of article {i + offset}.",
            score=0.9 - i * 0.1,
            published_date=f"2026-06-{10 + i:02d}T12:00:00Z",
        )
        for i in range(count)
    ]


def _sample_plan() -> QueryPlan:
    return QueryPlan(
        clause_queries=["evidence for criterion A", "evidence for criterion B"],
        recency_queries=["recent news since 2026-05-29 about X"],
        base_rate_queries=["historical base rate of events like X"],
    )


class TestPlannedResearch:
    @pytest.mark.asyncio
    async def test_default_path_populates_query_plan(self):
        """The default planned pipeline populates _query_plan in context."""
        question = _sample_question()
        plan = _sample_plan()
        gw = _make_gateway(extract_result=plan)

        with patch("research.strategies.ExaSearcher") as mock_exa_cls:
            mock_exa = MagicMock()
            mock_exa.search = AsyncMock(return_value=_sample_results(2))
            mock_exa_cls.from_env.return_value = mock_exa
            config = SimpleNamespace(research_config={}, gateway=gw)
            ctx = await planned_research(question, config)

        assert "_query_plan" in ctx
        assert ctx["_query_plan"] is not None
        assert ctx["_query_plan"]["clause_queries"] == plan.clause_queries
        assert "summary_report" in ctx
        assert "Clause Results" in ctx["summary_report"]

    @pytest.mark.asyncio
    async def test_planner_capping_truncates_excess_queries(self):
        """>4 clause queries → truncated to DEFAULT_CLAUSE_CAP."""
        question = _sample_question()
        plan = QueryPlan(
            clause_queries=[f"q{i}" for i in range(6)],
            recency_queries=[],
            base_rate_queries=[],
        )
        gw = _make_gateway(extract_result=plan)

        with patch("research.strategies.ExaSearcher") as mock_exa_cls:
            mock_exa = MagicMock()
            mock_exa.search = AsyncMock(return_value=_sample_results(1))
            mock_exa_cls.from_env.return_value = mock_exa
            config = SimpleNamespace(research_config={}, gateway=gw)
            ctx = await planned_research(question, config)

        # The plan in context should have only DEFAULT_CLAUSE_CAP clause queries.
        stored = ctx["_query_plan"]
        assert stored is not None
        assert len(stored["clause_queries"]) == DEFAULT_CLAUSE_CAP
        # Only DEFAULT_CLAUSE_CAP searches were actually run.
        assert mock_exa.search.await_count == DEFAULT_CLAUSE_CAP

    @pytest.mark.asyncio
    async def test_planner_failure_falls_back_to_generic_decomposition(self):
        """Planner failure → fall back to ExaSearcher.build_query_prompt + _decompose."""
        question = _sample_question()
        gw = _make_gateway(extract_side_effect=RuntimeError("planner failed"))
        gw.complete = AsyncMock(return_value="fallback query 1\nfallback query 2")

        with patch("research.strategies.ExaSearcher") as mock_exa_cls:
            mock_exa = MagicMock()
            mock_exa.search = AsyncMock(return_value=_sample_results(2))
            mock_exa_cls.from_env.return_value = mock_exa
            config = SimpleNamespace(research_config={}, gateway=gw)
            ctx = await planned_research(question, config)

        assert "_query_plan" in ctx
        assert ctx["_query_plan"] is None
        assert "summary_report" in ctx
        assert ctx["summary_report"] != ""
        # The legacy decompose path was used.
        gw.complete.assert_awaited()

    @pytest.mark.asyncio
    async def test_plan_legacy_forces_fallback(self):
        """plan=legacy skips the planner and uses the fallback path."""
        question = _sample_question()
        gw = _make_gateway()
        gw.complete = AsyncMock(return_value="legacy query a\nlegacy query b")

        with patch("research.strategies.ExaSearcher") as mock_exa_cls:
            mock_exa = MagicMock()
            mock_exa.search = AsyncMock(return_value=_sample_results(2))
            mock_exa_cls.from_env.return_value = mock_exa
            config = SimpleNamespace(research_config={"plan": "legacy"}, gateway=gw)
            ctx = await planned_research(question, config)

        assert "summary_report" in ctx
        assert ctx["summary_report"] != ""
        assert ctx["_query_plan"] is None
        # The planner was never consulted; fallback decompose was.
        gw.extract_structured.assert_not_awaited()
        gw.complete.assert_awaited()

    @pytest.mark.asyncio
    async def test_handles_empty_queries(self):
        """When plan has no queries, report states no queries generated."""
        question = _sample_question()
        plan = QueryPlan(
            clause_queries=[],
            recency_queries=[],
            base_rate_queries=[],
        )
        gw = _make_gateway(extract_result=plan)

        config = SimpleNamespace(research_config={}, gateway=gw)
        ctx = await planned_research(question, config)

        assert ctx["_query_plan"] is not None
        assert ctx["summary_report"] == "No search queries generated."

    @pytest.mark.asyncio
    async def test_result_grouping_by_section(self):
        """Results are grouped by section in the synthesis output."""
        question = _sample_question()
        plan = QueryPlan(
            clause_queries=["clause q"],
            recency_queries=["recency q"],
            base_rate_queries=["base rate q"],
        )
        gw = _make_gateway(extract_result=plan)

        with patch("research.strategies.ExaSearcher") as mock_exa_cls:
            mock_exa = MagicMock()
            # Unique URLs per batch to avoid cross-section dedup.
            clause_r = _sample_results(1, offset=0)
            recency_r = _sample_results(1, offset=10)
            base_rate_r = _sample_results(1, offset=20)
            mock_exa.search = AsyncMock(side_effect=[clause_r, recency_r, base_rate_r])
            mock_exa_cls.from_env.return_value = mock_exa

            config = SimpleNamespace(research_config={}, gateway=gw)
            ctx = await planned_research(question, config)

        report = ctx["summary_report"]
        assert "Clause Results" in report
        assert "Recency Results" in report
        assert "Base Rate Results" in report


class TestSearcherRouting:
    @pytest.mark.asyncio
    async def test_routing_news_asknews_routes_recency_to_asknews(self):
        """news=asknews routes recency queries to AskNews, clause/base-rate to Exa."""
        question = _sample_question()
        plan = QueryPlan(
            clause_queries=["clause q"],
            recency_queries=["recency q"],
            base_rate_queries=["base rate q"],
        )
        gw = _make_gateway(extract_result=plan)

        exa_search = AsyncMock(return_value=_sample_results(1, offset=0))
        asknews_search = AsyncMock(return_value=_sample_results(1, offset=100))

        exa_mock = MagicMock()
        exa_mock.search = exa_search
        asknews_mock = MagicMock()
        asknews_mock.search = asknews_search

        with (
            patch("research.strategies.ExaSearcher") as mock_exa_cls,
            patch("research.strategies.AskNewsSearcher") as mock_asknews_cls,
        ):
            mock_exa_cls.from_env.return_value = exa_mock
            mock_asknews_cls.from_env.return_value = asknews_mock
            config = SimpleNamespace(
                research_config={"news": "asknews"}, gateway=gw
            )
            ctx = await planned_research(question, config)

        # Exa is called for clause + base_rate (2 calls).
        assert exa_search.await_count == 2
        # AskNews is called for recency (1 call).
        assert asknews_search.await_count == 1

        report = ctx["summary_report"]
        assert "Clause Results" in report
        assert "Recency Results" in report
        assert "Base Rate Results" in report

    @pytest.mark.asyncio
    async def test_routing_asknews_unavailable_falls_back_to_exa(self):
        """When AskNews raises, recency queries fall back to Exa."""
        question = _sample_question()
        plan = QueryPlan(
            clause_queries=["clause q"],
            recency_queries=["recency q"],
            base_rate_queries=[],
        )
        gw = _make_gateway(extract_result=plan)

        exa_search = AsyncMock(return_value=_sample_results(1, offset=0))

        exa_mock = MagicMock()
        exa_mock.search = exa_search

        with (
            patch("research.strategies.ExaSearcher") as mock_exa_cls,
            patch("research.strategies.AskNewsSearcher") as mock_asknews_cls,
        ):
            mock_exa_cls.from_env.return_value = exa_mock
            mock_asknews_cls.from_env.side_effect = RuntimeError("no creds")
            config = SimpleNamespace(
                research_config={"news": "asknews"}, gateway=gw
            )
            ctx = await planned_research(question, config)

        # Exa handles clause + recency (2 calls).
        assert exa_search.await_count == 2

        report = ctx["summary_report"]
        assert "Clause Results" in report
        assert "Recency Results" in report

    @pytest.mark.asyncio
    async def test_routing_clause_and_base_rate_always_exa(self):
        """Clause and base-rate queries always go to Exa regardless of news config."""
        question = _sample_question()
        plan = QueryPlan(
            clause_queries=["clause q"],
            recency_queries=[],
            base_rate_queries=["base rate q"],
        )
        gw = _make_gateway(extract_result=plan)

        exa_search = AsyncMock(return_value=_sample_results(1, offset=0))

        exa_mock = MagicMock()
        exa_mock.search = exa_search

        with (
            patch("research.strategies.ExaSearcher") as mock_exa_cls,
            patch("research.strategies.AskNewsSearcher") as mock_asknews_cls,
        ):
            mock_exa_cls.from_env.return_value = exa_mock
            mock_asknews_cls.from_env.return_value = MagicMock()
            config = SimpleNamespace(
                research_config={"news": "asknews"}, gateway=gw
            )
            ctx = await planned_research(question, config)

        # Only Exa is called — no recency queries.
        assert exa_search.await_count == 2
        report = ctx["summary_report"]
        assert "Clause Results" in report
        assert "Base Rate Results" in report


class TestHop2:
    @pytest.mark.asyncio
    async def test_hops_2_with_gaps_fires_hop_2(self):
        """hops=2 with non-empty follow-up queries → hop 2 runs."""
        question = _sample_question()
        plan = QueryPlan(
            clause_queries=["clause q"],
            recency_queries=[],
            base_rate_queries=[],
        )
        followup = FollowupQueries(followup_queries=["gap query 1", "gap query 2"])
        gw = MagicMock()
        gw.extract_structured = AsyncMock(side_effect=[plan, followup])

        exa_search = AsyncMock(return_value=_sample_results(1, offset=0))
        exa_mock = MagicMock()
        exa_mock.search = exa_search

        with patch("research.strategies.ExaSearcher") as mock_exa_cls:
            mock_exa_cls.from_env.return_value = exa_mock
            config = SimpleNamespace(research_config={"hops": 2}, gateway=gw)
            ctx = await planned_research(question, config)

        # extract_structured called twice: planner + gap check
        assert gw.extract_structured.await_count == 2
        # search called for clause (1) + follow-up queries (2) = 3
        assert exa_search.await_count == 3
        assert ctx["_followup_queries"] == ["gap query 1", "gap query 2"]

    @pytest.mark.asyncio
    async def test_hops_2_with_full_coverage_skips_hop_2(self):
        """hops=2 with empty follow-up → hop 2 skipped."""
        question = _sample_question()
        plan = QueryPlan(
            clause_queries=["clause q"],
            recency_queries=[],
            base_rate_queries=[],
        )
        empty_followup = FollowupQueries(followup_queries=[])
        gw = MagicMock()
        gw.extract_structured = AsyncMock(side_effect=[plan, empty_followup])

        exa_search = AsyncMock(return_value=_sample_results(1, offset=0))
        exa_mock = MagicMock()
        exa_mock.search = exa_search

        with patch("research.strategies.ExaSearcher") as mock_exa_cls:
            mock_exa_cls.from_env.return_value = exa_mock
            config = SimpleNamespace(research_config={"hops": 2}, gateway=gw)
            ctx = await planned_research(question, config)

        # extract_structured called twice.
        assert gw.extract_structured.await_count == 2
        # search only called for clause (not for follow-up).
        assert exa_search.await_count == 1
        assert ctx["_followup_queries"] is None

    @pytest.mark.asyncio
    async def test_hops_1_default_never_fires_hop_2(self):
        """Without hops=2 knob, hop 2 never fires."""
        question = _sample_question()
        plan = QueryPlan(
            clause_queries=["clause q"],
            recency_queries=[],
            base_rate_queries=[],
        )
        gw = _make_gateway(extract_result=plan)

        exa_search = AsyncMock(return_value=_sample_results(1, offset=0))
        exa_mock = MagicMock()
        exa_mock.search = exa_search

        with patch("research.strategies.ExaSearcher") as mock_exa_cls:
            mock_exa_cls.from_env.return_value = exa_mock
            config = SimpleNamespace(research_config={}, gateway=gw)
            ctx = await planned_research(question, config)

        # extract_structured called only once (planner).
        assert gw.extract_structured.await_count == 1
        assert "_followup_queries" in ctx
        assert ctx["_followup_queries"] is None

    @pytest.mark.asyncio
    async def test_hops_2_followup_capped_at_3(self):
        """Follow-up queries from gap check are capped at 3."""
        question = _sample_question()
        plan = QueryPlan(
            clause_queries=["clause q"],
            recency_queries=[],
            base_rate_queries=[],
        )
        many_followup = FollowupQueries(followup_queries=["q1", "q2", "q3", "q4", "q5"])
        gw = MagicMock()
        gw.extract_structured = AsyncMock(side_effect=[plan, many_followup])

        exa_search = AsyncMock(return_value=_sample_results(1, offset=0))
        exa_mock = MagicMock()
        exa_mock.search = exa_search

        with patch("research.strategies.ExaSearcher") as mock_exa_cls:
            mock_exa_cls.from_env.return_value = exa_mock
            config = SimpleNamespace(research_config={"hops": 2}, gateway=gw)
            ctx = await planned_research(question, config)

        # Only 3 of the 5 follow-up queries were used.
        assert ctx["_followup_queries"] == ["q1", "q2", "q3"]
        # Search called: clause (1) + 3 follow-up = 4
        assert exa_search.await_count == 4

    def test_gap_input_empty_sections_show_no_results(self):
        """_build_gap_input shows [no results] for empty sections."""
        from research.strategies import _build_gap_input

        question = _sample_question()
        plan = QueryPlan(
            clause_queries=["c"],
            recency_queries=[],
            base_rate_queries=[],
        )
        gap = _build_gap_input(question, plan, [], [], [])
        assert "[no results]" in gap
        assert "Clause Results" in gap

    def test_followup_queries_captures_in_context(self):
        """FollowupQueries model serialises correctly."""
        fq = FollowupQueries(followup_queries=["a", "b"])
        assert fq.followup_queries == ["a", "b"]
        assert fq.model_dump() == {"followup_queries": ["a", "b"]}
        empty = FollowupQueries(followup_queries=[])
        assert empty.followup_queries == []


class TestAskNewsBuildQueryPromptDeleted:
    def test_build_query_prompt_not_present(self):
        """After deletion, AskNewsSearcher has no build_query_prompt method."""
        from research.asknews import AskNewsSearcher

        # Verify the search method still exists.
        assert hasattr(AskNewsSearcher, "search")
        assert callable(AskNewsSearcher.search)

        # Verify build_query_prompt is gone.
        assert not hasattr(AskNewsSearcher, "build_query_prompt")
