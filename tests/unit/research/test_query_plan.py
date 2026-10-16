"""Unit tests for QueryPlan and FollowupQueries."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from research.query_plan import (
    DEFAULT_BASE_RATE_CAP,
    DEFAULT_CLAUSE_CAP,
    DEFAULT_RECENCY_CAP,
    FollowupQueries,
    QueryPlan,
)


class TestQueryPlan:
    def test_extraction_produces_correct_fields(self) -> None:
        plan = QueryPlan(
            clause_queries=["criterion A search", "criterion B search"],
            recency_queries=["recent news since 2026-05-29"],
            base_rate_queries=["historical base rate of similar events"],
        )
        assert isinstance(plan.clause_queries, list)
        assert isinstance(plan.recency_queries, list)
        assert isinstance(plan.base_rate_queries, list)
        assert len(plan.clause_queries) == 2
        assert len(plan.recency_queries) == 1
        assert len(plan.base_rate_queries) == 1

    def test_empty_sections_allowed(self) -> None:
        plan = QueryPlan(
            clause_queries=["only clause"],
            recency_queries=[],
            base_rate_queries=[],
        )
        assert plan.clause_queries == ["only clause"]
        assert plan.recency_queries == []
        assert plan.base_rate_queries == []

    def test_missing_field_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            QueryPlan(clause_queries=["q"])  # type: ignore[call-arg]

    def test_model_dump_serializable(self) -> None:
        plan = QueryPlan(
            clause_queries=["q1"],
            recency_queries=["q2"],
            base_rate_queries=["q3"],
        )
        d = plan.model_dump()
        assert d == {
            "clause_queries": ["q1"],
            "recency_queries": ["q2"],
            "base_rate_queries": ["q3"],
        }


class TestFollowupQueries:
    def test_extraction_produces_correct_fields(self) -> None:
        fq = FollowupQueries(followup_queries=["gap query 1", "gap query 2"])
        assert isinstance(fq.followup_queries, list)
        assert len(fq.followup_queries) == 2

    def test_empty_list_allowed(self) -> None:
        fq = FollowupQueries(followup_queries=[])
        assert fq.followup_queries == []

    def test_missing_field_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            FollowupQueries()  # type: ignore[call-arg]


class TestDefaultCaps:
    def test_default_clause_cap(self) -> None:
        assert DEFAULT_CLAUSE_CAP == 4

    def test_default_recency_cap(self) -> None:
        assert DEFAULT_RECENCY_CAP == 2

    def test_default_base_rate_cap(self) -> None:
        assert DEFAULT_BASE_RATE_CAP == 3
