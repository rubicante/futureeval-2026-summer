"""QueryPlan and FollowupQueries — structured outputs for the v1 research pipeline."""

from __future__ import annotations

from pydantic import BaseModel

DEFAULT_CLAUSE_CAP = 4
DEFAULT_RECENCY_CAP = 2
DEFAULT_BASE_RATE_CAP = 3


class QueryPlan(BaseModel):
    clause_queries: list[str]
    recency_queries: list[str]
    base_rate_queries: list[str]


class FollowupQueries(BaseModel):
    followup_queries: list[str]
