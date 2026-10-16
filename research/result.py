"""ResearchResult — shared return type for all searchers."""

from __future__ import annotations

from typing import NamedTuple


class ResearchResult(NamedTuple):
    title: str
    url: str
    text: str
    score: float | None = None
    published_date: str | None = None
