"""Searcher Protocol — shared interface for search backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from research.result import ResearchResult


@runtime_checkable
class Searcher(Protocol):
    async def search(self, query: str) -> list[ResearchResult]: ...
