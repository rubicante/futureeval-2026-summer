"""Unit tests for the Searcher Protocol."""

from __future__ import annotations

from research.exa import ExaSearcher
from research.result import ResearchResult
from research.searcher import Searcher


class TestSearcherProtocol:
    def test_exa_searcher_conforms(self) -> None:
        assert issubclass(ExaSearcher, Searcher)

    def test_asknews_searcher_conforms(self) -> None:
        from research.asknews import AskNewsSearcher

        assert issubclass(AskNewsSearcher, Searcher)

    def test_mock_implements_search(self) -> None:
        """A class with async search of the right shape passes isinstance."""

        class MockSearch:
            async def search(self, query: str) -> list[ResearchResult]:
                return []

        assert isinstance(MockSearch(), Searcher)

    def test_missing_search_rejected(self) -> None:
        class Bad:
            def lookup(self) -> None: ...

        assert not isinstance(Bad(), Searcher)
