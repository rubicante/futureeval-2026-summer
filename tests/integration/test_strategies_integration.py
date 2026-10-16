"""Integration tests for research.strategies and forecasting.strategies."""

from __future__ import annotations

import datetime
from types import SimpleNamespace
from unittest import mock

import pytest

from common.models import ForecastQuestion, QuestionType
from research.asknews import AskNewsSearcher
from research.exa import ExaSearcher
from research.result import ResearchResult
from research.strategies import _compose, _decompose
from tests.integration.conftest import FakeLLMGateway


@pytest.fixture
def composer_config() -> SimpleNamespace:
    cfg = SimpleNamespace()
    cfg.gateway = FakeLLMGateway()
    cfg.disable_research = False
    return cfg


@pytest.fixture
def sample_question() -> ForecastQuestion:
    return ForecastQuestion(
        title="Will AI surpass human intelligence by 2030?",
        body="Test description for AI forecasting.",
        question_type=QuestionType.BINARY,
    )


# ── decompose ───────────────────────────────────────────────────────


class TestDecompose:
    async def test_returns_parsed_queries(self, composer_config, sample_question):
        async def fake_complete(request):
            return "AI progress 2025\nmachine learning benchmarks\nAGI expert predictions"

        researcher = ExaSearcher(api_key="test-key")
        composer_config.gateway = FakeLLMGateway(complete_fn=fake_complete)
        queries = await _decompose(sample_question, researcher, composer_config)

        assert len(queries) == 3
        assert "AI progress 2025" in queries

    async def test_exception_propagates(self, composer_config, sample_question):
        async def fake_complete(request):
            raise RuntimeError("LLM unavailable")

        researcher = ExaSearcher(api_key="test-key")
        composer_config.gateway = FakeLLMGateway(complete_fn=fake_complete)
        with pytest.raises(RuntimeError, match="LLM unavailable"):
            await _decompose(sample_question, researcher, composer_config)

    async def test_empty_response_falls_back_to_title(self, composer_config, sample_question):
        async def fake_complete(request):
            return ""

        researcher = ExaSearcher(api_key="test-key")
        composer_config.gateway = FakeLLMGateway(complete_fn=fake_complete)
        queries = await _decompose(sample_question, researcher, composer_config)

        assert queries == [sample_question.title]


# ── compose_research ────────────────────────────────────────────────


class TestComposeResearch:
    async def test_returns_formatted_results(self, composer_config):
        class FakeSearcher:
            async def search(self, query: str) -> list[ResearchResult]:
                return [
                    ResearchResult(
                        f"Result for {query}", f"https://x.com/{query}", "Text content", 0.9
                    ),
                ]

        result = await _compose(FakeSearcher(), ["query1", "query2"])
        assert "Result for query1" in result
        assert "Result for query2" in result
        assert "https://x.com/query1" in result

    async def test_deduplicates_by_url(self, composer_config):
        class FakeSearcher:
            async def search(self, query: str) -> list[ResearchResult]:
                return [
                    ResearchResult("Title", "https://same-url.com", "Text", 0.5),
                ]

        result = await _compose(FakeSearcher(), ["q1", "q2"])
        assert result.count("https://same-url.com") == 1

    async def test_empty_results(self, composer_config):
        class EmptySearcher:
            async def search(self, query: str) -> list[ResearchResult]:
                return []

        result = await _compose(EmptySearcher(), ["q1"])
        assert result == "No relevant information found."

    async def test_single_query_string(self, composer_config):
        class FakeSearcher:
            async def search(self, query: str) -> list[ResearchResult]:
                return [ResearchResult("T", "https://u.com", "text", 0.5)]

        result = await _compose(FakeSearcher(), "single query")
        assert "T" in result

    async def test_search_failure_graceful(self, composer_config):
        class BrokenSearcher:
            async def search(self, query: str) -> list[ResearchResult]:
                raise RuntimeError("search error")

        result = await _compose(BrokenSearcher(), ["q1"])
        assert "Research failed" in result


# ── ExaSearcher ──────────────────────────────────────────────────────


class TestExaSearcher:
    def test_from_env_inits_with_key(self):
        with mock.patch.dict("os.environ", {"EXA_API_KEY": "test-key"}, clear=True):
            searcher = ExaSearcher.from_env()
            assert searcher is not None
            assert searcher.api_key == "test-key"

    def test_from_env_raises_without_key(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="EXA_API_KEY not set"):
                ExaSearcher.from_env()

    def test_build_query_prompt_includes_title(self):
        q = ForecastQuestion(
            title="Will AI surpass humans by 2030?",
            body="Test body describing the AI forecasting question.",
            question_type=QuestionType.BINARY,
        )
        prompt = ExaSearcher.build_query_prompt(q)
        assert "Will AI surpass humans by 2030?" in prompt
        assert "Test body" in prompt

    async def test_search_returns_research_results(self, mock_exa_search):
        searcher = ExaSearcher(api_key="test-key")
        results = await searcher.search("test query")
        assert len(results) > 0
        assert isinstance(results[0], ResearchResult)
        assert results[0].title == "Result 1"
        assert results[0].url == "https://example.com/a"

    async def test_search_sorts_by_score_desc(self, mock_exa_search):
        searcher = ExaSearcher(api_key="test-key")
        results = await searcher.search("test query")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    async def test_search_deduplicates_by_text(self, mock_exa_search):
        searcher = ExaSearcher(api_key="test-key")
        results = await searcher.search("test query")
        urls = [r.url for r in results]
        assert urls.count("https://example.com/b") == 1

    async def test_search_skips_empty_highlights(self, mock_exa_search):
        existing_results = mock_exa_search.return_value.search.return_value.results

        class NoHighlightResult:
            def __init__(self, title, url, highlights, score):
                self.title = title
                self.url = url
                self.highlights = highlights
                self.score = score

        existing_results.append(NoHighlightResult("No HL", "https://example.com/z", [], 0.5))
        searcher = ExaSearcher(api_key="test-key")
        results = await searcher.search("test query")
        assert not any(r.title == "No HL" for r in results)

    async def test_search_handles_none_scores(self, mock_exa_search_with_no_scores):
        searcher = ExaSearcher(api_key="test-key")
        results = await searcher.search("test query")
        assert len(results) == 2
        assert results[0].score is None


# ── AskNewsSearcher ──────────────────────────────────────────────────


class TestAskNewsSearcher:
    def test_from_env_inits_with_key(self):
        with mock.patch.dict(
            "os.environ",
            {"ASKNEWS_API_KEY": "test-key"},
            clear=True,
        ):
            searcher = AskNewsSearcher.from_env()
            assert searcher is not None
            assert searcher.api_key == "test-key"

    def test_from_env_raises_without_key(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="ASKNEWS_API_KEY not set"):
                AskNewsSearcher.from_env()

    async def test_search_returns_research_results(self, mock_asknews_search):
        searcher = AskNewsSearcher(api_key="test-key")
        results = await searcher.search("test query")
        assert len(results) == 2
        assert isinstance(results[0], ResearchResult)
        assert results[0].title == "Test Article 1"
        assert results[0].url == "https://example.com/1"
        assert "Summary of article 1" in results[0].text

    async def test_search_includes_pub_date_in_text(self, mock_asknews_search):
        searcher = AskNewsSearcher(api_key="test-key")
        results = await searcher.search("test query")
        assert "January 15, 2025" in results[0].text
        assert "Publish date:" in results[0].text

    async def test_search_includes_source_id_in_text(self, mock_asknews_search):
        searcher = AskNewsSearcher(api_key="test-key")
        results = await searcher.search("test query")
        assert "test-source-1" in results[0].text
        assert "Source:" in results[0].text

    async def test_search_deduplicates_by_url(self, mock_asknews_search):
        class A:
            def __init__(self, u, t, s):
                self.article_url = u
                self.eng_title = t
                self.summary = s
                self.pub_date = datetime.datetime(2025, 1, 15)
                self.language = "en"
                self.source_id = "src"

        mock_asknews_search.return_value.news.search_news.return_value.as_dicts = [
            A("https://same-url.com", "Article A", "Summary A."),
            A("https://same-url.com", "Article B", "Summary B."),
        ]
        searcher = AskNewsSearcher(api_key="test-key")
        results = await searcher.search("test query")
        assert len(results) == 1
        assert results[0].title == "Article A"
