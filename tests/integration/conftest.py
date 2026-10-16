"""Integration test fixtures — mocked external services with real API shapes."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from unittest import mock

import pytest
from pytest_httpx import HTTPXMock

from common.config import BotConfig
from common.models import ForecastQuestion
from llm.gateway import LLMRequest
from metaculus.adapter import MetaculusAdapter
from metaculus.client import MetaculusClient

# ── Real API shape fixtures (loaded from docs/) ─────────────────────


def _load_json(name: str) -> dict:
    path = Path(__file__).parent / "fixtures" / name
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def api_post_details_binary() -> dict:
    return _load_json("api_post_details_binary.json")


@pytest.fixture(scope="session")
def api_post_details_numeric() -> dict:
    return _load_json("api_post_details_numeric.json")


@pytest.fixture(scope="session")
def api_post_details_discrete() -> dict:
    return _load_json("api_post_details_discrete.json")


@pytest.fixture(scope="session")
def api_post_details_date() -> dict:
    return _load_json("api_post_details_date.json")


@pytest.fixture(scope="session")
def api_post_details_mc() -> dict:
    return _load_json("api_post_details_multiple_choice.json")


@pytest.fixture(scope="session")
def api_post_details_binary_with_forecast() -> dict:
    return _load_json("api_post_details_binary_with_forecast.json")


@pytest.fixture(scope="session")
def api_posts_list_response() -> dict:
    return _load_json("api_posts_list_response.json")


@pytest.fixture(scope="session")
def api_paginated_result_item() -> dict:
    return _load_json("api_paginated_result_item.json")


# ── Client with mocked token ────────────────────────────────────────


@pytest.fixture
def metaculus_client() -> MetaculusClient:
    return MetaculusClient(token="test-token")


@pytest.fixture
def base_url() -> str:
    return "https://www.metaculus.com/api"


# ── Mock helpers for paginated list responses ───────────────────────


@pytest.fixture
def mock_list_single_page(httpx_mock: HTTPXMock, base_url: str, api_paginated_result_item: dict):
    """Register a single-page /api/posts/ response with one result."""

    def _register(
        num_results: int = 1,
        limit: int = 50,
        offset: int = 0,
        question_overrides: dict | None = None,
    ):
        results: list[dict] = []
        for i in range(num_results):
            item = json.loads(json.dumps(api_paginated_result_item))
            item["id"] = 43327 + i
            q = item["question"]
            q["id"] = 43332 + i
            if question_overrides:
                q.update(question_overrides)
            results.append(item)

        httpx_mock.add_response(
            url=f"{base_url}/posts/?limit={limit}&offset={offset}&tournaments=bot-testing-area",
            json={
                "results": results,
                "next": None
                if num_results < limit
                else f"{base_url}/posts/?limit={limit}&offset={offset + limit}&tournaments=bot-testing-area",
                "previous": None,
            },
        )

    return _register


# ── Mock helpers for post details ───────────────────────────────────


@pytest.fixture
def mock_post_details(httpx_mock: HTTPXMock, base_url: str, api_post_details_binary: dict):
    """Register a /api/posts/{id}/ response."""

    def _register(post_id: int, details: dict | None = None):
        if details is None:
            details = json.loads(json.dumps(api_post_details_binary))
            details["id"] = post_id
            details["question"]["id"] = post_id
        httpx_mock.add_response(
            url=f"{base_url}/posts/{post_id}/",
            json=details,
        )

    return _register


# ── Mock helpers for forecast submission ────────────────────────────


@pytest.fixture
def mock_forecast_submit(httpx_mock: HTTPXMock, base_url: str):
    """Register a POST /api/questions/forecast/ response."""

    def _register(status_code: int = 201):
        httpx_mock.add_response(
            url=f"{base_url}/questions/forecast/",
            method="POST",
            status_code=status_code,
        )

    return _register


# ── Mock helpers for comment posting ────────────────────────────────


@pytest.fixture
def mock_comment_post(httpx_mock: HTTPXMock, base_url: str):
    """Register a POST /api/comments/create/ response."""

    def _register(status_code: int = 201):
        httpx_mock.add_response(
            url=f"{base_url}/comments/create/",
            method="POST",
            status_code=status_code,
        )

    return _register


# ── LLM mock helpers ────────────────────────────────────────────────


class FakeLLMGateway:
    """Gateway stub for integration tests — no real HTTP calls.

    Inject via ``config.gateway = FakeLLMGateway(complete_fn=..., extract_fn=...)``.
    """

    def __init__(self, complete_fn=None, extract_fn=None):
        from forecasting.extraction import BinaryExtraction

        self._complete_fn = complete_fn or (lambda request: "Rationale\nProbability: 60%")
        self._extract_fn = extract_fn or (
            lambda request, output_type: BinaryExtraction(probability_percent=60)
        )
        self._model = "gpt-5.4-mini"

    def model_for(self, purpose: str) -> str:
        return self._model

    async def complete(self, request: LLMRequest) -> str:
        result = self._complete_fn(request)
        if hasattr(result, "__await__"):
            result = await result
        return result

    async def extract_structured(self, request: LLMRequest, output_type: object) -> object:
        result = self._extract_fn(request, output_type)
        if hasattr(result, "__await__"):
            result = await result
        return result

    async def prepare_cache(self, request: LLMRequest, count: int = 1) -> None:
        return

    async def expire_cache(self, request: LLMRequest) -> None:
        return


@pytest.fixture
def llm_config() -> BotConfig:

    return BotConfig(
        gateway=FakeLLMGateway(),
        disable_research=True,
        forecast_config={"runs": 3},
    )


@pytest.fixture
def patch_forecast_deps():
    """Context manager that provides a FakeLLMGateway with custom callbacks.

    Returns the gateway so callers can inspect calls or override behavior.

        with patch_forecast_deps(
            call_llm=my_call_llm_async_fn,
            extract=my_extract_async_fn,
        ) as gateway:
            ...
    """
    import contextlib

    @contextlib.contextmanager
    def _patcher(call_llm=None, extract=None):
        gateway = FakeLLMGateway(complete_fn=call_llm, extract_fn=extract)
        with mock.patch("forecasting.strategies.LLMRequest", wraps=LLMRequest):
            with mock.patch("forecasting.primitives.LLMRequest", wraps=LLMRequest):
                with mock.patch("research.strategies.LLMRequest", wraps=LLMRequest):
                    yield gateway

    return _patcher


# ── Fixtures that combine real API shapes into Question objects ─────


@pytest.fixture
def binary_question_from_api(api_post_details_binary: dict) -> ForecastQuestion:
    return MetaculusAdapter.question_from_api(api_post_details_binary)


@pytest.fixture
def numeric_question_from_api(api_post_details_numeric: dict) -> ForecastQuestion:
    return MetaculusAdapter.question_from_api(api_post_details_numeric)


@pytest.fixture
def date_question_from_api(api_post_details_date: dict) -> ForecastQuestion:
    return MetaculusAdapter.question_from_api(api_post_details_date)


@pytest.fixture
def mc_question_from_api(api_post_details_mc: dict) -> ForecastQuestion:
    return MetaculusAdapter.question_from_api(api_post_details_mc)


# ── AskNews / Exa SDK mocks ─────────────────────────────────────────


@pytest.fixture
def mock_asknews_search():
    """Returns function that returns mocked AskNews search results."""
    with mock.patch("research.asknews.AsyncAskNewsSDK") as mock_sdk:
        instance = mock_sdk.return_value
        instance.news.search_news = mock.AsyncMock()

        class FakeArticle:
            def __init__(self, article_url, pub_date, eng_title, summary, language, source_id):
                self.article_url = article_url
                self.pub_date = pub_date
                self.eng_title = eng_title
                self.summary = summary
                self.language = language
                self.source_id = source_id

        instance.news.search_news.return_value.as_dicts = [
            FakeArticle(
                "https://example.com/1",
                datetime.datetime(2025, 1, 15),
                "Test Article 1",
                "Summary of article 1.",
                "en",
                "test-source-1",
            ),
            FakeArticle(
                "https://example.com/2",
                datetime.datetime(2025, 1, 16),
                "Test Article 2",
                "Summary of article 2.",
                "en",
                "test-source-2",
            ),
        ]
        yield mock_sdk


@pytest.fixture
def mock_exa_search():
    """Returns function that returns mocked Exa search results with realistic scores."""
    with mock.patch("research.exa.AsyncExa") as mock_sdk:
        instance = mock_sdk.return_value

        class FakeResult:
            def __init__(self, title, url, highlights, score):
                self.title = title
                self.url = url
                self.highlights = highlights
                self.score = score

        instance.search = mock.AsyncMock()
        instance.search.return_value.results = [
            FakeResult("Result 1", "https://example.com/a", ["Highlight A1", "Highlight A2"], 0.95),
            FakeResult("Result 2", "https://example.com/b", ["Highlight B1"], 0.82),
            FakeResult("Result 3", "https://example.com/b", ["Highlight B1"], 0.80),
        ]
        yield mock_sdk


@pytest.fixture
def mock_exa_search_with_no_scores():
    """Returns function that returns mocked Exa search results with scores=None."""
    with mock.patch("research.exa.AsyncExa") as mock_sdk:
        instance = mock_sdk.return_value

        class FakeResult:
            def __init__(self, title, url, highlights, score):
                self.title = title
                self.url = url
                self.highlights = highlights
                self.score = score

        instance.search = mock.AsyncMock()
        instance.search.return_value.results = [
            FakeResult("No-score Result", "https://example.com/x", ["Highlight X1"], None),
            FakeResult("No-score Result 2", "https://example.com/y", ["Highlight Y1"], None),
        ]
        yield mock_sdk
