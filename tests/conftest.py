"""Shared pytest fixtures for all test layers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import BotConfig
from common.models import ForecastQuestion, NumericScaling, QuestionType
from llm.gateway import LLMRequest


class FakeGateway:
    """Minimal in-process gateway stub — satisfies the Gateway Protocol.

    Unit and integration tests can inject custom ``complete_fn`` /
    ``extract_fn`` callbacks or use the defaults.
    """

    def __init__(self, complete_fn=None, extract_fn=None, model: str = "test-model"):
        self._complete_fn = complete_fn or (lambda request: "Rationale\nProbability: 60%")
        self._extract_fn = extract_fn
        self._model = model

    def model_for(self, purpose: str) -> str:
        return self._model

    async def complete(self, request: LLMRequest) -> str:
        result = self._complete_fn(request)
        if hasattr(result, "__await__"):
            result = await result
        return result

    async def extract_structured(self, request: LLMRequest, output_type: Any) -> Any:
        if self._extract_fn is None:
            return output_type() if callable(output_type) else output_type
        result = self._extract_fn(request, output_type)
        if hasattr(result, "__await__"):
            result = await result
        return result

    async def prepare_cache(self, request: LLMRequest, count: int = 1) -> None:
        return

    async def expire_cache(self, request: LLMRequest) -> None:
        return


@pytest.fixture
def make_bot_config() -> Any:
    """Factory fixture: ``make_bot_config(gateway, **overrides) -> BotConfig``."""

    def _make(**overrides: Any) -> BotConfig:
        defaults: dict[str, Any] = {
            "gateway": FakeGateway(),
            "live": False,
            "submit": False,
            "limit": 0,
            "tournament_id": "",
            "reforecast": False,
            "disable_research": False,
            "strategy": "ensemble",
            "forecast_config": {},
            "research_config": {},
            "run_id": None,
            "code_version": None,
            "run_cfg": None,
        }
        defaults.update(overrides)
        return BotConfig(**defaults)

    return _make


@pytest.fixture
def linear_scaling() -> NumericScaling:
    return NumericScaling(
        range_min=0.0,
        range_max=100.0,
        zero_point=None,
        open_lower_bound=False,
        open_upper_bound=False,
        cdf_size=201,
    )


@pytest.fixture
def log_scaling() -> NumericScaling:
    return NumericScaling(
        range_min=10.0,
        range_max=100000.0,
        zero_point=1.0,
        open_lower_bound=False,
        open_upper_bound=False,
        cdf_size=201,
    )


@pytest.fixture
def open_bounds_scaling() -> NumericScaling:
    return NumericScaling(
        range_min=0.0,
        range_max=100.0,
        zero_point=None,
        open_lower_bound=True,
        open_upper_bound=True,
        cdf_size=201,
    )


@pytest.fixture
def sample_binary_question() -> ForecastQuestion:
    return ForecastQuestion(
        title="Will humans go extinct before 2100?",
        body="Test description.\n\nResolution criteria.\n\nFine print.",
        question_type=QuestionType.BINARY,
        resolve_time="2100-01-02T00:01:00Z",
        url="https://www.metaculus.com/questions/578/",
    )


@pytest.fixture
def sample_numeric_question(linear_scaling: NumericScaling) -> ForecastQuestion:
    return ForecastQuestion(
        title="What will be the global average temperature rise by 2050?",
        body="Test description for a numeric question.",
        question_type=QuestionType.NUMERIC,
        scaling=linear_scaling,
        unit="°C",
        resolve_time="2050-01-01T00:00:00Z",
        url="https://www.metaculus.com/questions/1234/",
    )


@pytest.fixture
def sample_log_numeric_question(log_scaling: NumericScaling) -> ForecastQuestion:
    return ForecastQuestion(
        title="What will be the total cost of project X?",
        body="Log-scaled question description.",
        question_type=QuestionType.NUMERIC,
        scaling=log_scaling,
        unit="USD",
        resolve_time="2030-01-01T00:00:00Z",
        url="https://www.metaculus.com/questions/5678/",
    )


@pytest.fixture
def sample_date_question() -> ForecastQuestion:
    scaling = NumericScaling(
        range_min=1700000000.0,
        range_max=2000000000.0,
        zero_point=None,
        open_lower_bound=False,
        open_upper_bound=False,
        cdf_size=201,
    )
    return ForecastQuestion(
        title="When will the first human land on Mars?",
        body="Test description for a date question.",
        question_type=QuestionType.DATE,
        scaling=scaling,
        resolve_time="2050-01-01T00:00:00Z",
        url="https://www.metaculus.com/questions/9999/",
    )


@pytest.fixture
def sample_multiple_choice_question() -> ForecastQuestion:
    return ForecastQuestion(
        title="Who will win the 2028 US presidential election?",
        body="Test description for a multiple choice question.",
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=["Democrat", "Republican", "Other"],
        resolve_time="2028-11-07T00:00:00Z",
        url="https://www.metaculus.com/questions/4321/",
    )
