"""Unit tests for llm/gateway.py — routing, retry, rate limiting."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from llm.gateway import GatewayConfig, LLMGateway, LLMRequest, ModelConfig, TransientLLMError


class FakeProvider:
    """Mock provider for gateway unit tests."""

    def __init__(self, complete_fn=None, extract_fn=None) -> None:
        self._complete_fn = complete_fn
        self._extract_fn = extract_fn
        self.complete_calls: list[tuple[str, LLMRequest]] = []
        self.extract_calls: list[tuple[str, LLMRequest, Any]] = []
        self.prepare_cache_calls: list[tuple[str, LLMRequest, int]] = []
        self.expire_cache_calls: list = []

    async def complete(self, model: str, request: LLMRequest) -> str:
        self.complete_calls.append((model, request))
        if self._complete_fn:
            return await self._complete_fn(model, request)
        return "default response"

    async def extract(self, model: str, request: LLMRequest, output_type: Any) -> Any:
        self.extract_calls.append((model, request, output_type))
        if self._extract_fn:
            return await self._extract_fn(model, request, output_type)
        return output_type

    async def prepare_cache(self, model: str, request: LLMRequest, count: int = 1) -> None:
        self.prepare_cache_calls.append((model, request, count))

    async def expire_cache(self, model: str, request: LLMRequest) -> None:
        self.expire_cache_calls.append((model, request))


@pytest.fixture
def mock_provider() -> FakeProvider:
    return FakeProvider()


# ── LLMRequest ────────────────────────────────────────────────────────


class TestLLMRequest:
    def test_defaults(self):
        req = LLMRequest(prompt="hello", purpose="test")
        assert req.prompt == "hello"
        assert req.purpose == "test"
        assert req.system_prompt is None
        assert req.temperature == 0.3
        assert req.reasoning_effort is None
        assert req.max_tokens is None

    def test_requires_purpose(self):
        req = LLMRequest(prompt="hello", purpose="forecast")
        assert req.purpose == "forecast"

    def test_immutable(self):
        req = LLMRequest(prompt="hello", purpose="test")
        with pytest.raises(FrozenInstanceError):
            req.prompt = "changed"  # type: ignore[misc]


# ── ModelConfig ───────────────────────────────────────────────────────


class TestModelConfig:
    def test_defaults(self):
        mc = ModelConfig(model="gpt-4")
        assert mc.model == "gpt-4"
        assert mc.provider == "openai"

    def test_immutable(self):
        mc = ModelConfig(model="gpt-4")
        with pytest.raises(FrozenInstanceError):
            mc.model = "changed"  # type: ignore[misc]


# ── GatewayConfig validation ──────────────────────────────────────────


class TestGatewayConfigPostInit:
    def test_valid_config_passes(self, mock_provider):
        cfg = GatewayConfig(
            providers={"mock": mock_provider},
            models={"test": ModelConfig(model="m", provider="mock")},
        )
        assert cfg.max_concurrent == 5
        assert cfg.max_retries == 3

    def test_missing_provider_raises(self, mock_provider):
        with pytest.raises(ValueError, match="references provider 'missing' not in registry"):
            GatewayConfig(
                providers={"mock": mock_provider},
                models={"test": ModelConfig(model="m", provider="missing")},
            )

    def test_custom_concurrency_and_retries(self, mock_provider):
        cfg = GatewayConfig(
            providers={"mock": mock_provider},
            models={"test": ModelConfig(model="m", provider="mock")},
            max_concurrent=10,
            max_retries=5,
        )
        assert cfg.max_concurrent == 10
        assert cfg.max_retries == 5


# ── Gateway routing ───────────────────────────────────────────────────


class TestGatewayRouting:
    async def test_routes_to_correct_provider(self, mock_provider, sample_request):
        cfg = GatewayConfig(
            providers={"mock": mock_provider},
            models={"test": ModelConfig(model="test-model", provider="mock")},
        )
        gateway = LLMGateway(cfg)
        await gateway.complete(sample_request)
        assert len(mock_provider.complete_calls) == 1
        call_model, call_request = mock_provider.complete_calls[0]
        assert call_model == "test-model"
        assert call_request.purpose == "test"

    async def test_routes_different_purposes_to_different_models(
        self, mock_provider, sample_request
    ):
        cfg = GatewayConfig(
            providers={"mock": mock_provider},
            models={
                "extract": ModelConfig(model="extract-model", provider="mock"),
                "forecast": ModelConfig(model="forecast-model", provider="mock"),
            },
        )
        gateway = LLMGateway(cfg)
        extract_req = LLMRequest(prompt="prob: 50%", purpose="extract")
        await gateway.extract_structured(extract_req, str)
        forecast_req = LLMRequest(prompt="predict", purpose="forecast")
        await gateway.complete(forecast_req)
        assert mock_provider.extract_calls[0][0] == "extract-model"
        assert mock_provider.complete_calls[0][0] == "forecast-model"


# ── Gateway complete ──────────────────────────────────────────────────


class TestGatewayComplete:
    async def test_delegates_to_provider(self, mock_provider, sample_request):
        cfg = GatewayConfig(
            providers={"mock": mock_provider},
            models={"test": ModelConfig(model="m", provider="mock")},
        )
        gateway = LLMGateway(cfg)
        result = await gateway.complete(sample_request)
        assert result == "default response"

    async def test_passes_system_prompt(self, mock_provider, sample_request_with_system):
        cfg = GatewayConfig(
            providers={"mock": mock_provider},
            models={"test": ModelConfig(model="m", provider="mock")},
        )
        gateway = LLMGateway(cfg)
        await gateway.complete(sample_request_with_system)
        assert len(mock_provider.complete_calls) == 1
        _, call_request = mock_provider.complete_calls[0]
        assert call_request.system_prompt == "You are a math tutor."

    async def test_passes_temperature(self, mock_provider, sample_request):
        cfg = GatewayConfig(
            providers={"mock": mock_provider},
            models={"test": ModelConfig(model="m", provider="mock")},
        )
        gateway = LLMGateway(cfg)
        await gateway.complete(sample_request)
        _, call_request = mock_provider.complete_calls[0]
        assert call_request.temperature == 0.3


# ── Gateway retry ─────────────────────────────────────────────────────


class TestGatewayRetry:
    async def test_retries_on_transient_error(self, mock_provider, sample_request):
        call_count = 0

        async def flaky_complete(model, request):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TransientLLMError("rate limited")
            return "success"

        mock_provider._complete_fn = flaky_complete
        cfg = GatewayConfig(
            providers={"mock": mock_provider},
            models={"test": ModelConfig(model="m", provider="mock")},
            max_retries=3,
        )
        gateway = LLMGateway(cfg)
        result = await gateway.complete(sample_request)
        assert result == "success"
        assert call_count == 3

    async def test_all_attempts_exhausted_raises(self, mock_provider, sample_request):
        async def always_fail(model, request):
            raise TransientLLMError("rate limited")

        mock_provider._complete_fn = always_fail
        cfg = GatewayConfig(
            providers={"mock": mock_provider},
            models={"test": ModelConfig(model="m", provider="mock")},
            max_retries=2,
        )
        gateway = LLMGateway(cfg)
        with pytest.raises(TransientLLMError):
            await gateway.complete(sample_request)

    async def test_non_transient_error_propagates(self, mock_provider, sample_request):
        async def raise_value(model, request):
            raise ValueError("not transient")

        mock_provider._complete_fn = raise_value
        cfg = GatewayConfig(
            providers={"mock": mock_provider},
            models={"test": ModelConfig(model="m", provider="mock")},
        )
        gateway = LLMGateway(cfg)
        with pytest.raises(ValueError, match="not transient"):
            await gateway.complete(sample_request)

    async def test_extract_retries_on_transient_error(self, mock_provider, sample_extract_request):
        call_count = 0

        async def flaky_extract(model, request, output_type):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TransientLLMError("rate limited")
            return output_type

        mock_provider._extract_fn = flaky_extract
        cfg = GatewayConfig(
            providers={"mock": mock_provider},
            models={"extract": ModelConfig(model="m", provider="mock")},
            max_retries=2,
        )
        gateway = LLMGateway(cfg)
        result = await gateway.extract_structured(sample_extract_request, str)
        assert result is str
        assert call_count == 2


# ── Gateway extract ───────────────────────────────────────────────────


class TestGatewayExtract:
    async def test_delegates_extraction(self, mock_provider, sample_extract_request):
        cfg = GatewayConfig(
            providers={"mock": mock_provider},
            models={"extract": ModelConfig(model="extract-model", provider="mock")},
        )
        gateway = LLMGateway(cfg)
        await gateway.extract_structured(sample_extract_request, str)
        assert len(mock_provider.extract_calls) == 1
        model, request, output_type = mock_provider.extract_calls[0]
        assert model == "extract-model"
        assert request.purpose == "extract"

    async def test_passes_output_type(self, mock_provider):
        class FakeModel:
            pass

        cfg = GatewayConfig(
            providers={"mock": mock_provider},
            models={"extract": ModelConfig(model="m", provider="mock")},
        )
        gateway = LLMGateway(cfg)
        req = LLMRequest(prompt="test", purpose="extract")
        await gateway.extract_structured(req, FakeModel)
        _, _, output_type = mock_provider.extract_calls[0]
        assert output_type is FakeModel


# ── Gateway rate limiting ─────────────────────────────────────────────


class TestGatewayRateLimiting:
    async def test_semaphore_bounds_concurrency(self):
        max_concurrent = 0
        in_flight = 0
        lock = asyncio.Lock()
        provider = FakeProvider()

        async def delayed_complete(model, request):
            nonlocal max_concurrent, in_flight
            async with lock:
                in_flight += 1
                max_concurrent = max(max_concurrent, in_flight)
            await asyncio.sleep(0.01)
            async with lock:
                in_flight -= 1
            return "ok"

        provider._complete_fn = delayed_complete
        cfg = GatewayConfig(
            providers={"mock": provider},
            models={"test": ModelConfig(model="m", provider="mock")},
            max_concurrent=3,
        )
        gateway = LLMGateway(cfg)
        tasks = [gateway.complete(LLMRequest(prompt=f"p-{i}", purpose="test")) for i in range(10)]
        await asyncio.gather(*tasks)
        assert max_concurrent <= 3


# ── Gateway cache ──────────────────────────────────────────────────────


class TestGatewayCache:
    async def test_prepare_cache_delegates_to_correct_provider(self, mock_provider, sample_request):
        cfg = GatewayConfig(
            providers={"mock": mock_provider},
            models={"test": ModelConfig(model="test-model", provider="mock")},
        )
        gateway = LLMGateway(cfg)
        await gateway.prepare_cache(sample_request)
        assert len(mock_provider.prepare_cache_calls) == 1
        call_model, call_request, call_count = mock_provider.prepare_cache_calls[0]
        assert call_model == "test-model"
        assert call_request.purpose == "test"

    async def test_expire_cache_delegates_to_correct_provider(self, mock_provider, sample_request):
        cfg = GatewayConfig(
            providers={"mock": mock_provider},
            models={"test": ModelConfig(model="test-model", provider="mock")},
        )
        gateway = LLMGateway(cfg)
        await gateway.expire_cache(sample_request)
        assert len(mock_provider.expire_cache_calls) == 1

    async def test_routes_cache_to_different_providers(self, mock_provider, sample_request):
        provider2 = FakeProvider()
        cfg = GatewayConfig(
            providers={"mock": mock_provider, "other": provider2},
            models={
                "test": ModelConfig(model="m", provider="mock"),
                "extract": ModelConfig(model="e", provider="other"),
            },
        )
        gateway = LLMGateway(cfg)
        await gateway.prepare_cache(LLMRequest(prompt="p", purpose="test"))
        await gateway.prepare_cache(LLMRequest(prompt="p", purpose="extract"))
        assert len(mock_provider.prepare_cache_calls) == 1
        assert len(provider2.prepare_cache_calls) == 1
