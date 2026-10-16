"""LLM Gateway — routing, retry, rate limiting.

Moves the *what* from the caller to a value object (LLMRequest) and owns the
*how* — model lookup, provider dispatch, retry policy, concurrency control.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from llm.providers import Provider


@runtime_checkable
class Gateway(Protocol):
    """Structural interface for LLM gateways — real and test fakes alike.

    Mirrors the ``Searcher`` Protocol pattern from ``research/searcher.py``.
    """

    async def complete(self, request: LLMRequest) -> str: ...
    async def extract_structured(self, request: LLMRequest, output_type: Any) -> Any: ...
    async def prepare_cache(self, request: LLMRequest, count: int = 1) -> None: ...
    async def expire_cache(self, request: LLMRequest) -> None: ...
    def model_for(self, purpose: str) -> str: ...


class TransientLLMError(Exception):
    """Provider-agnostic transient error — wraps native provider exceptions.

    Each provider catches its own transient errors (rate limits, timeouts,
    connection failures, server errors) and re-raises them as this type.
    The gateway retries on TransientLLMError without knowing which backend
    threw it.
    """


TRANSIENT_ERRORS = (TransientLLMError,)


@dataclass(frozen=True)
class LLMRequest:
    """Neutral intent — no provider/model details. The gateway decides how.

    Every field is provider-agnostic.  Each provider maps this to its own dialect:
    ``reasoning_effort`` → OpenAI ``reasoning_effort``, Anthropic ``thinking`` budget,
    Gemini ``thinkingConfig``.  A model that doesn't support a field ignores it.
    """

    prompt: str
    purpose: str
    system_prompt: str | None = None
    temperature: float | None = 0.3
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    max_tokens: int | None = None


@dataclass(frozen=True)
class ModelConfig:
    """The model to use for a purpose.  ``provider`` matches a key in
    ``GatewayConfig.providers``."""

    model: str
    provider: str = "openai"


@dataclass
class GatewayConfig:
    """Wires providers to purposes.  One model per purpose — no fallback chains yet."""

    providers: dict[str, Provider]
    models: dict[str, ModelConfig]
    max_concurrent: int = 5
    max_retries: int = 3

    def __post_init__(self) -> None:
        for purpose, mc in self.models.items():
            if mc.provider not in self.providers:
                raise ValueError(
                    f"Model '{mc.model}' (purpose='{purpose}') "
                    f"references provider '{mc.provider}' not in registry"
                )


class LLMGateway:
    """Completion and structured extraction with same-model retry on transient errors."""

    def __init__(self, config: GatewayConfig):
        self._config = config
        self._rate_limiter = asyncio.Semaphore(config.max_concurrent)

    def model_for(self, purpose: str) -> str:
        """The model name configured for *purpose* — e.g. replay pinning checks."""
        return self._config.models[purpose].model

    async def prepare_cache(self, request: LLMRequest, count: int = 1) -> None:
        """One-time cache setup before a batch of ``count`` identical-prefix
        complete/extract calls on the same purpose."""
        mc = self._config.models[request.purpose]
        provider = self._config.providers[mc.provider]
        await provider.prepare_cache(mc.model, request, count)

    async def expire_cache(self, request: LLMRequest) -> None:
        """Release cache resources for the provider behind ``request.purpose``."""
        mc = self._config.models[request.purpose]
        provider = self._config.providers[mc.provider]
        await provider.expire_cache(mc.model, request)

    async def complete(self, request: LLMRequest) -> str:
        """Look up the model for ``request.purpose``, dispatch to its provider.
        Retry the *same* model up to ``max_retries`` times on transient errors.
        All attempts exhausted → raise."""
        mc = self._config.models[request.purpose]
        provider = self._config.providers[mc.provider]

        last_error: Exception | None = None
        for _attempt in range(1, self._config.max_retries + 1):
            async with self._rate_limiter:
                try:
                    return await provider.complete(mc.model, request)
                except TRANSIENT_ERRORS as e:
                    last_error = e
                    continue

        raise last_error or RuntimeError(f"No response after {self._config.max_retries} attempts")

    async def extract_structured(self, request: LLMRequest, output_type: Any) -> Any:
        """Symmetric with ``complete`` — takes a request, routes per-purpose,
        delegates to the provider's extraction mechanism."""
        mc = self._config.models[request.purpose]
        provider = self._config.providers[mc.provider]

        last_error: Exception | None = None
        for _attempt in range(1, self._config.max_retries + 1):
            try:
                return await provider.extract(mc.model, request, output_type)
            except TRANSIENT_ERRORS as e:
                last_error = e
                continue

        raise last_error or RuntimeError(f"No response after {self._config.max_retries} attempts")
