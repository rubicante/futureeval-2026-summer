"""Provider protocol and implementations — one per API backend.

Each provider is constructed with its own credentials and client.  It owns the
neutral→wire translation, all per-model quirks, and the structured-output mechanism.
The gateway never sees wire format.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, NoReturn, Protocol, runtime_checkable

import instructor
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

from llm.costs import LEDGER
from llm.gateway import LLMRequest, TransientLLMError

logger = logging.getLogger(__name__)

OPENAI_TRANSIENT = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


@runtime_checkable
class Provider(Protocol):
    """Executes an LLMRequest against one backend.

    Each provider is constructed with its own credentials, client, and instructor
    wrapper.  It owns the neutral→wire translation, all per-model quirks, and
    structured-output mechanism.  The gateway never sees wire format.
    """

    async def complete(self, model: str, request: LLMRequest) -> str: ...

    async def extract(self, model: str, request: LLMRequest, output_type: Any) -> Any: ...

    async def prepare_cache(self, model: str, request: LLMRequest, count: int = 1) -> None:
        """One-time setup so subsequent complete/extract calls reuse cached tokens.

        Called once before a batch of ``count`` identical-prefix requests.
        Priming costs roughly one uncached prompt, so providers skip it when
        ``count`` < 2 — a lone request is cheaper sent directly.
        Default no-op — providers that support caching override this.
        """
        return

    async def expire_cache(self, model: str, request: LLMRequest) -> None:
        """Release cache resources for *request*'s prefix.  No-op for backends
        without paid storage."""
        return


def _cache_key(model: str, request: LLMRequest) -> tuple[str, str | None, str]:
    """Identity of a cached prefix.  Questions forecast concurrently share one
    provider instance, so cache state must be keyed per prefix — a single slot
    lets one question's prepare/expire destroy another's in-flight cache."""
    return (model, request.system_prompt, request.prompt)


# ═══════════════════════════════════════════════════════════════════════
#  OpenAIProvider
# ═══════════════════════════════════════════════════════════════════════


class OpenAIProvider:
    """OpenAI SDK — also handles OpenRouter, Perplexity, and any OpenAI-compatible API.

    Constructed with api_key / base_url.  Owns the AsyncOpenAI client and
    instructor wrapper.  Provider-specific quirks (Perplexity prefix skip,
    model temperature support) live here, not in the gateway.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature_unsupported_models: set[str] | None = None,
    ) -> None:
        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**client_kwargs)
        self._instructor = instructor.from_openai(self._client)
        self._temperature_unsupported = (
            temperature_unsupported_models.copy() if temperature_unsupported_models else set()
        )

    async def complete(self, model: str, request: LLMRequest) -> str:
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        if model.startswith("perplexity/"):
            return await self._complete_openai_compatible(model, messages, request)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        self._apply_temperature(kwargs, model, request)
        self._apply_reasoning(kwargs, request)
        self._apply_max_tokens(kwargs, request)

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except OPENAI_TRANSIENT as e:
            raise TransientLLMError(str(e)) from e
        answer = response.choices[0].message.content
        if answer is None:
            raise ValueError(f"No answer returned from LLM (model={model})")
        self._log_usage(response.usage, model, request.purpose)
        return answer

    async def extract(self, model: str, request: LLMRequest, output_type: Any) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "response_model": output_type,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        self._apply_temperature(kwargs, model, request)
        self._apply_reasoning(kwargs, request)
        self._apply_max_tokens(kwargs, request)

        try:
            parsed, raw = await self._instructor.chat.completions.create_with_completion(**kwargs)
        except OPENAI_TRANSIENT as e:
            raise TransientLLMError(str(e)) from e
        self._log_usage(raw.usage, model, request.purpose)
        return parsed

    async def _complete_openai_compatible(
        self, model: str, messages: list[dict[str, str]], request: LLMRequest
    ) -> str:
        """Perplexity and other OpenAI-compatible backends — skip prefix, same shape."""
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        self._apply_temperature(kwargs, model, request)
        self._apply_reasoning(kwargs, request)
        self._apply_max_tokens(kwargs, request)

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except OPENAI_TRANSIENT as e:
            raise TransientLLMError(str(e)) from e
        answer = response.choices[0].message.content
        if answer is None:
            raise ValueError(f"No answer returned from LLM (model={model})")
        self._log_usage(response.usage, model, request.purpose)
        return answer

    def _apply_temperature(self, kwargs: dict[str, Any], model: str, request: LLMRequest) -> None:
        if request.temperature is not None and model not in self._temperature_unsupported:
            kwargs["temperature"] = request.temperature

    def _apply_reasoning(self, kwargs: dict[str, Any], request: LLMRequest) -> None:
        if request.reasoning_effort is not None:
            kwargs["reasoning_effort"] = request.reasoning_effort

    def _apply_max_tokens(self, kwargs: dict[str, Any], request: LLMRequest) -> None:
        if request.max_tokens is not None:
            kwargs["max_completion_tokens"] = request.max_tokens

    async def prepare_cache(self, model: str, request: LLMRequest, count: int = 1) -> None:
        """Seed the server-side KV cache with a completion.

        OpenAI prompt caching is automatic — the first request with a given
        prefix populates a server-side KV cache.  This sends a request so
        subsequent parallel completions hit the cache.
        """
        if count < 2:
            return
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        self._apply_temperature(kwargs, model, request)
        self._apply_reasoning(kwargs, request)
        # Use a small token limit for cheap cache priming.
        # GPT-5.x models need enough budget for internal reasoning tokens;
        # 100 is sufficient while keeping the priming request nearly cost-free.
        cache_req = LLMRequest(prompt="", purpose=request.purpose, max_tokens=100)
        self._apply_max_tokens(kwargs, cache_req)

        try:
            await self._client.chat.completions.create(**kwargs)
        except OPENAI_TRANSIENT:
            pass  # expected — cache warm-up competes for rate-limit budget
        except Exception as e:
            logger.warning("prepare_cache request rejected: %s", e, exc_info=False)

    async def expire_cache(self, model: str, request: LLMRequest) -> None:
        return

    def _log_usage(self, usage: Any, model: str, purpose: str) -> None:
        if usage is None:
            return
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) if details else 0
        logger.debug(f"cached_tokens={cached}/{usage.prompt_tokens} ({model})")
        LEDGER.record(
            purpose=purpose,
            model=model,
            prompt_tokens=usage.prompt_tokens or 0,
            cached_tokens=cached or 0,
            completion_tokens=usage.completion_tokens or 0,
        )


# ═══════════════════════════════════════════════════════════════════════
#  AnthropicProvider
# ═══════════════════════════════════════════════════════════════════════


class AnthropicProvider:
    """Anthropic native SDK.  Proves the protocol survives a backend that differs on
    3 axes at once: message format (top-level ``system``, not a role), reasoning
    params (``thinking`` budget), and structured output (tool-forcing).
    """

    def __init__(
        self, api_key: str | None = None, temperature_unsupported_models: set[str] | None = None
    ) -> None:
        try:
            from anthropic import AsyncAnthropic  # noqa: I001, PLC0415
            from anthropic import (  # noqa: I001, PLC0415
                APIConnectionError as AnthropicAPIConnectionError,
                APITimeoutError as AnthropicAPITimeoutError,
                InternalServerError as AnthropicInternalServerError,
                RateLimitError as AnthropicRateLimitError,
            )
        except ImportError as err:
            raise ImportError(
                "AnthropicProvider requires the 'anthropic' package. "
                "Install it with: uv add anthropic"
            ) from err
        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        self._client = AsyncAnthropic(**client_kwargs)
        self._instructor = instructor.from_anthropic(self._client)
        self._temperature_unsupported = (
            temperature_unsupported_models.copy() if temperature_unsupported_models else set()
        )
        self._transient_errors: tuple[type[Exception], ...] = (
            AnthropicRateLimitError,
            AnthropicAPITimeoutError,
            AnthropicAPIConnectionError,
            AnthropicInternalServerError,
        )
        self._primed: set[tuple[str, str | None, str]] = set()

    async def prepare_cache(self, model: str, request: LLMRequest, count: int = 1) -> None:
        """Write the prompt to Anthropic's server-side cache with a 1-token request.

        The cache entry only becomes readable after the writing request has
        been processed, so parallel identical requests all pay the 1.25×
        write premium — priming first is what lets the batch read at 0.1×.
        Prime-then-read costs 1.25 + 0.1n of the uncached prompt price, so a
        single request (n=1) is cheaper sent uncached.
        """
        # Anthropic prompt caching is not available on Haiku-class models.
        if count < 2 or "haiku" in model.lower():
            return

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": 1,
            "cache_control": {"type": "ephemeral"},
        }
        if request.system_prompt:
            kwargs["system"] = request.system_prompt
        self._apply_temperature(kwargs, model, request)
        # No thinking config: changing thinking params invalidates message
        # cache breakpoints, and forecast requests don't set reasoning_effort.
        # If they ever do, the primer must mirror it (with max_tokens above
        # the thinking budget) or the samples will miss the cache.
        try:
            response = await self._client.messages.create(**kwargs)
        except self._transient_errors:
            return  # samples still work; the first one writes the cache instead
        except Exception as e:
            logger.warning("prepare_cache request rejected: %s", e, exc_info=False)
            return
        self._log_usage(response, model, request.purpose)
        self._primed.add(_cache_key(model, request))

    async def expire_cache(self, model: str, request: LLMRequest) -> None:
        # Nothing to release server-side — entries expire on their own TTL.
        self._primed.discard(_cache_key(model, request))

    async def complete(self, model: str, request: LLMRequest) -> str:
        messages: list[dict[str, Any]] = []
        if request.prompt:
            messages.append({"role": "user", "content": request.prompt})
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
        }
        if request.system_prompt:
            kwargs["system"] = request.system_prompt
        # cache_control marks the prefix for matching — without it a request
        # never reads the cache, even when the primed entry exists.
        if _cache_key(model, request) in self._primed:
            kwargs["cache_control"] = {"type": "ephemeral"}
        self._apply_temperature(kwargs, model, request)
        self._apply_reasoning(kwargs, model, request)

        try:
            response = await self._client.messages.create(**kwargs)
        except self._transient_errors as e:
            raise TransientLLMError(str(e)) from e
        self._log_usage(response, model, request.purpose)
        for block in response.content:
            if block.type == "text":
                return block.text
        raise ValueError(f"No text content in Anthropic response (model={model})")

    async def extract(self, model: str, request: LLMRequest, output_type: Any) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "response_model": output_type,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens or 4096,
        }
        if request.system_prompt:
            kwargs["system"] = request.system_prompt
        if _cache_key(model, request) in self._primed:
            kwargs["cache_control"] = {"type": "ephemeral"}
        self._apply_temperature(kwargs, model, request)
        self._apply_reasoning(kwargs, model, request)

        try:
            parsed, raw = await self._instructor.messages.create_with_completion(**kwargs)
        except self._transient_errors as e:
            raise TransientLLMError(str(e)) from e
        self._log_usage(raw, model, request.purpose)
        return parsed

    def _apply_temperature(self, kwargs: dict[str, Any], model: str, request: LLMRequest) -> None:
        if request.temperature is not None and model not in self._temperature_unsupported:
            kwargs["temperature"] = request.temperature

    def _apply_reasoning(self, kwargs: dict[str, Any], model: str, request: LLMRequest) -> None:
        if request.reasoning_effort is not None:
            effort_to_tokens = {"low": 1024, "medium": 4096, "high": 16384}
            budget = effort_to_tokens.get(request.reasoning_effort, 4096)
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}

    @staticmethod
    def _log_usage(response: Any, model: str, purpose: str) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        # Anthropic's input_tokens EXCLUDES cache tokens (OpenAI/Gemini
        # include them); the ledger formula assumes the inclusive convention,
        # so translate here.  Cache writes bill at 1.25× the input rate —
        # unmodeled by the price table, pennies at our volume.
        direct = getattr(usage, "input_tokens", 0) or 0
        cached = getattr(usage, "cache_read_input_tokens", 0) or 0
        written = getattr(usage, "cache_creation_input_tokens", 0) or 0
        prompt = direct + cached + written
        completion = getattr(usage, "output_tokens", 0) or 0
        logger.debug(
            "input_tokens=%s output_tokens=%s cache_read=%s cache_creation=%s (%s)",
            direct,
            completion,
            cached,
            written,
            model,
        )
        LEDGER.record(
            purpose=purpose,
            model=model,
            prompt_tokens=prompt,
            cached_tokens=cached,
            completion_tokens=completion,
        )


# ═══════════════════════════════════════════════════════════════════════
#  GeminiProvider
# ═══════════════════════════════════════════════════════════════════════


class GeminiProvider:
    """Google Gemini native SDK — also supports Vertex AI.

    Proves the protocol survives a backend that differs on SDK library
    (``google.genai``), message format (``contents``/``parts`` + config
    ``system_instruction``), and auth (API key vs. ADC/service-account for
    Vertex AI, no ``base_url`` concept).

    Construct with ``api_key`` for Gemini API or ``vertexai=True`` with
    ``project``/``location`` for Vertex AI.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        vertexai: bool = False,
        project: str | None = None,
        location: str | None = None,
    ) -> None:
        try:
            from google import genai as google_genai  # noqa: I001, PLC0415
        except ImportError as err:
            raise ImportError(
                "GeminiProvider requires the 'google-genai' package. "
                "Install it with: uv add google-genai"
            ) from err

        if vertexai:
            self._client = google_genai.Client(vertexai=True, project=project, location=location)
        else:
            self._client = google_genai.Client(api_key=api_key)
        self._instructor = instructor.from_genai(self._client)
        self._caches: dict[tuple[str, str | None, str], str] = {}

    async def prepare_cache(self, model: str, request: LLMRequest, count: int = 1) -> None:
        from google.genai import types as genai_types  # noqa: I001, PLC0415

        # Explicit-cache write + one discounted read costs more than one
        # uncached request — only worth it for a batch.
        if count < 2:
            return

        contents = [
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=request.prompt)],
            )
        ]
        config = genai_types.CreateCachedContentConfig(
            contents=contents,
            display_name=f"scuttlebot/{request.purpose}",
            ttl="3600s",
        )
        if request.system_prompt:
            config.system_instruction = genai_types.Content(
                role="system",
                parts=[genai_types.Part(text=request.system_prompt)],
            )

        key = _cache_key(model, request)
        stale = self._caches.pop(key, None)
        if stale is not None:
            with contextlib.suppress(Exception):
                self._client.caches.delete(name=stale)
        try:
            cache = self._client.caches.create(model=model, config=config)
        except Exception as exc:
            # Prompts under Gemini's 1024-token cache minimum (e.g. no
            # research context) raise INVALID_ARGUMENT — run uncached.
            logger.info("Gemini cache skipped: %s", exc)
            return
        if cache.name:
            self._caches[key] = cache.name

    async def expire_cache(self, model: str, request: LLMRequest) -> None:
        cache_name = self._caches.pop(_cache_key(model, request), None)
        if cache_name is not None:
            with contextlib.suppress(Exception):
                self._client.caches.delete(name=cache_name)

    async def complete(self, model: str, request: LLMRequest) -> str:
        from google.genai import types as genai_types  # noqa: I001, PLC0415

        config = genai_types.GenerateContentConfig()
        if request.system_prompt:
            config.system_instruction = request.system_prompt
        if request.temperature is not None:
            config.temperature = request.temperature
        if request.max_tokens is not None:
            config.max_output_tokens = request.max_tokens
        if request.reasoning_effort is not None:
            config.thinking_config = genai_types.ThinkingConfig(
                thinking_level=request.reasoning_effort  # type: ignore[arg-type]
            )
        cache_name = self._caches.get(_cache_key(model, request))
        if cache_name:
            config.cached_content = cache_name

        try:
            response = await self._client.aio.models.generate_content(
                model=model,
                contents=request.prompt,
                config=config,
            )
        except self._transient_error_type as e:
            self._re_raise_transient(e)
        text = response.text
        self._log_usage(response, model, request.purpose)
        if text is None:
            raise ValueError(f"No text content in Gemini response (model={model})")
        return text

    async def extract(self, model: str, request: LLMRequest, output_type: Any) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "response_model": output_type,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens

        try:
            result = await self._instructor.chat.completions.create(**kwargs)
        except self._transient_error_type as e:
            self._re_raise_transient(e)
        return result

    @staticmethod
    def _log_usage(response: Any, model: str, purpose: str) -> None:
        um = getattr(response, "usage_metadata", None)
        if um is None:
            return
        prompt = um.prompt_token_count or 0
        candidates = um.candidates_token_count or 0
        cached = um.cached_content_token_count or 0
        thoughts = um.thoughts_token_count or 0
        logger.debug(
            f"tokens: prompt={prompt} candidates={candidates} "
            f"cached={cached} thoughts={thoughts} ({model})"
        )
        LEDGER.record(
            purpose=purpose,
            model=model,
            prompt_tokens=prompt,
            cached_tokens=cached,
            completion_tokens=candidates + thoughts,
        )

    @staticmethod
    def _re_raise_transient(e: Exception) -> NoReturn:
        from google.genai.errors import ClientError  # noqa: I001, PLC0415

        if isinstance(e, ClientError) and getattr(e, "code", None) != 429:
            raise  # noqa: PLE0704 — re-raises in caller's except handler
        raise TransientLLMError(str(e)) from e

    @property
    def _transient_error_type(self) -> tuple[type[Exception], ...]:
        from asyncio import TimeoutError as AsyncTimeoutError  # noqa: I001, PLC0415
        from google.genai.errors import ClientError, ServerError  # noqa: I001, PLC0415

        return (ServerError, ClientError, AsyncTimeoutError)
