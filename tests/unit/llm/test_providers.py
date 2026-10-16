"""Unit tests for llm/providers.py — wire translation per provider."""

from __future__ import annotations

from unittest import mock

import pytest

from llm.gateway import LLMRequest
from llm.providers import AnthropicProvider, GeminiProvider, OpenAIProvider, Provider

# ── Provider protocol adherence ───────────────────────────────────────


class TestProviderProtocol:
    def test_openai_provider_conforms(self):
        provider = OpenAIProvider(api_key="test-key")
        assert isinstance(provider, Provider)

    def test_anthropic_provider_conforms(self):
        try:
            provider = AnthropicProvider(api_key="test-key")
        except ImportError:
            pytest.skip("anthropic package not installed")
        assert isinstance(provider, Provider)

    def test_gemini_provider_conforms(self):
        try:
            provider = GeminiProvider(api_key="test-key")
        except ImportError:
            pytest.skip("google-genai package not installed")
        assert isinstance(provider, Provider)


# ── OpenAIProvider wire translation ───────────────────────────────────


class TestOpenAIProviderComplete:
    @pytest.fixture
    def mock_client(self):
        with mock.patch("llm.providers.AsyncOpenAI") as mock_cls:
            instance = mock_cls.return_value
            instance.chat.completions.create = mock.AsyncMock()
            yield instance

    @pytest.fixture
    def provider(self, mock_client):
        return OpenAIProvider(api_key="test-key")

    def _make_response(self, content: str = "response"):
        resp = mock.MagicMock()
        resp.choices = [mock.MagicMock()]
        resp.choices[0].message.content = content
        resp.usage = None
        return resp

    async def test_builds_messages_with_system_role(self, mock_client, provider):
        mock_client.chat.completions.create.return_value = self._make_response()
        req = LLMRequest(prompt="hello", purpose="test", system_prompt="be helpful")
        await provider.complete("gpt-4", req)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][0]["content"] == "be helpful"
        assert call_kwargs["messages"][1]["role"] == "user"
        assert call_kwargs["messages"][1]["content"] == "hello"

    async def test_no_system_prompt_omits_system_message(self, mock_client, provider):
        mock_client.chat.completions.create.return_value = self._make_response()
        req = LLMRequest(prompt="hello", purpose="test")
        await provider.complete("gpt-4", req)
        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        assert messages[0]["role"] == "user"

    async def test_passes_temperature(self, mock_client, provider):
        mock_client.chat.completions.create.return_value = self._make_response()
        req = LLMRequest(prompt="hello", purpose="test", temperature=0.7)
        await provider.complete("gpt-4", req)
        assert mock_client.chat.completions.create.call_args[1]["temperature"] == 0.7

    async def test_omits_temperature_when_none(self, mock_client, provider):
        mock_client.chat.completions.create.return_value = self._make_response()
        req = LLMRequest(prompt="hello", purpose="test", temperature=None)
        await provider.complete("gpt-4", req)
        assert "temperature" not in mock_client.chat.completions.create.call_args[1]

    async def test_omits_temperature_for_unsupported_model(self, mock_client):
        provider = OpenAIProvider(api_key="test-key", temperature_unsupported_models={"o1-pro"})
        mock_client.chat.completions.create.return_value = self._make_response()
        req = LLMRequest(prompt="hello", purpose="test", temperature=0.7)
        await provider.complete("o1-pro", req)
        assert "temperature" not in mock_client.chat.completions.create.call_args[1]

    async def test_passes_reasoning_effort(self, mock_client, provider):
        mock_client.chat.completions.create.return_value = self._make_response()
        req = LLMRequest(prompt="solve", purpose="test", reasoning_effort="high")
        await provider.complete("gpt-4", req)
        assert mock_client.chat.completions.create.call_args[1]["reasoning_effort"] == "high"

    async def test_passes_max_tokens(self, mock_client, provider):
        mock_client.chat.completions.create.return_value = self._make_response()
        req = LLMRequest(prompt="hello", purpose="test", max_tokens=200)
        await provider.complete("gpt-4", req)
        assert mock_client.chat.completions.create.call_args[1]["max_completion_tokens"] == 200

    async def test_none_content_raises(self, mock_client, provider):
        mock_client.chat.completions.create.return_value = self._make_response(content=None)
        req = LLMRequest(prompt="hello", purpose="test")
        with pytest.raises(ValueError, match="No answer returned"):
            await provider.complete("gpt-4", req)

    async def test_perplexity_prefix_skip(self, mock_client, provider):
        mock_client.chat.completions.create.return_value = self._make_response("perplexity answer")
        req = LLMRequest(prompt="query", purpose="test")
        result = await provider.complete("perplexity/test-model", req)
        assert result == "perplexity answer"

    async def test_returns_content(self, mock_client, provider):
        mock_client.chat.completions.create.return_value = self._make_response("hello world")
        req = LLMRequest(prompt="greet", purpose="test")
        result = await provider.complete("gpt-4", req)
        assert result == "hello world"


class TestOpenAIProviderExtract:
    @pytest.fixture
    def mock_instructor(self):
        with mock.patch("llm.providers.instructor.from_openai") as mock_from:
            instance = mock_from.return_value
            instance.chat.completions.create_with_completion = mock.AsyncMock()
            yield instance

    @pytest.fixture
    def provider(self):
        return OpenAIProvider(api_key="test-key")

    def _make_response(self, parsed):
        raw = mock.MagicMock()
        raw.usage = None
        return (parsed, raw)

    async def test_returns_parsed_model(self, mock_instructor, provider):
        mock_instructor.chat.completions.create_with_completion.return_value = self._make_response(
            {"key": "value"}
        )
        req = LLMRequest(prompt="parse me", purpose="extract")
        result = await provider.extract("gpt-4", req, dict)
        assert result == {"key": "value"}

    async def test_passes_temperature_to_instructor(self, mock_instructor, provider):
        mock_instructor.chat.completions.create_with_completion.return_value = self._make_response(
            "result"
        )
        req = LLMRequest(prompt="test", purpose="extract", temperature=0.5)
        await provider.extract("gpt-4", req, str)
        assert (
            mock_instructor.chat.completions.create_with_completion.call_args[1]["temperature"]
            == 0.5
        )

    async def test_omits_temperature_for_unsupported(self, mock_instructor):
        provider = OpenAIProvider(api_key="test-key", temperature_unsupported_models={"o1-pro"})
        mock_instructor.chat.completions.create_with_completion.return_value = self._make_response(
            "result"
        )
        req = LLMRequest(prompt="test", purpose="extract", temperature=0.7)
        await provider.extract("o1-pro", req, str)
        assert (
            "temperature"
            not in mock_instructor.chat.completions.create_with_completion.call_args[1]
        )

    async def test_passes_reasoning_effort(self, mock_instructor, provider):
        mock_instructor.chat.completions.create_with_completion.return_value = self._make_response(
            "result"
        )
        req = LLMRequest(prompt="test", purpose="extract", reasoning_effort="medium")
        await provider.extract("gpt-4", req, str)
        assert (
            mock_instructor.chat.completions.create_with_completion.call_args[1]["reasoning_effort"]
            == "medium"
        )

    async def test_passes_max_tokens(self, mock_instructor, provider):
        mock_instructor.chat.completions.create_with_completion.return_value = self._make_response(
            "result"
        )
        req = LLMRequest(prompt="test", purpose="extract", max_tokens=100)
        await provider.extract("gpt-4", req, str)
        assert (
            mock_instructor.chat.completions.create_with_completion.call_args[1][
                "max_completion_tokens"
            ]
            == 100
        )


# ── AnthropicProvider wire translation ────────────────────────────────


class TestAnthropicProviderComplete:
    @pytest.fixture
    def mock_anthropic(self):
        with (
            mock.patch("anthropic.AsyncAnthropic") as mock_cls,
            mock.patch("llm.providers.instructor.from_anthropic"),
        ):
            instance = mock_cls.return_value
            instance.messages.create = mock.AsyncMock()
            yield instance

    @pytest.fixture
    def provider(self, mock_anthropic):
        return AnthropicProvider(api_key="test-key")

    def _make_response(self, text: str = "anthropic response"):
        resp = mock.MagicMock()
        block = mock.MagicMock()
        block.type = "text"
        block.text = text
        resp.content = [block]
        return resp

    async def test_builds_messages_with_top_level_system(self, mock_anthropic, provider):
        mock_anthropic.messages.create.return_value = self._make_response()
        req = LLMRequest(prompt="hello", purpose="test", system_prompt="be helpful")
        await provider.complete("claude-sonnet-4-20250514", req)
        call_kwargs = mock_anthropic.messages.create.call_args[1]
        assert call_kwargs["system"] == "be helpful"
        assert call_kwargs["messages"][0]["role"] == "user"
        assert call_kwargs["messages"][0]["content"] == "hello"

    async def test_no_system_prompt_omits_system_param(self, mock_anthropic, provider):
        mock_anthropic.messages.create.return_value = self._make_response()
        req = LLMRequest(prompt="hello", purpose="test")
        await provider.complete("claude-sonnet-4-20250514", req)
        assert "system" not in mock_anthropic.messages.create.call_args[1]

    async def test_passes_temperature(self, mock_anthropic, provider):
        mock_anthropic.messages.create.return_value = self._make_response()
        req = LLMRequest(prompt="hello", purpose="test", temperature=0.5)
        await provider.complete("claude-sonnet-4-20250514", req)
        assert mock_anthropic.messages.create.call_args[1]["temperature"] == 0.5

    async def test_maps_reasoning_effort_to_thinking_budget_low(self, mock_anthropic, provider):
        mock_anthropic.messages.create.return_value = self._make_response()
        req = LLMRequest(prompt="think", purpose="test", reasoning_effort="low")
        await provider.complete("claude-sonnet-4-20250514", req)
        thinking = mock_anthropic.messages.create.call_args[1]["thinking"]
        assert thinking["type"] == "enabled"
        assert thinking["budget_tokens"] == 1024

    async def test_maps_reasoning_effort_to_thinking_budget_high(self, mock_anthropic, provider):
        mock_anthropic.messages.create.return_value = self._make_response()
        req = LLMRequest(prompt="think", purpose="test", reasoning_effort="high")
        await provider.complete("claude-sonnet-4-20250514", req)
        thinking = mock_anthropic.messages.create.call_args[1]["thinking"]
        assert thinking["budget_tokens"] == 16384

    async def test_returns_text_content(self, mock_anthropic, provider):
        mock_anthropic.messages.create.return_value = self._make_response("claude says hi")
        req = LLMRequest(prompt="hi", purpose="test")
        result = await provider.complete("claude-sonnet-4-20250514", req)
        assert result == "claude says hi"

    async def test_no_text_content_raises(self, mock_anthropic, provider):
        resp = mock.MagicMock()
        resp.content = []  # no text blocks
        mock_anthropic.messages.create.return_value = resp
        req = LLMRequest(prompt="hi", purpose="test")
        with pytest.raises(ValueError, match="No text content"):
            await provider.complete("claude-sonnet-4-20250514", req)

    async def test_default_max_tokens(self, mock_anthropic, provider):
        mock_anthropic.messages.create.return_value = self._make_response()
        req = LLMRequest(prompt="hi", purpose="test")
        await provider.complete("claude-sonnet-4-20250514", req)
        assert mock_anthropic.messages.create.call_args[1]["max_tokens"] == 4096


class TestAnthropicProviderExtract:
    @pytest.fixture
    def mock_instructor(self):
        with mock.patch("llm.providers.instructor.from_anthropic") as mock_from:
            instance = mock_from.return_value
            instance.messages.create_with_completion = mock.AsyncMock()
            yield instance

    @pytest.fixture
    def provider(self):
        return AnthropicProvider(api_key="test-key")

    def _make_response(self, parsed):
        raw = mock.MagicMock()
        raw.usage = None
        return (parsed, raw)

    async def test_returns_parsed_model(self, mock_instructor, provider):
        mock_instructor.messages.create_with_completion.return_value = self._make_response(
            {"key": "value"}
        )
        req = LLMRequest(prompt="parse me", purpose="extract")
        result = await provider.extract("claude-sonnet-4-20250514", req, dict)
        assert result == {"key": "value"}

    async def test_passes_system_prompt(self, mock_instructor, provider):
        mock_instructor.messages.create_with_completion.return_value = self._make_response("ok")
        req = LLMRequest(prompt="test", purpose="extract", system_prompt="be precise")
        await provider.extract("claude-sonnet-4-20250514", req, str)
        call_kwargs = mock_instructor.messages.create_with_completion.call_args[1]
        assert call_kwargs["system"] == "be precise"

    async def test_maps_reasoning_effort_to_thinking(self, mock_instructor, provider):
        mock_instructor.messages.create_with_completion.return_value = self._make_response("ok")
        req = LLMRequest(prompt="test", purpose="extract", reasoning_effort="medium")
        await provider.extract("claude-sonnet-4-20250514", req, str)
        call_kwargs = mock_instructor.messages.create_with_completion.call_args[1]
        assert call_kwargs["thinking"]["budget_tokens"] == 4096

    async def test_default_max_tokens(self, mock_instructor, provider):
        mock_instructor.messages.create_with_completion.return_value = self._make_response("ok")
        req = LLMRequest(prompt="test", purpose="extract")
        await provider.extract("claude-sonnet-4-20250514", req, str)
        call_kwargs = mock_instructor.messages.create_with_completion.call_args[1]
        assert call_kwargs["max_tokens"] == 4096


# ── GeminiProvider wire translation ──────────────────────────────────


class TestGeminiProviderComplete:
    @pytest.fixture
    def mock_gemini(self):
        with (
            mock.patch("google.genai.Client") as mock_cls,
            mock.patch("llm.providers.instructor.from_genai"),
        ):
            instance = mock_cls.return_value
            instance.aio.models.generate_content = mock.AsyncMock()
            yield instance

    @pytest.fixture
    def provider(self, mock_gemini):
        return GeminiProvider(api_key="test-key")

    def _make_response(self, text: str = "gemini response"):
        resp = mock.MagicMock()
        resp.text = text
        return resp

    async def test_returns_response_text(self, mock_gemini, provider):
        mock_gemini.aio.models.generate_content.return_value = self._make_response(
            "hello from gemini"
        )
        req = LLMRequest(prompt="hi", purpose="test")
        result = await provider.complete("gemini-2.5-flash", req)
        assert result == "hello from gemini"

    async def test_passes_prompt_as_contents(self, mock_gemini, provider):
        mock_gemini.aio.models.generate_content.return_value = self._make_response()
        req = LLMRequest(prompt="my prompt", purpose="test")
        await provider.complete("gemini-2.5-flash", req)
        call_kwargs = mock_gemini.aio.models.generate_content.call_args[1]
        assert call_kwargs["contents"] == "my prompt"

    async def test_sets_system_instruction_in_config(self, mock_gemini, provider):
        mock_gemini.aio.models.generate_content.return_value = self._make_response()
        req = LLMRequest(prompt="hello", purpose="test", system_prompt="be helpful")
        await provider.complete("gemini-2.5-flash", req)
        config = mock_gemini.aio.models.generate_content.call_args[1]["config"]
        assert config.system_instruction == "be helpful"

    async def test_no_system_instruction_when_none(self, mock_gemini, provider):
        mock_gemini.aio.models.generate_content.return_value = self._make_response()
        req = LLMRequest(prompt="hello", purpose="test")
        await provider.complete("gemini-2.5-flash", req)
        config = mock_gemini.aio.models.generate_content.call_args[1]["config"]
        assert config.system_instruction is None

    async def test_sets_temperature_in_config(self, mock_gemini, provider):
        mock_gemini.aio.models.generate_content.return_value = self._make_response()
        req = LLMRequest(prompt="hello", purpose="test", temperature=0.5)
        await provider.complete("gemini-2.5-flash", req)
        config = mock_gemini.aio.models.generate_content.call_args[1]["config"]
        assert config.temperature == 0.5

    async def test_sets_max_output_tokens(self, mock_gemini, provider):
        mock_gemini.aio.models.generate_content.return_value = self._make_response()
        req = LLMRequest(prompt="hello", purpose="test", max_tokens=200)
        await provider.complete("gemini-2.5-flash", req)
        config = mock_gemini.aio.models.generate_content.call_args[1]["config"]
        assert config.max_output_tokens == 200

    async def test_maps_reasoning_effort_to_thinking_level(self, mock_gemini, provider):
        from google.genai.types import ThinkingLevel

        mock_gemini.aio.models.generate_content.return_value = self._make_response()
        req = LLMRequest(prompt="solve", purpose="test", reasoning_effort="high")
        await provider.complete("gemini-2.5-flash", req)
        config = mock_gemini.aio.models.generate_content.call_args[1]["config"]
        assert config.thinking_config.thinking_level == ThinkingLevel.HIGH

    async def test_wraps_transient_error(self, mock_gemini, provider):
        from google.genai.errors import ServerError

        from llm.gateway import TransientLLMError

        mock_gemini.aio.models.generate_content.side_effect = ServerError(
            500,
            {},
        )
        req = LLMRequest(prompt="hi", purpose="test")
        with pytest.raises(TransientLLMError):
            await provider.complete("gemini-2.5-flash", req)

    async def test_non_transient_client_error_propagates(self, mock_gemini, provider):
        from google.genai.errors import ClientError

        err = ClientError(400, {})
        mock_gemini.aio.models.generate_content.side_effect = err
        req = LLMRequest(prompt="hi", purpose="test")
        with pytest.raises(ClientError):
            await provider.complete("gemini-2.5-flash", req)

    async def test_rate_limit_429_is_transient(self, mock_gemini, provider):
        from google.genai.errors import ClientError

        from llm.gateway import TransientLLMError

        err = ClientError(429, {})
        mock_gemini.aio.models.generate_content.side_effect = err
        req = LLMRequest(prompt="hi", purpose="test")
        with pytest.raises(TransientLLMError):
            await provider.complete("gemini-2.5-flash", req)

    async def test_vertexai_mode(self):
        with (
            mock.patch("google.genai.Client") as mock_cls,
            mock.patch("llm.providers.instructor.from_genai"),
        ):
            GeminiProvider(vertexai=True, project="p", location="l")
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["vertexai"] is True
            assert call_kwargs["project"] == "p"
            assert call_kwargs["location"] == "l"

    async def test_timeout_is_transient(self, mock_gemini, provider):
        from llm.gateway import TransientLLMError

        mock_gemini.aio.models.generate_content.side_effect = TimeoutError()
        req = LLMRequest(prompt="hi", purpose="test")
        with pytest.raises(TransientLLMError):
            await provider.complete("gemini-2.5-flash", req)


class TestGeminiProviderExtract:
    @pytest.fixture
    def mock_instructor(self):
        with (
            mock.patch("google.genai.Client"),
            mock.patch("llm.providers.instructor.from_genai") as mock_from,
        ):
            instance = mock_from.return_value
            instance.chat.completions.create = mock.AsyncMock()
            yield instance

    @pytest.fixture
    def provider(self):
        return GeminiProvider(api_key="test-key")

    async def test_returns_parsed_model(self, mock_instructor, provider):
        mock_instructor.chat.completions.create.return_value = {"key": "value"}
        req = LLMRequest(prompt="parse me", purpose="extract")
        result = await provider.extract("gemini-2.5-flash", req, dict)
        assert result == {"key": "value"}

    async def test_passes_max_tokens(self, mock_instructor, provider):
        mock_instructor.chat.completions.create.return_value = "ok"
        req = LLMRequest(prompt="test", purpose="extract", max_tokens=100)
        await provider.extract("gemini-2.5-flash", req, str)
        call_kwargs = mock_instructor.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 100

    async def test_extract_wraps_transient_error(self, mock_instructor, provider):
        from google.genai.errors import ServerError

        from llm.gateway import TransientLLMError

        mock_instructor.chat.completions.create.side_effect = ServerError(500, {})
        req = LLMRequest(prompt="test", purpose="extract")
        with pytest.raises(TransientLLMError):
            await provider.extract("gemini-2.5-flash", req, str)


# ── OpenAIProvider cache ───────────────────────────────────────────────


class TestOpenAIProviderCache:
    @pytest.fixture
    def mock_client(self):
        with mock.patch("llm.providers.AsyncOpenAI") as mock_cls:
            instance = mock_cls.return_value
            instance.chat.completions.create = mock.AsyncMock()
            yield instance

    @pytest.fixture
    def provider(self, mock_client):
        return OpenAIProvider(api_key="test-key")

    def _make_response(self, content: str = "response"):
        resp = mock.MagicMock()
        resp.choices = [mock.MagicMock()]
        resp.choices[0].message.content = content
        resp.usage = None
        return resp

    async def test_prepare_cache_fires_completion(self, mock_client, provider):
        mock_client.chat.completions.create.return_value = self._make_response()
        req = LLMRequest(prompt="hello", purpose="test", system_prompt="be helpful")
        await provider.prepare_cache("gpt-4", req, count=2)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_completion_tokens"] == 100
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][0]["content"] == "be helpful"
        assert call_kwargs["messages"][1]["role"] == "user"
        assert call_kwargs["messages"][1]["content"] == "hello"

    async def test_prepare_cache_does_not_propagate_exceptions(self, mock_client, provider):
        mock_client.chat.completions.create.side_effect = RuntimeError("boom")
        req = LLMRequest(prompt="hello", purpose="test")
        await provider.prepare_cache("gpt-4", req, count=2)

    async def test_prepare_cache_skips_single_request(self, mock_client, provider):
        await provider.prepare_cache("gpt-4", LLMRequest(prompt="x", purpose="test"))
        mock_client.chat.completions.create.assert_not_called()

    async def test_expire_cache_noops(self, provider):
        await provider.expire_cache("gpt-4", LLMRequest(prompt="x", purpose="test"))

    async def test_prepare_cache_passes_reasoning_effort(self, mock_client, provider):
        mock_client.chat.completions.create.return_value = self._make_response()
        req = LLMRequest(prompt="solve", purpose="test", reasoning_effort="high")
        await provider.prepare_cache("gpt-4", req, count=2)
        assert mock_client.chat.completions.create.call_args[1]["reasoning_effort"] == "high"


# ── AnthropicProvider cache ────────────────────────────────────────────


class TestAnthropicProviderCache:
    @pytest.fixture
    def mock_anthropic(self):
        with (
            mock.patch("anthropic.AsyncAnthropic") as mock_cls,
            mock.patch("llm.providers.instructor.from_anthropic"),
        ):
            instance = mock_cls.return_value
            instance.messages.create = mock.AsyncMock()
            yield instance

    @pytest.fixture
    def provider(self, mock_anthropic):
        return AnthropicProvider(api_key="test-key")

    def _make_response(self, text: str = "anthropic response"):
        resp = mock.MagicMock()
        block = mock.MagicMock()
        block.type = "text"
        block.text = text
        resp.content = [block]
        return resp

    async def test_prepare_cache_fires_primer(self, mock_anthropic, provider):
        mock_anthropic.messages.create.return_value = self._make_response()
        req = LLMRequest(prompt="forecast", purpose="test", system_prompt="be helpful")
        await provider.prepare_cache("claude-sonnet-4-20250514", req, count=2)
        call_kwargs = mock_anthropic.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 1
        assert call_kwargs["cache_control"] == {"type": "ephemeral"}
        assert call_kwargs["system"] == "be helpful"
        assert call_kwargs["messages"] == [{"role": "user", "content": "forecast"}]
        assert "thinking" not in call_kwargs

    async def test_prepare_cache_skips_single_request(self, mock_anthropic, provider):
        req = LLMRequest(prompt="forecast", purpose="test")
        await provider.prepare_cache("claude-sonnet-4-20250514", req)
        mock_anthropic.messages.create.assert_not_called()

    async def test_prepare_cache_skips_haiku(self, mock_anthropic, provider):
        req = LLMRequest(prompt="forecast", purpose="test")
        await provider.prepare_cache("claude-haiku-4-5-20251001", req, count=3)
        mock_anthropic.messages.create.assert_not_called()

    async def test_prepare_cache_does_not_propagate_exceptions(self, mock_anthropic, provider):
        mock_anthropic.messages.create.side_effect = RuntimeError("boom")
        req = LLMRequest(prompt="forecast", purpose="test")
        await provider.prepare_cache("claude-sonnet-4-20250514", req, count=2)
        assert provider._primed == set()

    async def test_primed_complete_marks_cache_control_prompt_once(
        self, mock_anthropic, provider
    ):
        mock_anthropic.messages.create.return_value = self._make_response()
        req = LLMRequest(prompt="forecast", purpose="test", system_prompt="be helpful")
        await provider.prepare_cache("claude-sonnet-4-20250514", req, count=2)

        await provider.complete("claude-sonnet-4-20250514", req)
        call_kwargs = mock_anthropic.messages.create.call_args[1]
        # prompt appears exactly once — the old prefix path sent it twice
        assert call_kwargs["messages"] == [{"role": "user", "content": "forecast"}]
        assert call_kwargs["cache_control"] == {"type": "ephemeral"}
        assert call_kwargs["system"] == "be helpful"

    async def test_unprimed_complete_omits_cache_control(self, mock_anthropic, provider):
        mock_anthropic.messages.create.return_value = self._make_response()
        req = LLMRequest(prompt="hi", purpose="test", system_prompt="be helpful")
        await provider.complete("claude-sonnet-4-20250514", req)
        call_kwargs = mock_anthropic.messages.create.call_args[1]
        assert call_kwargs["system"] == "be helpful"
        assert "cache_control" not in call_kwargs

    async def test_expire_cache_unprimes(self, mock_anthropic, provider):
        mock_anthropic.messages.create.return_value = self._make_response()
        req = LLMRequest(prompt="forecast", purpose="test")
        await provider.prepare_cache("claude-sonnet-4-20250514", req, count=2)
        await provider.expire_cache("claude-sonnet-4-20250514", req)
        await provider.complete("claude-sonnet-4-20250514", req)
        assert "cache_control" not in mock_anthropic.messages.create.call_args[1]

    async def test_concurrent_questions_keep_separate_priming(self, mock_anthropic, provider):
        """Regression: a second question's prepare/expire must not touch the
        first question's cache state (2026-07-11 cross-question cache race)."""
        mock_anthropic.messages.create.return_value = self._make_response()
        req_a = LLMRequest(prompt="question A", purpose="test")
        req_b = LLMRequest(prompt="question B", purpose="test")
        await provider.prepare_cache("claude-sonnet-4-20250514", req_a, count=2)
        await provider.prepare_cache("claude-sonnet-4-20250514", req_b, count=2)
        await provider.expire_cache("claude-sonnet-4-20250514", req_b)

        await provider.complete("claude-sonnet-4-20250514", req_a)
        call_kwargs = mock_anthropic.messages.create.call_args[1]
        assert call_kwargs["messages"] == [{"role": "user", "content": "question A"}]
        assert call_kwargs["cache_control"] == {"type": "ephemeral"}


# ── GeminiProvider cache ───────────────────────────────────────────────


class TestGeminiProviderCache:
    @pytest.fixture
    def mock_gemini(self):
        with (
            mock.patch("google.genai.Client") as mock_cls,
            mock.patch("llm.providers.instructor.from_genai"),
        ):
            instance = mock_cls.return_value
            instance.aio.models.generate_content = mock.AsyncMock()
            instance.caches = mock.MagicMock()
            instance.caches.create = mock.MagicMock()
            instance.caches.delete = mock.MagicMock()
            yield instance

    @pytest.fixture
    def provider(self, mock_gemini):
        return GeminiProvider(api_key="test-key")

    def _make_response(self, text: str = "gemini response"):
        resp = mock.MagicMock()
        resp.text = text
        return resp

    def _make_cache_result(self, name: str = "projects/p/locations/l/cachedContents/123"):
        cache = mock.MagicMock()
        cache.name = name
        return cache

    async def test_prepare_cache_calls_caches_create(self, mock_gemini, provider):
        mock_gemini.caches.create.return_value = self._make_cache_result()
        req = LLMRequest(prompt="forecast", purpose="test", system_prompt="be helpful")
        await provider.prepare_cache("gemini-2.5-flash", req, count=2)
        mock_gemini.caches.create.assert_called_once()
        call_kwargs = mock_gemini.caches.create.call_args[1]
        assert call_kwargs["model"] == "gemini-2.5-flash"
        assert call_kwargs["config"].system_instruction is not None

    async def test_prepare_cache_stores_cache_name(self, mock_gemini, provider):
        mock_gemini.caches.create.return_value = self._make_cache_result(
            name="projects/p/locations/l/cachedContents/abc"
        )
        req = LLMRequest(prompt="forecast", purpose="test")
        await provider.prepare_cache("gemini-2.5-flash", req, count=2)
        assert list(provider._caches.values()) == ["projects/p/locations/l/cachedContents/abc"]

    async def test_expire_cache_calls_caches_delete(self, mock_gemini, provider):
        mock_gemini.caches.create.return_value = self._make_cache_result()
        req = LLMRequest(prompt="forecast", purpose="test")
        await provider.prepare_cache("gemini-2.5-flash", req, count=2)
        await provider.expire_cache("gemini-2.5-flash", req)
        mock_gemini.caches.delete.assert_called_once_with(
            name="projects/p/locations/l/cachedContents/123"
        )
        assert provider._caches == {}

    async def test_complete_passes_cached_content(self, mock_gemini, provider):
        mock_gemini.caches.create.return_value = self._make_cache_result(
            name="projects/p/locations/l/cachedContents/abc"
        )
        mock_gemini.aio.models.generate_content.return_value = self._make_response()
        req = LLMRequest(prompt="forecast", purpose="test")
        await provider.prepare_cache("gemini-2.5-flash", req, count=2)
        await provider.complete("gemini-2.5-flash", req)
        call_kwargs = mock_gemini.aio.models.generate_content.call_args[1]
        assert call_kwargs["config"].cached_content == "projects/p/locations/l/cachedContents/abc"

    async def test_complete_without_cache_no_cached_content(self, mock_gemini, provider):
        mock_gemini.aio.models.generate_content.return_value = self._make_response()
        req = LLMRequest(prompt="forecast", purpose="test")
        await provider.complete("gemini-2.5-flash", req)
        call_kwargs = mock_gemini.aio.models.generate_content.call_args[1]
        assert call_kwargs["config"].cached_content is None

    async def test_double_prepare_same_request_deletes_old_cache(self, mock_gemini, provider):
        mock_gemini.caches.create.return_value = self._make_cache_result(
            name="projects/p/locations/l/cachedContents/abc"
        )
        req = LLMRequest(prompt="forecast", purpose="test")
        await provider.prepare_cache("gemini-2.5-flash", req, count=2)

        mock_gemini.caches.create.return_value = self._make_cache_result(
            name="projects/p/locations/l/cachedContents/def"
        )
        await provider.prepare_cache("gemini-2.5-flash", req, count=2)
        mock_gemini.caches.delete.assert_called_once_with(
            name="projects/p/locations/l/cachedContents/abc"
        )
        assert list(provider._caches.values()) == ["projects/p/locations/l/cachedContents/def"]

    async def test_concurrent_questions_keep_separate_caches(self, mock_gemini, provider):
        """Regression: question B's prepare_cache deleted question A's live
        cache, 403-ing A's in-flight samples (2026-07-11 run 020003)."""
        mock_gemini.aio.models.generate_content.return_value = self._make_response()
        req_a = LLMRequest(prompt="question A", purpose="test")
        req_b = LLMRequest(prompt="question B", purpose="test")

        mock_gemini.caches.create.return_value = self._make_cache_result(name="caches/a")
        await provider.prepare_cache("gemini-2.5-flash", req_a, count=2)
        mock_gemini.caches.create.return_value = self._make_cache_result(name="caches/b")
        await provider.prepare_cache("gemini-2.5-flash", req_b, count=2)
        mock_gemini.caches.delete.assert_not_called()

        await provider.complete("gemini-2.5-flash", req_a)
        call_kwargs = mock_gemini.aio.models.generate_content.call_args[1]
        assert call_kwargs["config"].cached_content == "caches/a"

        await provider.expire_cache("gemini-2.5-flash", req_b)
        mock_gemini.caches.delete.assert_called_once_with(name="caches/b")
        await provider.complete("gemini-2.5-flash", req_a)
        call_kwargs = mock_gemini.aio.models.generate_content.call_args[1]
        assert call_kwargs["config"].cached_content == "caches/a"

    async def test_prepare_cache_skips_single_request(self, mock_gemini, provider):
        await provider.prepare_cache("gemini-2.5-flash", LLMRequest(prompt="x", purpose="test"))
        mock_gemini.caches.create.assert_not_called()

    async def test_expire_cache_safe_when_no_active_cache(self, provider):
        await provider.expire_cache("gemini-2.5-flash", LLMRequest(prompt="x", purpose="test"))

    async def test_expire_cache_swallows_delete_failure(self, mock_gemini, provider):
        mock_gemini.caches.create.return_value = self._make_cache_result()
        mock_gemini.caches.delete.side_effect = RuntimeError("gone")
        req = LLMRequest(prompt="forecast", purpose="test")
        await provider.prepare_cache("gemini-2.5-flash", req, count=2)
        await provider.expire_cache("gemini-2.5-flash", req)
        assert provider._caches == {}
