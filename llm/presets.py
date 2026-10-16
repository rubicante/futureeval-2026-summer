"""Default gateway wiring — the app's standard model configuration.

Shared by ``main.py`` (live forecasting) and ``scripts/run_experiment.py``
(replay) so both build identical gateways for a given ``live`` flag.
"""

from __future__ import annotations

import logging
import os

from llm.gateway import GatewayConfig, LLMGateway, ModelConfig
from llm.providers import AnthropicProvider, GeminiProvider, OpenAIProvider

logger = logging.getLogger(__name__)

LIVE_MODEL = "gpt-5.5"
DRY_RUN_MODEL = "gpt-5.4-mini"
EXTRACT_MODEL = "gpt-5.4-mini"

# Dry-run tiers — cheap models for rehearsals and integration tests.
ANTHROPIC_DRY_RUN_MODEL = "claude-haiku-4-5"
ANTHROPIC_LIVE_MODEL = "claude-opus-4-8"

GEMINI_DRY_RUN_MODEL = "gemini-2.5-flash"
GEMINI_LIVE_MODEL = "gemini-3.5-flash"


def build_default_gateway(live: bool) -> tuple[LLMGateway, str, str, dict[str, str]]:
    """Build the standard gateway. Returns (gateway, forecast_model, decompose_model,
    forecast_models)."""
    forecast_model = LIVE_MODEL if live else DRY_RUN_MODEL
    decompose_model = forecast_model

    anthropic_model = ANTHROPIC_LIVE_MODEL if live else ANTHROPIC_DRY_RUN_MODEL
    gemini_model = GEMINI_LIVE_MODEL if live else GEMINI_DRY_RUN_MODEL

    providers: dict[str, object] = {
        "openai": OpenAIProvider(temperature_unsupported_models={LIVE_MODEL}),
    }

    # Providers beyond OpenAI are optional — missing keys/packages only
    # raise if a ``mix`` string actually references the family.
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            providers["anthropic"] = AnthropicProvider(
                api_key=anthropic_key,
                temperature_unsupported_models={ANTHROPIC_LIVE_MODEL},
            )
        except ImportError:
            logger.warning("Anthropic provider unavailable: package 'anthropic' not installed")
        except Exception as exc:
            logger.warning("Anthropic provider unavailable: %s", exc)
    else:
        logger.info("ANTHROPIC_API_KEY not set — Anthropic provider skipped")

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        try:
            providers["gemini"] = GeminiProvider(api_key=gemini_key)
        except ImportError:
            logger.warning("Gemini provider unavailable: package 'google-genai' not installed")
        except Exception as exc:
            logger.warning("Gemini provider unavailable: %s", exc)
    else:
        logger.info("GEMINI_API_KEY not set — Gemini provider skipped")

    models: dict[str, ModelConfig] = {
        "extract": ModelConfig(EXTRACT_MODEL, provider="openai"),
        "decompose": ModelConfig(decompose_model, provider="openai"),
        "forecast": ModelConfig(forecast_model, provider="openai"),
    }

    if "anthropic" in providers:
        models["forecast:anthropic"] = ModelConfig(anthropic_model, provider="anthropic")
    if "gemini" in providers:
        models["forecast:gemini"] = ModelConfig(gemini_model, provider="gemini")

    gateway = LLMGateway(
        GatewayConfig(
            providers=providers,  # type: ignore[arg-type]
            models=models,
            # Questions are forecast concurrently inside short tournament
            # windows — the semaphore is the only LLM concurrency bound.
            max_concurrent=20,
        )
    )

    forecast_models: dict[str, str] = {"forecast": forecast_model}
    if "forecast:anthropic" in models:
        forecast_models["forecast:anthropic"] = anthropic_model
    if "forecast:gemini" in models:
        forecast_models["forecast:gemini"] = gemini_model

    return gateway, forecast_model, decompose_model, forecast_models
