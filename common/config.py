"""Application config — typed wiring object replacing the old SimpleNamespace.

``BotConfig`` is a frozen dataclass (construct once, never mutate) that holds
live objects like the gateway.  The pydantic ``RunConfig`` in ``models.py``
is the serialized-provenance counterpart — keep them separate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from common.models import RunConfig
from llm.gateway import Gateway

# Production defaults for the --fc/--rc knobs. main.py merges these under
# CLI overrides; BotConfig itself stays neutral so replay and tests see
# exactly the config they were given. Escape hatches: --rc plan=legacy,
# news=exa, hops=1, anchor=none; --fc quant=none, tail=0.
DEFAULT_FORECAST_CONFIG: dict[str, Any] = {
    "mix": "openai:2,anthropic:1,gemini:2",
    "quant": "series,assist",
    "tail": 0.1,
}
DEFAULT_RESEARCH_CONFIG: dict[str, Any] = {
    "news": "asknews",
    "hops": 2,
    "anchor": "markets",
}


@dataclass(frozen=True)
class BotConfig:
    """Immutable application config — one instance per run, threaded everywhere.

    All fields have sensible defaults; callers override what they need.
    """

    gateway: Gateway
    live: bool = False
    submit: bool = False
    limit: int = 0
    tournament_id: str = ""
    reforecast: bool = False
    disable_research: bool = False
    strategy: str = "ensemble"
    forecast_config: dict[str, Any] = field(default_factory=dict)
    research_config: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    code_version: str | None = None
    run_cfg: RunConfig | None = None
