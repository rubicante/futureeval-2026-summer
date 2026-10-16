"""Cost ledger — token and dollar accounting across LLM calls.

Providers record usage into the module-level ``LEDGER``; ``main`` sets
``current_question`` so spend is attributable per question. Prices are a
static table — estimates until re-verified, not billing truth.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from contextvars import ContextVar
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

current_question: ContextVar[int | None] = ContextVar("current_question", default=None)

# USD per million tokens: (input, cached input, output). Re-verify against
# provider pricing pages when models change.
PRICES_PER_M: dict[str, tuple[float, float, float]] = {
    "gpt-5.5": (5.0, 0.5, 30.0),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "claude-haiku-4-5": (1.0, 0.10, 5.0),
    "claude-opus-4-8": (5.0, 0.50, 25.0),
    "gemini-2.5-flash": (0.30, 0.030, 2.50),
    "gemini-2.5-pro": (2.0, 0.20, 12.0),
    "gemini-3.5-flash": (1.50, 0.15, 9.0),
}

_warned_missing_prices: set[str] = set()


@dataclass
class Tally:
    calls: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


class CostLedger:
    def __init__(self) -> None:
        self._by_key: dict[tuple[str, str], Tally] = defaultdict(Tally)
        self._by_question: dict[int, Tally] = defaultdict(Tally)

    def record(
        self,
        *,
        purpose: str,
        model: str,
        prompt_tokens: int,
        cached_tokens: int,
        completion_tokens: int,
    ) -> None:
        if model not in PRICES_PER_M:
            if model not in _warned_missing_prices:
                logger.warning(
                    "Model %r not in PRICES_PER_M — cost will be $0. "
                    "Add a price row to llm/costs.py.",
                    model,
                )
                _warned_missing_prices.add(model)
            in_rate = cached_rate = out_rate = 0.0
        else:
            in_rate, cached_rate, out_rate = PRICES_PER_M[model]
        cost = (
            (prompt_tokens - cached_tokens) * in_rate
            + cached_tokens * cached_rate
            + completion_tokens * out_rate
        ) / 1e6
        tallies = [self._by_key[(purpose, model)]]
        qid = current_question.get()
        if qid is not None:
            tallies.append(self._by_question[qid])
        for t in tallies:
            t.calls += 1
            t.prompt_tokens += prompt_tokens
            t.cached_tokens += cached_tokens
            t.completion_tokens += completion_tokens
            t.cost_usd += cost

    def question_cost(self, question_id: int) -> dict | None:
        tally = self._by_question.get(question_id)
        return asdict(tally) if tally else None

    def total_cost_usd(self) -> float:
        return sum(t.cost_usd for t in self._by_key.values())

    def format_summary(self) -> str:
        if not self._by_key:
            return "Cost summary: no LLM usage recorded"
        lines = ["Cost summary (static price table — estimates):"]
        for (purpose, model), t in sorted(self._by_key.items()):
            lines.append(
                f"  {purpose:<10} {model:<14} calls={t.calls:<4} "
                f"in={t.prompt_tokens} (cached {t.cached_tokens}) "
                f"out={t.completion_tokens}  ${t.cost_usd:.4f}"
            )
        lines.append(f"  total ${self.total_cost_usd():.4f}")
        return "\n".join(lines)


LEDGER = CostLedger()
