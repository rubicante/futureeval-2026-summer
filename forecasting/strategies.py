"""Forecast strategies — how forecasts are produced.

Each Strategy implements a forecasting methodology reusing the same
parse/aggregate/trim primitives.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

from common.config import BotConfig
from forecasting.humility import apply_tail_humility
from llm.gateway import LLMRequest
from research.market_anchor import anchor_modes, anchor_report
from research.quant_assist import quant_assist_report, quant_modes
from research.strategies import ResearchStrategy, no_research, planned_research

from .primitives import (
    ParseDeps,
    Primitive,
    _enrich_context,
    aggregate_weighted_scalar_mean,
    format_forecast,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class Sample(Generic[T]):
    """One forecast sample — the value, rationale, and per-sample metadata.

    ``model`` identifies which LLM produced this sample (needed for capture
    replay and per-family accuracy tracking).  ``weight`` is the aggregation
    weight (1.0 by default; varied by cross-model ensembles).
    """

    forecast: T
    rationale: str
    model: str | None = None
    weight: float = 1.0


def _sample_model_list(results: list[Sample[Any]]) -> list[str] | None:
    """Return a list of model names only if every sample has one.

    Partial None values would silently misalign with ``raw_results`` in the
    capture (the exact failure class Appendix A.1 guards against), so the
    field is left ``None`` unless attribution is complete.
    """
    models = [s.model for s in results]
    if any(m is None for m in models):
        return None
    return models  # type: ignore[return-value]


def _parse_mix(mix_spec: str) -> list[tuple[str, int, float]]:
    """Parse a mix string into ordered *(purpose, count, weight)* entries.

    Examples::

        "openai:2,anthropic:2,gemini:1"          → equal weights
        "openai:2@1.0,anthropic:2@1.0,gemini:1@0.5"
    """
    entries: list[tuple[str, int, float]] = []
    for raw_item in mix_spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        family_part, _, weight_str = item.partition("@")
        family, _, count_str = family_part.partition(":")
        count = int(count_str)
        weight = float(weight_str) if weight_str else 1.0
        purpose = "forecast" if family == "openai" else f"forecast:{family}"
        entries.append((purpose, count, weight))
    if not entries:
        raise ValueError(f"Empty or unparseable mix spec: {mix_spec!r}")
    return entries


@dataclass
class EnsembleHealth:
    """Per-question ensemble health — how many planned samples survived.

    ``by_family`` maps family name (e.g. ``"openai"``) to
    ``(planned, survived)`` counts for that question.
    """

    planned: int
    survived: int
    by_family: dict[str, tuple[int, int]]

    def format_one_line(self) -> str:
        parts = [f"{family}: {s}/{p}" for family, (p, s) in sorted(self.by_family.items())]
        line = f"Ensemble health: {', '.join(parts)}"
        if self.planned > self.survived:
            line += f" ({self.planned - self.survived} lost)"
        return line


@dataclass
class StrategyResult(Generic[T]):
    forecast: T
    comment: str
    ctx: dict[str, Any]
    raw_results: list[tuple[T, str]]
    sample_models: list[str] | None = None
    health: EnsembleHealth | None = None


class Strategy(ABC, Generic[T]):
    """A forecasting methodology.

    Subclass and override ``produce`` to define a new strategy.
    Override ``format``, or ``run`` for deeper customization.
    """

    def __init__(
        self,
        research: ResearchStrategy = planned_research,
    ):
        self._research = research

    async def run(
        self,
        question: Any,
        config: BotConfig,
        prim: Primitive[T],
    ) -> StrategyResult[T]:
        ctx = await self._research(question, config)
        ctx = _enrich_context(question, ctx)
        if "assist" in quant_modes(config.forecast_config.get("quant")):
            report = await quant_assist_report(question)
            if report:
                ctx["_quant_assist"] = report
                ctx["summary_report"] = f"{report}\n\n{ctx.get('summary_report') or ''}"
        if "markets" in anchor_modes(config.research_config.get("anchor")):
            anchor = await anchor_report(question, config.gateway)
            if anchor:
                block, meta = anchor
                ctx["_anchor"] = meta
                ctx["summary_report"] = f"{block}\n\n{ctx.get('summary_report') or ''}"
        return await self.run_from_context(question, ctx, config, prim)

    async def run_from_context(
        self,
        question: Any,
        ctx: dict[str, Any],
        config: BotConfig,
        prim: Primitive[T],
    ) -> StrategyResult[T]:
        """Run produce/trim/aggregate from a pre-built context dict.

        Skips ``_research()`` and ``_enrich_context()`` — the *ctx* is
        assumed to already contain everything the prompt templates need.
        """
        deps = ParseDeps(
            gateway=config.gateway,
            scaling=question.scaling,
            options=question.options,
        )
        results = await self.produce(question, ctx, deps, config, prim)
        health = getattr(self, "_last_health", None)
        forecasts = [s.forecast for s in results]
        keep = prim.trim(forecasts) or list(range(len(results)))
        clean_samples = [results[i] for i in keep]
        clean = [s.forecast for s in clean_samples]

        # ── aggregation ──────────────────────────────────────────────
        values: list[Any] = [s.forecast for s in clean_samples]
        weights = [s.weight for s in clean_samples]

        fc = config.forecast_config
        agg_mode: str | None = fc.get("agg")
        qtype = getattr(question, "question_type", None)
        qtype_value: object | None = getattr(qtype, "value", None)
        if qtype_value is not None and hasattr(qtype_value, "value"):
            qtype_value = qtype_value.value  # type: ignore[assignment]

        if agg_mode == "mean" and qtype_value == "binary":
            final = aggregate_weighted_scalar_mean(  # type: ignore[assignment]
                cast("list[float]", values), weights
            )
        else:
            if agg_mode == "mean" and qtype_value is not None:
                logger.debug(
                    "agg=mean ignored for question type %r (binary only)",
                    qtype_value,
                )
            if any(w != 1.0 for w in weights) and prim.aggregate_weighted is not None:
                final = prim.aggregate_weighted(values, weights)  # type: ignore[assignment]
            else:
                final = prim.aggregate(values)  # type: ignore[assignment]

        # Tail humility (--fc tail=<weight>): mix continuous CDFs with a
        # widened copy, floor MC options. Insurance against overconfident
        # misses; no-op for binary (already clamped) and unknown types.
        tail_weight = float(fc.get("tail") or 0.0)
        if tail_weight > 0:
            scaling = getattr(question, "scaling", None)
            # Live continuous questions always carry scaling; the getattr
            # defaults are only a safety net, and they fail closed on purpose.
            # Treating a bound as closed at worst relaxes an open reserve;
            # treating a closed bound as open puts mass past it and gets the
            # submission hard-rejected (the Halo failure mode).
            final = apply_tail_humility(
                final,
                str(qtype_value),
                tail_weight,
                lower_open=getattr(scaling, "open_lower_bound", False),
                upper_open=getattr(scaling, "open_upper_bound", False),
            )

        comment = self.format(
            question,
            ctx,
            [(s.forecast, s.rationale) for s in results],
            clean,
            final,  # type: ignore[arg-type]
            prim,
        )
        return StrategyResult(
            forecast=final,  # type: ignore[arg-type]
            comment=comment,
            ctx=ctx,
            raw_results=[(s.forecast, s.rationale) for s in results],
            sample_models=_sample_model_list(results),
            health=health,
        )

    @abstractmethod
    async def produce(
        self,
        question: Any,
        ctx: dict[str, Any],
        deps: ParseDeps,
        config: BotConfig,
        prim: Primitive[T],
    ) -> list[Sample[T]]:
        """Generate raw forecast samples. The strategy's core logic."""
        ...

    def format(
        self,
        question: Any,
        ctx: dict[str, Any],
        results: list[tuple[T, str]],
        clean: list[T],
        final: T,
        prim: Primitive[T],
    ) -> str:
        """Override to customize comment output."""
        sections = "\n".join(
            f"## Rationale {i + 1}\n{r.strip()}" for i, (_, r) in enumerate(results)
        )
        return f"{format_forecast(final, question)}\n\n{sections}"


class Ensemble(Strategy[T]):
    """Run N identical prompts in parallel, parse each, return raw results.

    When ``forecast_config`` contains a ``mix`` key (e.g.
    ``mix=openai:2,anthropic:2,gemini:1``) samples are distributed across
    provider families.  Otherwise all samples go through the default
    ``"forecast"`` purpose.
    """

    async def produce(
        self,
        question: Any,
        ctx: dict[str, Any],
        deps: ParseDeps,
        config: BotConfig,
        prim: Primitive[T],
    ) -> list[Sample[T]]:
        prompt = prim.prompt_template.format(**ctx)
        logger.debug("Forecast prompt:\n%s", prompt)
        fc = config.forecast_config
        mix_spec: str | None = fc.get("mix")

        if mix_spec is not None:
            plan = _parse_mix(mix_spec)
        else:
            plan = [("forecast", fc.get("runs", 5), 1.0)]

        return await self._produce_plan(prompt, deps, config, prim, plan)

    async def _produce_plan(
        self,
        prompt: str,
        deps: ParseDeps,
        config: BotConfig,
        prim: Primitive[T],
        plan: list[tuple[str, int, float]],
    ) -> list[Sample[T]]:
        # Validate every family in the plan has a registered provider.
        for purpose, _count, _weight in plan:
            try:
                config.gateway.model_for(purpose)
            except KeyError:
                family = purpose.split(":", 1)[-1]
                raise ValueError(
                    f"Plan references '{family}' but no provider is configured for "
                    f"purpose '{purpose}'. Check *_API_KEY env vars and package installation."
                ) from None

        # Prime caches once per distinct purpose in the plan.
        seen_purposes: set[str] = set()
        for purpose, count, _weight in plan:
            if purpose not in seen_purposes:
                try:
                    await config.gateway.prepare_cache(
                        LLMRequest(prompt=prompt, purpose=purpose), count
                    )
                except Exception:
                    logger.warning(
                        "Cache prime failed for purpose=%s, continuing without cache",
                        purpose,
                        exc_info=True,
                    )
                seen_purposes.add(purpose)

        # Flatten the plan into a per-sample list.
        sample_specs: list[tuple[str, float]] = []
        for purpose, count, weight in plan:
            for _ in range(count):
                sample_specs.append((purpose, weight))

        async def single_run(purpose: str, weight: float) -> Sample[T] | BaseException:
            try:
                rationale = await config.gateway.complete(
                    LLMRequest(prompt=prompt, purpose=purpose)
                )
                forecast = await prim.parse(rationale, deps)
                model = config.gateway.model_for(purpose)
                return Sample(forecast=forecast, rationale=rationale, model=model, weight=weight)
            except Exception as e:
                logger.warning("Forecast sample failed: %s", e, exc_info=False)
                return e

        results = await asyncio.gather(
            *[single_run(p, w) for p, w in sample_specs],
            return_exceptions=True,
        )

        # Expire caches for distinct purposes.
        for purpose in seen_purposes:
            try:
                await config.gateway.expire_cache(LLMRequest(prompt=prompt, purpose=purpose))
            except Exception:
                logger.warning("Cache expire failed for purpose=%s", purpose, exc_info=True)

        good = [r for r in results if not isinstance(r, BaseException)]

        # ── ensemble health ──────────────────────────────────────────
        planned_by_family: dict[str, int] = {}
        survived_by_family: dict[str, int] = {}
        for (purpose, _weight), r in zip(sample_specs, results, strict=True):
            family = purpose.split(":", 1)[-1] if ":" in purpose else "openai"
            planned_by_family[family] = planned_by_family.get(family, 0) + 1
            if not isinstance(r, BaseException):
                survived_by_family[family] = survived_by_family.get(family, 0) + 1

        planned_total = len(sample_specs)
        survived_total = len(good)
        by_family = {
            f: (planned_by_family[f], survived_by_family.get(f, 0)) for f in planned_by_family
        }
        self._last_health = EnsembleHealth(
            planned=planned_total,
            survived=survived_total,
            by_family=by_family,
        )

        if survived_total < planned_total:
            lost_by_family = {
                f: planned_by_family[f] - survived_by_family.get(f, 0)
                for f in planned_by_family
                if planned_by_family[f] > survived_by_family.get(f, 0)
            }
            logger.warning(
                "Ensemble degraded: %d/%d lost, by family: %s",
                planned_total - survived_total,
                planned_total,
                lost_by_family,
            )

        if survived_total < 2:
            raise RuntimeError("Fewer than 2 forecast samples survived")
        return cast("list[Sample[T]]", good)


_strategy_classes: dict[str, type[Strategy[Any]]] = {
    "ensemble": Ensemble,
}


def strategy_for(name: str, *, disable_research: bool = False) -> Strategy[Any]:
    """Return a strategy instance configured for the given research mode."""
    try:
        cls = _strategy_classes[name]
    except KeyError:
        raise ValueError(
            f"Unknown strategy {name!r}. Available strategies: {sorted(_strategy_classes.keys())}"
        ) from None
    research = no_research if disable_research else planned_research
    return cls(research=research)
