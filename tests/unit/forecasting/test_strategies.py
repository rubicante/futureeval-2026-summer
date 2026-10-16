"""Unit tests for forecasting.strategies — Strategy ABC, Ensemble."""

from __future__ import annotations

import pytest

from common.config import BotConfig
from forecasting.primitives import (
    PRIMITIVES,
    Primitive,
    aggregate_scalar_median,
    aggregate_weighted_scalar_mean,
    trim_scalar_outliers,
)
from forecasting.strategies import (
    Ensemble,
    Sample,
    Strategy,
    _parse_mix,
    strategy_for,
)


class TestStrategyABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Strategy()  # type: ignore[abstract]

    def test_ensemble_is_strategy(self):
        assert isinstance(strategy_for("ensemble"), Strategy)


class TestEnsemble:
    def test_instantiates(self):
        s = strategy_for("ensemble")
        assert s is not None

    def test_produce_is_abstract_method(self):
        assert callable(strategy_for("ensemble").produce)

    def test_format_default(self, sample_binary_question):
        s = strategy_for("ensemble")
        result = s.format(
            question=sample_binary_question,
            ctx={},
            results=[(0.5, "rationale 1"), (0.6, "rationale 2")],
            clean=[0.5, 0.6],
            final=0.55,
            prim=PRIMITIVES["binary"],
        )
        assert "## Rationale 1" in result
        assert "## Rationale 2" in result


class TestStrategyFactory:
    def test_ensemble_resolves(self):
        assert isinstance(strategy_for("ensemble"), Ensemble)

    async def test_disable_research_injects_noop(self, sample_binary_question):
        s = strategy_for("ensemble", disable_research=True)
        assert s._research is not None
        ctx = await s._research(sample_binary_question, None)  # type: ignore[arg-type]
        assert "title" in ctx
        assert ctx.get("summary_report", "") == ""


# ── Sample dataclass ───────────────────────────────────────────────────


class TestSample:
    def test_construction_defaults(self):
        s = Sample(forecast=0.7, rationale="because")
        assert s.forecast == 0.7
        assert s.rationale == "because"
        assert s.model is None
        assert s.weight == 1.0

    def test_construction_with_model_and_weight(self):
        s = Sample(forecast=0.7, rationale="x", model="gpt-5.5", weight=0.5)
        assert s.model == "gpt-5.5"
        assert s.weight == 0.5

    def test_frozen_raises_on_mutation(self):
        from dataclasses import FrozenInstanceError

        s = Sample(forecast=0.7, rationale="x")
        with pytest.raises(FrozenInstanceError):
            s.rationale = "changed"  # type: ignore[misc]

    def test_generic_with_cdf(self):
        s = Sample(forecast=[0.1, 0.2, 0.3], rationale="cdf rationale")
        assert s.forecast == [0.1, 0.2, 0.3]


# ── _parse_mix ─────────────────────────────────────────────────────────


class TestParseMix:
    def test_parse_equal_weights(self):
        plan = _parse_mix("openai:2,anthropic:2,gemini:1")
        assert plan == [
            ("forecast", 2, 1.0),
            ("forecast:anthropic", 2, 1.0),
            ("forecast:gemini", 1, 1.0),
        ]

    def test_parse_weighted(self):
        plan = _parse_mix("openai:2@1.0,anthropic:2@1.0,gemini:1@0.5")
        assert plan == [
            ("forecast", 2, 1.0),
            ("forecast:anthropic", 2, 1.0),
            ("forecast:gemini", 1, 0.5),
        ]

    def test_parse_single_family(self):
        plan = _parse_mix("openai:5")
        assert plan == [("forecast", 5, 1.0)]

    def test_parse_default_weight_is_one(self):
        plan = _parse_mix("openai:1,gemini:3")
        assert plan[0][2] == 1.0
        assert plan[1][2] == 1.0

    def test_parse_weight_with_whitespace(self):
        plan = _parse_mix(" openai:2@1.0 , anthropic:1@0.75 ")
        assert plan == [
            ("forecast", 2, 1.0),
            ("forecast:anthropic", 1, 0.75),
        ]

    def test_parse_empty_raises(self):
        with pytest.raises(ValueError, match="Empty or unparseable"):
            _parse_mix("")


# ── StrategyResult with sample_models ──────────────────────────────────


class TestStrategyResult:
    def test_no_sample_models_by_default(self) -> None:
        from forecasting.strategies import StrategyResult

        sr: StrategyResult[float] = StrategyResult(
            forecast=0.5, comment="c", ctx={}, raw_results=[(0.5, "r")]
        )
        assert sr.sample_models is None

    def test_sample_models_present(self) -> None:
        from forecasting.strategies import StrategyResult

        sr: StrategyResult[float] = StrategyResult(
            forecast=0.5,
            comment="c",
            ctx={},
            raw_results=[(0.5, "r")],
            sample_models=["gpt-5.5"],
        )
        assert sr.sample_models == ["gpt-5.5"]


# ── agg=mean knob tests ────────────────────────────────────────────────


class _FakeGateway:
    def __init__(self, model: str = "test-model"):
        self._model = model

    def model_for(self, purpose: str) -> str:
        return self._model

    async def complete(self, request):
        return "Rationale\nProbability: 60%"

    async def extract_structured(self, request, output_type):
        return output_type() if callable(output_type) else output_type

    async def prepare_cache(self, request, count=1):
        return

    async def expire_cache(self, request):
        return


class _DummyQuestion:
    question_type = None
    scaling = None
    options = None


class _SimpleStrategy(Strategy[float]):
    """A produce() that always returns the same samples, for testing aggregation."""

    def __init__(self, samples: list[Sample[float]]):
        super().__init__()
        self._samples = samples

    async def produce(self, question, ctx, deps, config, prim):
        return self._samples


@pytest.fixture
def binary_prim() -> Primitive[float]:
    return Primitive[float](
        prompt_template="",
        parse=lambda r, d: 0.5,  # type: ignore[arg-type,return-value]
        aggregate=aggregate_scalar_median,
        trim=trim_scalar_outliers,
        dispersion=lambda vs: 0.0,
        supervisor_threshold=0.15,
        output_format_hint="",
        aggregate_weighted=aggregate_weighted_scalar_mean,
    )


class TestAggKnob:
    async def test_mean_binary(self, binary_prim: Primitive[float]):
        """agg=mean on binary → weighted mean of probabilities."""
        samples = [
            Sample(forecast=0.1, rationale="a", model="m"),
            Sample(forecast=0.3, rationale="b", model="m"),
            Sample(forecast=0.5, rationale="c", model="m"),
        ]
        strat = _SimpleStrategy(samples)
        q = _DummyQuestion()
        q.question_type = type("QT", (), {"value": "binary"})()

        config = BotConfig(
            gateway=_FakeGateway(),
            forecast_config={"agg": "mean"},
        )

        result = await strat.run_from_context(q, {}, config, binary_prim)
        expected = aggregate_weighted_scalar_mean([0.1, 0.3, 0.5], [1.0, 1.0, 1.0])
        assert result.forecast == pytest.approx(expected)

    async def test_median_without_knob(self, binary_prim: Primitive[float]):
        """No agg knob → existing median behavior."""
        samples = [
            Sample(forecast=0.1, rationale="a", model="m"),
            Sample(forecast=0.3, rationale="b", model="m"),
            Sample(forecast=0.9, rationale="c", model="m"),
        ]
        strat = _SimpleStrategy(samples)
        q = _DummyQuestion()
        q.question_type = type("QT", (), {"value": "binary"})()

        config = BotConfig(
            gateway=_FakeGateway(),
            forecast_config={},
        )

        result = await strat.run_from_context(q, {}, config, binary_prim)
        assert result.forecast == 0.3  # median, not mean

    async def test_mean_numeric_falls_back_to_median(self, binary_prim: Primitive[float], caplog):
        """agg=mean on numeric → debug log + fall through to median."""
        import logging

        caplog.set_level(logging.DEBUG)
        samples = [
            Sample(forecast=0.1, rationale="a", model="m"),
            Sample(forecast=0.9, rationale="b", model="m"),
        ]
        strat = _SimpleStrategy(samples)
        q = _DummyQuestion()
        q.question_type = type("QT", (), {"value": "numeric"})()

        config = BotConfig(
            gateway=_FakeGateway(),
            forecast_config={"agg": "mean"},
        )

        result = await strat.run_from_context(q, {}, config, binary_prim)
        assert result.forecast == 0.5  # median, not mean (0.5)
        assert "agg=mean ignored" in caplog.text
        assert "numeric" in caplog.text

    async def test_mean_with_weights(self, binary_prim: Primitive[float]):
        """agg=mean with non-uniform weights → weighted mean."""
        samples = [
            Sample(forecast=0.0, rationale="a", model="m", weight=10.0),
            Sample(forecast=1.0, rationale="b", model="m", weight=1.0),
        ]
        strat = _SimpleStrategy(samples)
        q = _DummyQuestion()
        q.question_type = type("QT", (), {"value": "binary"})()

        config = BotConfig(
            gateway=_FakeGateway(),
            forecast_config={"agg": "mean"},
        )

        result = await strat.run_from_context(q, {}, config, binary_prim)
        assert result.forecast == pytest.approx(1.0 / 11.0)


# ── EnsembleHealth ─────────────────────────────────────────────────────


class TestEnsembleHealth:
    def test_format_all_survived(self):
        from forecasting.strategies import EnsembleHealth

        h = EnsembleHealth(
            planned=5,
            survived=5,
            by_family={"openai": (2, 2), "anthropic": (2, 2), "gemini": (1, 1)},
        )
        line = h.format_one_line()
        assert "openai: 2/2" in line
        assert "anthropic: 2/2" in line
        assert "gemini: 1/1" in line
        assert "lost" not in line

    def test_format_degraded(self):
        from forecasting.strategies import EnsembleHealth

        h = EnsembleHealth(
            planned=5,
            survived=3,
            by_family={"openai": (2, 2), "anthropic": (2, 0), "gemini": (1, 1)},
        )
        line = h.format_one_line()
        assert "openai: 2/2" in line
        assert "anthropic: 0/2" in line
        assert "gemini: 1/1" in line
        assert "(2 lost)" in line


# ── _produce_plan degradation & guard ─────────────────────────────────


class _DegradingGateway:
    """Gateway that fails samples for a specific purpose."""

    def __init__(self, fail_purpose: str | None = None):
        self._fail_purpose = fail_purpose

    def model_for(self, purpose: str) -> str:
        return f"model-{purpose}"

    async def complete(self, request):
        if self._fail_purpose and request.purpose == self._fail_purpose:
            raise RuntimeError("simulated provider failure")
        return "Rationale\nProbability: 60%"

    async def prepare_cache(self, request, count=1):
        return

    async def expire_cache(self, request):
        return


class TestProducePlanDegradation:
    @staticmethod
    def _async_prim() -> Primitive[float]:
        """Return a Primitive whose parse is async (required by _produce_plan)."""

        async def _parse(rationale: str, deps: object) -> float:
            return 0.5

        return Primitive[float](
            prompt_template="",
            parse=_parse,  # type: ignore[arg-type]
            aggregate=aggregate_scalar_median,
            trim=trim_scalar_outliers,
            dispersion=lambda vs: 0.0,
            supervisor_threshold=0.15,
            output_format_hint="",
            aggregate_weighted=aggregate_weighted_scalar_mean,
        )

    async def test_raises_when_fewer_than_two_survive(self):
        """< 2 surviving samples → RuntimeError."""
        from forecasting.strategies import Ensemble

        plan = [("forecast:anthropic", 1, 1.0)]
        s = Ensemble()
        config = BotConfig(
            gateway=_DegradingGateway(fail_purpose="forecast:anthropic"),
        )
        prim = self._async_prim()

        with pytest.raises(RuntimeError, match="Fewer than 2"):
            await s._produce_plan(  # type: ignore[attr-defined]
                "test prompt",
                object(),
                config,
                prim,
                plan,
            )

    async def test_warns_when_degraded(self, caplog):
        """Degraded ensemble → WARNING with correct breakdown."""
        import logging

        from forecasting.strategies import Ensemble

        caplog.set_level(logging.WARNING)
        plan = [
            ("forecast", 2, 1.0),  # openai: 2 planned
            ("forecast:anthropic", 2, 1.0),  # anthropic: 2 planned
            ("forecast:gemini", 1, 1.0),  # gemini: 1 planned
        ]
        s = Ensemble()
        config = BotConfig(
            gateway=_DegradingGateway(fail_purpose="forecast:anthropic"),
        )
        prim = self._async_prim()

        await s._produce_plan(  # type: ignore[attr-defined]
            "test prompt",
            object(),
            config,
            prim,
            plan,
        )

        assert "Ensemble degraded:" in caplog.text
        assert "2/5" in caplog.text
        assert "anthropic" in caplog.text

    async def test_stores_last_health_on_success(self):
        from forecasting.strategies import Ensemble

        plan = [("forecast", 2, 1.0), ("forecast:gemini", 1, 1.0)]
        s = Ensemble()
        config = BotConfig(
            gateway=_DegradingGateway(),
        )
        prim = self._async_prim()

        await s._produce_plan(  # type: ignore[attr-defined]
            "test prompt",
            object(),
            config,
            prim,
            plan,
        )

        health = s._last_health  # type: ignore[attr-defined]
        assert health is not None
        assert health.planned == 3
        assert health.survived == 3

    async def test_stores_last_health_with_partial_failure(self):
        from forecasting.strategies import Ensemble

        plan = [("forecast", 2, 1.0), ("forecast:anthropic", 2, 1.0)]
        s = Ensemble()
        config = BotConfig(
            gateway=_DegradingGateway(fail_purpose="forecast:anthropic"),
        )
        prim = self._async_prim()

        await s._produce_plan(  # type: ignore[attr-defined]
            "test prompt",
            object(),
            config,
            prim,
            plan,
        )

        health = s._last_health  # type: ignore[attr-defined]
        assert health.planned == 4
        assert health.survived == 2
        assert health.by_family["openai"] == (2, 2)
        assert health.by_family["anthropic"] == (2, 0)
