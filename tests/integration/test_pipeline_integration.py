"""Integration tests for forecasting — end-to-end with mocked LLM."""

from __future__ import annotations

from dataclasses import replace
from unittest import mock

import pytest

from forecasting.extraction import (
    BinaryExtraction,
    DateExtraction,
    NumericExtraction,
    OptionPrediction,
)
from forecasting.primitives import PRIMITIVES
from forecasting.strategies import strategy_for

# ── Ensemble strategy ──────────────────────────────────────────────


class TestEnsembleStrategy:
    async def test_all_runs_succeed(self, binary_question_from_api, llm_config):
        call_count = 0

        async def fake_complete(request):
            nonlocal call_count
            call_count += 1
            return f"run-{call_count} rationale\nProbability: 73%"

        async def fake_extract(request, output_type):
            return BinaryExtraction(probability_percent=73)

        from tests.integration.conftest import FakeLLMGateway

        llm_config = replace(llm_config, forecast_config={"runs": 3})
        llm_config = replace(llm_config, disable_research=True)
        llm_config = replace(
            llm_config, gateway=FakeLLMGateway(complete_fn=fake_complete, extract_fn=fake_extract)
        )

        result = await strategy_for("ensemble", disable_research=True).run(
            binary_question_from_api, llm_config, PRIMITIVES["binary"]
        )
        assert result.forecast == pytest.approx(0.73)

    async def test_some_runs_fail(self, binary_question_from_api, llm_config):
        call_count = 0

        async def fake_complete(request):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("simulated failure")
            return f"run-{call_count} rationale\nProbability: 50%"

        async def fake_extract(request, output_type):
            return BinaryExtraction(probability_percent=50)

        from tests.integration.conftest import FakeLLMGateway

        llm_config = replace(llm_config, forecast_config={"runs": 3})
        llm_config = replace(llm_config, disable_research=True)
        llm_config = replace(
            llm_config, gateway=FakeLLMGateway(complete_fn=fake_complete, extract_fn=fake_extract)
        )

        result = await strategy_for("ensemble", disable_research=True).run(
            binary_question_from_api, llm_config, PRIMITIVES["binary"]
        )
        assert result.forecast == pytest.approx(0.5)
        assert "run-1" in result.comment
        assert "run-3" in result.comment

    async def test_all_runs_fail_raises(self, binary_question_from_api, llm_config):
        async def fake_complete(request):
            raise RuntimeError("all fail")

        async def fake_extract(request, output_type):
            return BinaryExtraction(probability_percent=50)

        from tests.integration.conftest import FakeLLMGateway

        llm_config = replace(llm_config, forecast_config={"runs": 2})
        llm_config = replace(llm_config, disable_research=True)
        llm_config = replace(
            llm_config, gateway=FakeLLMGateway(complete_fn=fake_complete, extract_fn=fake_extract)
        )

        with pytest.raises(RuntimeError, match="Fewer than 2 forecast samples survived"):
            await strategy_for("ensemble", disable_research=True).run(
                binary_question_from_api, llm_config, PRIMITIVES["binary"]
            )

    async def test_outlier_trimming(self, binary_question_from_api, llm_config):
        values = iter([0.5, 0.51, 0.49, 0.52, 0.01])

        async def fake_complete(request):
            v = next(values)
            return f"rationale\nProbability: {int(v * 100)}%"

        async def fake_extract(request, output_type):
            import re

            text = request.prompt
            m = re.search(r"Probability:\s*(\d+)", text)
            pct = int(m.group(1)) if m else 50
            return BinaryExtraction(probability_percent=pct)

        from tests.integration.conftest import FakeLLMGateway

        llm_config = replace(llm_config, forecast_config={"runs": 5})
        llm_config = replace(llm_config, disable_research=True)
        llm_config = replace(
            llm_config, gateway=FakeLLMGateway(complete_fn=fake_complete, extract_fn=fake_extract)
        )

        result = await strategy_for("ensemble", disable_research=True).run(
            binary_question_from_api, llm_config, PRIMITIVES["binary"]
        )
        assert 0.49 <= result.forecast <= 0.52


# ── forecast_binary via Ensemble ────────────────────────────────────


class TestForecastBinaryIntegration:
    async def test_end_to_end(self, binary_question_from_api, llm_config, patch_forecast_deps):
        async def fake_call_llm(request):
            return "Based on analysis, I estimate the probability at 60%.\n\nProbability: 60%"

        async def fake_extract(request, output_type):
            return BinaryExtraction(probability_percent=60)

        llm_config = replace(llm_config, disable_research=True)

        with patch_forecast_deps(call_llm=fake_call_llm, extract=fake_extract) as gateway:
            llm_config = replace(llm_config, gateway=gateway)
            result = await strategy_for("ensemble", disable_research=True).run(
                binary_question_from_api, llm_config, PRIMITIVES["binary"]
            )
        assert 0.0 <= result.forecast <= 1.0
        assert isinstance(result.forecast, float)


# ── forecast_numeric via Ensemble ───────────────────────────────────


class TestForecastNumericIntegration:
    async def test_end_to_end(self, numeric_question_from_api, llm_config, patch_forecast_deps):
        percentile_text = (
            "Percentile 10: -30\n"
            "Percentile 20: -25\n"
            "Percentile 40: -20\n"
            "Percentile 60: -15\n"
            "Percentile 80: -10\n"
            "Percentile 90: -5"
        )

        async def fake_call_llm(prompt, **kwargs):
            return f"Analysis here.\n\n{percentile_text}"

        async def fake_extract(text, output_type, **kwargs):
            return NumericExtraction(
                percentile_10=-30,
                percentile_20=-25,
                percentile_40=-20,
                percentile_60=-15,
                percentile_80=-10,
                percentile_90=-5,
            )

        llm_config = replace(llm_config, disable_research=True)

        with patch_forecast_deps(call_llm=fake_call_llm, extract=fake_extract) as gateway:
            llm_config = replace(llm_config, gateway=gateway)
            result = await strategy_for("ensemble", disable_research=True).run(
                numeric_question_from_api, llm_config, PRIMITIVES["numeric"]
            )
        assert len(result.forecast) == 201
        assert all(0.0 <= v <= 1.0 for v in result.forecast)

    async def test_cdf_monotonic(self, numeric_question_from_api, llm_config, patch_forecast_deps):
        s = numeric_question_from_api.scaling
        rmin, rmax = s.range_min, s.range_max
        vals = [rmin + (rmax - rmin) * p / 100 for p in [10, 20, 40, 60, 80, 90]]

        async def fake_call_llm(prompt, **kwargs):
            return f"Percentile 10: {vals[0]:.1f}, Percentile 20: {vals[1]:.1f}..."

        async def fake_extract(text, output_type, **kwargs):
            return NumericExtraction(
                percentile_10=vals[0],
                percentile_20=vals[1],
                percentile_40=vals[2],
                percentile_60=vals[3],
                percentile_80=vals[4],
                percentile_90=vals[5],
            )

        llm_config = replace(llm_config, disable_research=True)

        with patch_forecast_deps(call_llm=fake_call_llm, extract=fake_extract) as gateway:
            llm_config = replace(llm_config, gateway=gateway)
            result = await strategy_for("ensemble", disable_research=True).run(
                numeric_question_from_api, llm_config, PRIMITIVES["numeric"]
            )
        cdf = result.forecast
        for i in range(len(cdf) - 1):
            assert cdf[i] <= cdf[i + 1]


# ── forecast_date via Ensemble ──────────────────────────────────────


class TestForecastDateIntegration:
    async def test_end_to_end(self, date_question_from_api, llm_config, patch_forecast_deps):
        async def fake_call_llm(request):
            return (
                "Percentile 10: 2025-06-01\n"
                "Percentile 20: 2025-09-01\n"
                "Percentile 40: 2026-01-01\n"
                "Percentile 60: 2026-06-01\n"
                "Percentile 80: 2026-12-01\n"
                "Percentile 90: 2027-06-01"
            )

        async def fake_extract(request, output_type):
            return DateExtraction(
                percentile_10="2025-06-01",
                percentile_20="2025-09-01",
                percentile_40="2026-01-01",
                percentile_60="2026-06-01",
                percentile_80="2026-12-01",
                percentile_90="2027-06-01",
            )

        llm_config = replace(llm_config, disable_research=True)

        with patch_forecast_deps(call_llm=fake_call_llm, extract=fake_extract) as gateway:
            llm_config = replace(llm_config, gateway=gateway)
            result = await strategy_for("ensemble", disable_research=True).run(
                date_question_from_api, llm_config, PRIMITIVES["date"]
            )
        assert len(result.forecast) == 201


# ── forecast_multiple_choice via Ensemble ───────────────────────────


class TestForecastMultipleChoiceIntegration:
    async def test_end_to_end(self, mc_question_from_api, llm_config, patch_forecast_deps):
        options = mc_question_from_api.options or ["A", "B"]
        option_probs = {opt: 1.0 / len(options) for opt in options}

        async def fake_call_llm(request):
            lines = "\n".join(f"{opt}: {option_probs[opt]}" for opt in options)
            return lines

        async def fake_extract(request, output_type):
            return [OptionPrediction(name=opt, probability=option_probs[opt]) for opt in options]

        llm_config = replace(llm_config, disable_research=True)

        with patch_forecast_deps(call_llm=fake_call_llm, extract=fake_extract) as gateway:
            llm_config = replace(llm_config, gateway=gateway)
            result = await strategy_for("ensemble", disable_research=True).run(
                mc_question_from_api, llm_config, PRIMITIVES["multiple_choice"]
            )
        assert isinstance(result.forecast, dict)
        assert sum(result.forecast.values()) == pytest.approx(1.0, rel=1e-2)


# ── research context building ──────────────────────────────────────


class TestBuildResearchContext:
    async def test_research_disabled_returns_basic_context(
        self, binary_question_from_api, llm_config
    ):
        s = strategy_for("ensemble", disable_research=True)
        ctx = await s._research(binary_question_from_api, llm_config)
        assert "title" in ctx
        assert ctx.get("summary_report", "") == ""

    async def test_missing_api_key_raises(self, binary_question_from_api, llm_config):
        from research.strategies import _fallback_research

        with mock.patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="EXA_API_KEY not set"):
                await _fallback_research(binary_question_from_api, llm_config)

    async def test_with_exa_and_decompose_llm(
        self, binary_question_from_api, llm_config, mock_exa_search
    ):
        from research.strategies import _fallback_research
        from tests.integration.conftest import FakeLLMGateway

        async def fake_decompose(request):
            return "query one\nquery two"

        llm_config = replace(
            llm_config,
            disable_research=False,
            gateway=FakeLLMGateway(complete_fn=fake_decompose),
        )

        with mock.patch.dict("os.environ", {"EXA_API_KEY": "test-key"}):
            ctx = await _fallback_research(binary_question_from_api, llm_config)

        assert "summary_report" in ctx
        report = ctx["summary_report"]
        assert "Result 1" in report
        assert "https://example.com/a" in report

    async def test_research_failure_propagates(
        self, binary_question_from_api, llm_config, mock_exa_search
    ):
        from research.strategies import _fallback_research
        from tests.integration.conftest import FakeLLMGateway

        async def fake_decompose(request):
            return "search query"

        mock_exa_search.return_value.search.side_effect = RuntimeError("Exa down")

        llm_config = replace(
            llm_config,
            disable_research=False,
            gateway=FakeLLMGateway(complete_fn=fake_decompose),
        )

        with mock.patch.dict("os.environ", {"EXA_API_KEY": "test-key"}):
            ctx = await _fallback_research(binary_question_from_api, llm_config)

        assert "summary_report" in ctx
        report = ctx["summary_report"]
        assert "Research failed" in report or "Proceeding without" in report

    async def test_forecast_binary_with_full_research_pipeline(
        self, binary_question_from_api, llm_config, mock_exa_search, patch_forecast_deps
    ):
        llm_config = replace(llm_config, disable_research=False)

        async def fake_call_llm(request):
            return "Probability: 60%"

        async def fake_extract(request, output_type):
            from research.query_plan import QueryPlan

            if output_type is QueryPlan:
                return QueryPlan(
                    clause_queries=["resolution criterion query"],
                    recency_queries=[],
                    base_rate_queries=[],
                )
            return BinaryExtraction(probability_percent=60)

        with mock.patch.dict("os.environ", {"EXA_API_KEY": "test-key"}):
            with patch_forecast_deps(call_llm=fake_call_llm, extract=fake_extract) as gateway:
                llm_config = replace(llm_config, gateway=gateway)
            result = await strategy_for("ensemble").run(
                binary_question_from_api, llm_config, PRIMITIVES["binary"]
            )

        assert 0.0 <= result.forecast <= 1.0
        assert isinstance(result.forecast, float)
