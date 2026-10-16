"""Unit tests for forecasting.primitives — parse, aggregate, trim, format."""

from __future__ import annotations

import datetime as dt
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from common.models import ForecastQuestion, QuestionType
from forecasting.primitives import (
    ParseDeps,
    Primitive,
    _align_mcq_options,
    _build_date_bound_messages,
    _cdf_median_position,
    _cdf_quantile,
    _dispersion_binary,
    _dispersion_cdf,
    _dispersion_mcq,
    _mcq_tvd,
    _timestamp_to_date,
    aggregate_array_median,
    aggregate_dict_mean,
    aggregate_scalar_median,
    aggregate_weighted_array_median,
    aggregate_weighted_dict_mean,
    aggregate_weighted_scalar_mean,
    aggregate_weighted_scalar_median,
    format_forecast,
    trim_array_outliers,
    trim_dict_outliers,
    trim_scalar_outliers,
)
from llm.gateway import LLMRequest


class _FakeGateway:
    async def complete(self, request: LLMRequest) -> str:
        return "default"

    async def extract_structured(self, request: LLMRequest, output_type: object) -> object:
        return output_type() if callable(output_type) else output_type  # type: ignore[misc]


class TestParseDeps:
    def test_construction_minimal(self):
        deps = ParseDeps(gateway=_FakeGateway())  # type: ignore[arg-type]
        assert deps.scaling is None
        assert deps.options is None

    def test_construction_with_all_fields(self, linear_scaling):
        deps = ParseDeps(
            gateway=_FakeGateway(),  # type: ignore[arg-type]
            scaling=linear_scaling,
            options=["A", "B"],
        )
        assert deps.scaling is not None
        assert deps.options == ["A", "B"]

    def test_frozen_raises_on_mutation(self):
        deps = ParseDeps(gateway=_FakeGateway())  # type: ignore[arg-type]
        with pytest.raises(FrozenInstanceError):
            deps.gateway = _FakeGateway()  # type: ignore[misc]


class TestPrimitive:
    def test_construction(self):
        prim = Primitive[float](
            prompt_template="test {title}",
            parse=lambda r, d: 0.5,  # type: ignore[arg-type,return-value]
            aggregate=lambda vs: vs[0],
            trim=lambda vs: list(range(len(vs))),
            dispersion=lambda vs: 0.0,  # type: ignore[arg-type]
            supervisor_threshold=0.15,
            output_format_hint="Probability: ZZ%",
        )
        assert prim.prompt_template == "test {title}"
        assert prim.supervisor_threshold == 0.15
        assert prim.output_format_hint == "Probability: ZZ%"

    def test_frozen_raises_on_mutation(self):
        prim = Primitive[float](
            prompt_template="test",
            parse=lambda r, d: 0.5,  # type: ignore[arg-type,return-value]
            aggregate=lambda vs: vs[0],
            trim=lambda vs: list(range(len(vs))),
            dispersion=lambda vs: 0.0,  # type: ignore[arg-type]
            supervisor_threshold=0.15,
            output_format_hint="Probability: ZZ%",
        )
        with pytest.raises(FrozenInstanceError):
            prim.prompt_template = "changed"  # type: ignore[misc]


# ── Scalar trimming ─────────────────────────────────────────────────


class TestTrimScalars:
    def test_empty_returns_empty(self):
        assert trim_scalar_outliers([]) == []

    def test_single_value_returns_same(self):
        assert trim_scalar_outliers([0.5]) == [0]

    def test_three_or_fewer_no_trimming(self):
        values = [0.1, 0.5, 0.9]
        assert trim_scalar_outliers(values) == [0, 1, 2]

    def test_trim_outliers(self):
        values = [0.5, 0.51, 0.49, 0.52, 0.48, 0.1]
        keep = trim_scalar_outliers(values)
        assert 5 not in keep  # index 5 is the outlier 0.1
        assert len(keep) < len(values)
        assert all(v == pytest.approx(values[i]) for i, v in enumerate([values[i] for i in keep]))

    def test_all_identical_no_trimming(self):
        values = [0.5, 0.5, 0.5, 0.5]
        assert trim_scalar_outliers(values) == [0, 1, 2, 3]

    def test_zero_median_distance_no_trimming(self):
        values = [0.5, 0.5, 0.5, 0.50001, 0.50002]
        assert trim_scalar_outliers(values) == list(range(len(values)))

    def test_two_outliers_trimmed(self):
        values = [0.5, 0.51, 0.49, 0.52, 0.48, 0.01, 0.99]
        keep = trim_scalar_outliers(values)
        assert keep == [0, 1, 2, 3, 4]


# ── Array trimming ──────────────────────────────────────────────────


class TestTrimArrays:
    def test_empty_returns_empty(self):
        assert trim_array_outliers([]) == []

    def test_single_array_returns_same(self):
        assert trim_array_outliers([[0.1, 0.2, 0.3]]) == [0]

    def test_three_or_fewer_no_trimming(self):
        arrays = [[0.1, 0.2], [0.15, 0.25], [0.12, 0.22]]
        assert trim_array_outliers(arrays) == [0, 1, 2]

    def test_trim_outliers(self):
        arrays = [
            [0.1, 0.2, 0.3],
            [0.11, 0.21, 0.31],
            [0.12, 0.22, 0.32],
            [0.10, 0.20, 0.29],
            [0.5, 0.6, 0.7],
        ]
        keep = trim_array_outliers(arrays)
        assert len(keep) < len(arrays)
        assert 4 not in keep  # outlier

    def test_all_identical_no_trimming(self):
        arrays = [[0.1, 0.2], [0.1, 0.2], [0.1, 0.2], [0.1, 0.2]]
        assert trim_array_outliers(arrays) == [0, 1, 2, 3]


# ── Dict trimming ───────────────────────────────────────────────────


class TestTrimDicts:
    def test_empty_returns_empty(self):
        assert trim_dict_outliers([]) == []

    def test_single_dict_returns_same(self):
        assert trim_dict_outliers([{"A": 0.3, "B": 0.7}]) == [0]

    def test_three_or_fewer_no_trimming(self):
        dicts = [{"A": 0.3}, {"A": 0.35}, {"A": 0.32}]
        assert trim_dict_outliers(dicts) == [0, 1, 2]

    def test_trim_outliers(self):
        dicts = [
            {"A": 0.3, "B": 0.3, "C": 0.4},
            {"A": 0.32, "B": 0.31, "C": 0.37},
            {"A": 0.28, "B": 0.29, "C": 0.43},
            {"A": 0.34, "B": 0.30, "C": 0.36},
            {"A": 0.8, "B": 0.1, "C": 0.1},
        ]
        keep = trim_dict_outliers(dicts)
        assert len(keep) < len(dicts)
        assert 4 not in keep  # index 4 is the outlier dict

    def test_missing_keys_handled(self):
        dicts = [
            {"A": 0.5},
            {"A": 0.6},
            {"A": 0.4},
            {"A": 0.55},
            {"A": 0.9},
        ]
        keep = trim_dict_outliers(dicts)
        assert len(keep) < len(dicts)
        assert 4 not in keep  # outlier


# ── CDF quantile extraction ─────────────────────────────────────────


class TestCDFQuantile:
    def test_linear_midpoint(self, linear_scaling):
        cdf = [i / 200 for i in range(201)]
        val = _cdf_quantile(cdf, linear_scaling, 0.5)
        assert val == pytest.approx(50.0, rel=1e-2)

    def test_linear_low(self, linear_scaling):
        cdf = [i / 200 for i in range(201)]
        val = _cdf_quantile(cdf, linear_scaling, 0.1)
        assert val == pytest.approx(10.0, rel=1e-2)

    def test_linear_high(self, linear_scaling):
        cdf = [i / 200 for i in range(201)]
        val = _cdf_quantile(cdf, linear_scaling, 0.9)
        assert val == pytest.approx(90.0, rel=1e-2)

    def test_log_scale(self, log_scaling):
        cdf = [i / 200 for i in range(201)]
        val = _cdf_quantile(cdf, log_scaling, 0.5)
        assert log_scaling.range_min < val < log_scaling.range_max

    def test_edge_probability_zero(self, linear_scaling):
        cdf = [0.01] * 200 + [1.0]
        val = _cdf_quantile(cdf, linear_scaling, 0.005)
        assert val == pytest.approx(linear_scaling.range_min, rel=1e-2)

    def test_edge_probability_one(self, linear_scaling):
        cdf = [0.0] * 200 + [0.9]
        val = _cdf_quantile(cdf, linear_scaling, 1.0)
        assert val == pytest.approx(linear_scaling.range_max, rel=1e-2)


# ── Date helpers ────────────────────────────────────────────────────


class TestTimestampToDate:
    def test_known_timestamp(self):
        result = _timestamp_to_date(0)
        assert result == "1970-01-01"

    def test_recent_timestamp(self):
        ts = dt.datetime(2025, 1, 15, tzinfo=dt.UTC).timestamp()
        result = _timestamp_to_date(ts)
        assert result == "2025-01-15"


class TestBuildDateBoundMessages:
    def test_closed_bounds(self, linear_scaling):
        lower, upper = _build_date_bound_messages(linear_scaling)
        assert "can not be later than" in upper
        assert "can not be earlier than" in lower

    def test_open_bounds(self, open_bounds_scaling):
        lower, upper = _build_date_bound_messages(open_bounds_scaling)
        assert "likely not later than" in upper
        assert "likely not earlier than" in lower


# ── Aggregate functions ─────────────────────────────────────────────


class TestAggregateScalarMedian:
    def test_odd_count(self):
        assert aggregate_scalar_median([0.1, 0.5, 0.3]) == 0.3

    def test_even_count(self):
        assert aggregate_scalar_median([0.1, 0.5, 0.2, 0.8]) == 0.35

    def test_single_value(self):
        assert aggregate_scalar_median([0.7]) == 0.7

    def test_negative_values(self):
        assert aggregate_scalar_median([-5.0, -3.0, -1.0]) == -3.0


class TestAggregateArrayMedian:
    def test_column_wise(self):
        result = aggregate_array_median([[0.1, 0.9], [0.3, 0.7], [0.5, 0.5]])
        assert len(result) == 2
        assert result[0] == 0.3
        assert result[1] == 0.7

    def test_single_array_passthrough(self):
        result = aggregate_array_median([[1.0, 2.0, 3.0]])
        assert result == [1.0, 2.0, 3.0]


class TestAggregateDictMean:
    def test_basic(self):
        result = aggregate_dict_mean([{"A": 0.3, "B": 0.7}, {"A": 0.4, "B": 0.6}])
        assert result == pytest.approx({"A": 0.35, "B": 0.65})

    def test_empty(self):
        assert aggregate_dict_mean([]) == {}


# ── Format forecast ─────────────────────────────────────────────────


class TestFormatForecast:
    def test_binary(self, sample_binary_question):
        result = format_forecast(0.73, sample_binary_question)
        assert "73" in result
        assert "%" in result

    def test_numeric(self, sample_numeric_question):
        cdf = [i / 200 for i in range(201)]
        result = format_forecast(cdf, sample_numeric_question)
        assert "Median:" in result
        assert "°C" in result

    def test_numeric_no_unit(self, linear_scaling):
        q = ForecastQuestion(
            title="Test",
            body="Test",
            question_type=QuestionType.NUMERIC,
            scaling=linear_scaling,
        )
        cdf = [i / 200 for i in range(201)]
        result = format_forecast(cdf, q)
        assert "Median:" in result
        assert "°C" not in result

    def test_date(self, sample_date_question):
        cdf = [i / 200 for i in range(201)]
        result = format_forecast(cdf, sample_date_question)
        assert "Median:" in result

    def test_multiple_choice(self, sample_multiple_choice_question):
        probs = {"Democrat": 0.45, "Republican": 0.40, "Other": 0.15}
        result = format_forecast(probs, sample_multiple_choice_question)
        assert "Democrat" in result
        assert "Republican" in result


# ── MCQ option alignment ─────────────────────────────────────────────


class TestAlignMCQOptions:
    def test_exact_match_passthrough(self):
        raw = {"A": 0.3, "B": 0.7}
        assert _align_mcq_options(raw, ["A", "B"]) == raw

    def test_case_and_whitespace_insensitive(self):
        raw = {" democrat ": 0.6, "REPUBLICAN": 0.4}
        result = _align_mcq_options(raw, ["Democrat", "Republican"])
        assert result == {"Democrat": 0.6, "Republican": 0.4}

    def test_fuzzy_match_paraphrase(self):
        raw = {"Yes, it will happen": 0.8, "No": 0.2}
        result = _align_mcq_options(raw, ["Yes", "No"])
        assert result["No"] == 0.2
        assert result["Yes"] == 0.8

    def test_missing_option_filled_at_floor(self):
        raw = {"A": 0.5, "B": 0.5}
        result = _align_mcq_options(raw, ["A", "B", "C"])
        assert result["C"] == 0.001

    def test_extra_extracted_names_dropped(self):
        raw = {"A": 0.5, "B": 0.3, "Hallucinated": 0.2}
        result = _align_mcq_options(raw, ["A", "B"])
        assert set(result) == {"A", "B"}

    def test_no_overlap_raises(self):
        with pytest.raises(ValueError):
            _align_mcq_options({"completely": 0.5, "unrelated": 0.5}, ["Alpha", "Beta"])

    def test_output_keyed_in_question_order(self):
        raw = {"B": 0.7, "A": 0.3}
        result = _align_mcq_options(raw, ["A", "B"])
        assert list(result) == ["A", "B"]

    def test_lettered_placeholders_mapped_by_position(self):
        # The live 2026-06-29 failure: models copied the prompt's format
        # illustration verbatim for numeric-range options.
        options = [
            "Less than or equal to 45",
            ">45 and <56",
            ">55 and <66",
            ">65 and <76",
            ">75 and <86",
            ">85",
        ]
        probs = [0.05, 0.1, 0.3, 0.35, 0.15, 0.05]
        raw = {f"Option_{letter}": p for letter, p in zip("ABCDEF", probs, strict=True)}
        result = _align_mcq_options(raw, options)
        assert result[">55 and <66"] == 0.3
        assert result[">85"] == 0.05
        assert list(result) == options

    def test_placeholder_variants(self):
        options = ["Under 10", "10 to 20", "Over 20"]
        for raw in (
            {"option a": 0.2, "option b": 0.5, "option c": 0.3},
            {"(A)": 0.2, "(B)": 0.5, "(C)": 0.3},
            {"Option-A": 0.2, "Option B": 0.5, "option_C": 0.3},
        ):
            result = _align_mcq_options(raw, options)
            assert result == {"Under 10": 0.2, "10 to 20": 0.5, "Over 20": 0.3}

    def test_real_names_win_over_placeholder_interpretation(self):
        # Options that ARE bare letters must keep exact-match semantics:
        # "B" maps to option "B", not positionally to options[1].
        raw = {"B": 0.9, "A": 0.1}
        result = _align_mcq_options(raw, ["B", "A"])
        assert result == {"B": 0.9, "A": 0.1}

    def test_mixed_exact_and_placeholder(self):
        options = ["Red", "Green", "Blue"]
        raw = {"Red": 0.5, "Option_B": 0.3, "Option_C": 0.2}
        result = _align_mcq_options(raw, options)
        assert result == {"Red": 0.5, "Green": 0.3, "Blue": 0.2}

    def test_placeholder_beyond_option_count_ignored(self):
        raw = {"Option_A": 0.6, "Option_E": 0.4}
        result = _align_mcq_options(raw, ["Yes", "No"])
        assert result["Yes"] == 0.6
        assert result["No"] == 0.001


# ── CDF median position ──────────────────────────────────────────────


class TestCDFMedianPosition:
    def test_uniform_cdf(self):
        cdf = [i / 200 for i in range(201)]
        assert _cdf_median_position(cdf) == pytest.approx(0.5)

    def test_flat_through_median(self):
        cdf = [0.4] * 100 + [0.6] * 101
        assert _cdf_median_position(cdf) == pytest.approx(99.5 / 200)

    def test_single_element(self):
        assert _cdf_median_position([0.5]) == 0.5

    def test_median_at_start(self):
        cdf = [0.6] * 201
        assert _cdf_median_position(cdf) == 0.0

    def test_never_reaches_half(self):
        cdf = [0.4] * 201
        assert _cdf_median_position(cdf) == 1.0

    def test_interpolation_between_indices(self):
        cdf = [0.0] * 100 + [0.499] + [0.501] + [0.8] * 99
        pos = _cdf_median_position(cdf)
        assert 0.49 < pos < 0.51

    def test_exact_half_multiple_equal(self):
        n = 200
        cdf = [0.0] * 80 + [0.5, 0.5, 0.5] + [0.9] * (n - 83)
        pos = _cdf_median_position(cdf)
        assert pos == pytest.approx(80.0 / (n - 1))


# ── Dispersion functions ─────────────────────────────────────────────


class TestDispersionBinary:
    def test_empty_returns_zero(self):
        assert _dispersion_binary([]) == pytest.approx(0.0)

    def test_single_returns_zero(self):
        assert _dispersion_binary([0.5]) == pytest.approx(0.0)

    def test_identical_no_dispersion(self):
        assert _dispersion_binary([0.3, 0.3, 0.3]) == pytest.approx(0.0)

    def test_divergent(self):
        disp = _dispersion_binary([0.1, 0.5, 0.9])
        assert disp > 0.3  # large spread → high std

    def test_tight_cluster(self):
        disp = _dispersion_binary([0.48, 0.50, 0.52])
        assert disp < 0.1

    def test_edge_range(self):
        disp = _dispersion_binary([0.0, 1.0])
        assert disp >= 0.5  # max possible std for two points


class TestDispersionCDF:
    @staticmethod
    def _uniform_cdf(center: float = 0.5) -> list[float]:
        return [min(1.0, max(0.0, i / 200 + (center - 0.5))) for i in range(201)]

    def test_empty_returns_zero(self):
        assert _dispersion_cdf([]) == pytest.approx(0.0)

    def test_single_returns_zero(self):
        assert _dispersion_cdf([self._uniform_cdf()]) == pytest.approx(0.0)

    def test_identical_cdfs_no_dispersion(self):
        cdf = self._uniform_cdf()
        assert _dispersion_cdf([cdf, cdf, cdf]) == pytest.approx(0.0, abs=1e-6)

    def test_divergent_medians(self):
        a = self._uniform_cdf(0.3)
        b = self._uniform_cdf(0.5)
        c = self._uniform_cdf(0.7)
        disp = _dispersion_cdf([a, b, c])
        assert 0.15 < disp < 0.25  # positions ~0.3, 0.5, 0.7 → std ~0.163

    def test_two_cdfs(self):
        a = self._uniform_cdf(0.35)
        b = self._uniform_cdf(0.65)
        disp = _dispersion_cdf([a, b])
        assert disp > 0.1


class TestMCQTVD:
    def test_identical(self):
        assert _mcq_tvd({"A": 0.5, "B": 0.5}, {"A": 0.5, "B": 0.5}) == 0.0

    def test_completely_different(self):
        assert _mcq_tvd({"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0}) == 1.0

    def test_partial_overlap(self):
        tvd = _mcq_tvd({"A": 0.6, "B": 0.4}, {"A": 0.4, "B": 0.6})
        assert tvd == pytest.approx(0.2)

    def test_different_option_sets(self):
        tvd = _mcq_tvd({"A": 0.7, "B": 0.3}, {"A": 0.7, "C": 0.3})
        assert tvd == pytest.approx(0.3)

    def test_three_options(self):
        tvd = _mcq_tvd({"A": 0.5, "B": 0.3, "C": 0.2}, {"A": 0.4, "B": 0.4, "C": 0.2})
        assert tvd == pytest.approx(0.1)


class TestDispersionMCQ:
    def test_empty_returns_zero(self):
        assert _dispersion_mcq([]) == pytest.approx(0.0)

    def test_single_returns_zero(self):
        assert _dispersion_mcq([{"A": 0.5, "B": 0.5}]) == pytest.approx(0.0)

    def test_identical_no_dispersion(self):
        d = {"A": 0.5, "B": 0.3, "C": 0.2}
        assert _dispersion_mcq([d, d, d]) == pytest.approx(0.0)

    def test_flipped_winners(self):
        values = [{"A": 0.6, "B": 0.4}, {"A": 0.4, "B": 0.6}, {"A": 0.5, "B": 0.5}]
        disp = _dispersion_mcq(values)
        assert disp > 0.05  # catches the disagreement

    def test_divergent_confidence(self):
        values = [{"A": 0.4, "B": 0.6}, {"A": 0.3, "B": 0.7}, {"A": 0.5, "B": 0.5}]
        disp = _dispersion_mcq(values)
        assert disp > 0.05


# ── Weighted aggregators ──────────────────────────────────────────────


class TestWeightedScalarMedian:
    def test_uniform_weights_equals_unweighted(self):
        values = [0.1, 0.3, 0.5, 0.7, 0.9]
        weights = [1.0] * 5
        assert aggregate_weighted_scalar_median(values, weights) == pytest.approx(
            aggregate_scalar_median(values)
        )

    def test_weighted_skews_toward_heavy(self):
        values = [0.1, 0.5, 0.9]
        assert aggregate_weighted_scalar_median(values, [100.0, 1.0, 1.0]) == pytest.approx(0.1)
        assert aggregate_weighted_scalar_median(values, [1.0, 100.0, 1.0]) == pytest.approx(0.5)
        assert aggregate_weighted_scalar_median(values, [1.0, 1.0, 100.0]) == pytest.approx(0.9)

    def test_weighted_odd_count(self):
        values = [0.2, 0.4, 0.6]
        weights = [2.0, 1.0, 1.0]
        result = aggregate_weighted_scalar_median(values, weights)
        assert 0.2 <= result <= 0.4

    def test_weighted_interpolation(self):
        values = [0.0, 1.0]
        weights = [1.0, 1.0]
        result = aggregate_weighted_scalar_median(values, weights)
        assert result == pytest.approx(0.5)

    def test_all_zero_weights_falls_back_to_median(self):
        values = [0.3, 0.5, 0.7]
        assert aggregate_weighted_scalar_median(values, [0.0, 0.0, 0.0]) == 0.5

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aggregate_weighted_scalar_median([], [])

    def test_weighted_median_equivalent_to_median_uniform(self):
        import random

        random.seed(0)
        for _ in range(20):
            vals = [random.random() for _ in range(7)]
            assert aggregate_weighted_scalar_median(vals, [1.0] * 7) == pytest.approx(
                aggregate_scalar_median(vals)
            )


class TestWeightedArrayMedian:
    def test_uniform_weights_equals_unweighted(self):
        arrays = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]]
        weights = [1.0, 1.0, 1.0]
        result = aggregate_weighted_array_median(arrays, weights)
        expected = aggregate_array_median(arrays)
        assert result == pytest.approx(expected)

    def test_weighted_cdf(self):
        arrays = [
            [0.0, 0.1, 0.2],
            [0.5, 0.6, 0.7],
            [0.8, 0.9, 1.0],
        ]
        # Heavy weight on first → result closer to first
        result = aggregate_weighted_array_median(arrays, [100.0, 1.0, 1.0])
        assert result[0] == pytest.approx(0.0)
        assert result[2] == pytest.approx(0.2)


class TestWeightedDictMean:
    def test_uniform_weights_equals_unweighted(self):
        dicts = [{"A": 0.5, "B": 0.5}, {"A": 0.3, "B": 0.7}]
        weights = [1.0, 1.0]
        result = aggregate_weighted_dict_mean(dicts, weights)
        expected = aggregate_dict_mean(dicts)
        for k in result:
            assert result[k] == pytest.approx(expected[k])

    def test_weighted_skews(self):
        dicts = [{"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0}]
        # Heavy weight on first
        result = aggregate_weighted_dict_mean(dicts, [10.0, 1.0])
        assert result["A"] > 0.9
        assert result["B"] < 0.1

    def test_all_zero_weights_falls_back(self):
        dicts = [{"A": 0.5, "B": 0.5}, {"A": 0.3, "B": 0.7}]
        result = aggregate_weighted_dict_mean(dicts, [0.0, 0.0])
        expected = aggregate_dict_mean(dicts)
        for k in result:
            assert result[k] == pytest.approx(expected[k])

    def test_empty_returns_empty(self):
        assert aggregate_weighted_dict_mean([], []) == {}


class TestWeightedScalarMean:
    def test_uniform_weights_equals_simple_mean(self):
        values = [0.1, 0.5, 0.9]
        assert aggregate_weighted_scalar_mean(values, [1.0, 1.0, 1.0]) == pytest.approx(
            np.mean(values)
        )

    def test_weighted_mean_skews_toward_heavy(self):
        values = [0.0, 1.0]
        assert aggregate_weighted_scalar_mean(values, [10.0, 1.0]) == pytest.approx(1.0 / 11.0)
        assert aggregate_weighted_scalar_mean(values, [1.0, 10.0]) > 0.9

    def test_zero_weights_falls_back_to_unweighted_mean(self):
        values = [0.2, 0.5, 0.8]
        result = aggregate_weighted_scalar_mean(values, [0.0, 0.0, 0.0])
        assert result == pytest.approx(float(np.mean(values)))

    def test_single_value(self):
        assert aggregate_weighted_scalar_mean([0.42], [0.5]) == pytest.approx(0.42)

    def test_weighted_mean_equivalent_to_mean_uniform(self):
        import random

        random.seed(1)
        for _ in range(20):
            vals = [random.random() for _ in range(7)]
            assert aggregate_weighted_scalar_mean(vals, [1.0] * 7) == pytest.approx(
                float(np.mean(vals))
            )
