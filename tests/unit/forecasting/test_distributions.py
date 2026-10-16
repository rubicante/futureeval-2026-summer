"""Unit tests for forecasting.distributions — CDF construction and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from forecasting.distributions import (
    NumericDefaults,
    NumericDistribution,
    Percentile,
    generate_continuous_cdf,
    project_cdf_to_valid,
)


def assert_metaculus_valid(cdf, *, open_lower_bound: bool, open_upper_bound: bool) -> None:
    """Assert *cdf* satisfies every Metaculus continuous-CDF constraint."""
    n = len(cdf)
    cap = NumericDefaults.get_max_pmf_value(n, include_wiggle_room=False)
    steps = [cdf[i + 1] - cdf[i] for i in range(n - 1)]
    reserve = NumericDefaults.MIN_OUTSIDE_OPEN
    assert min(steps) >= NumericDefaults.MIN_STEP, f"step below floor: {min(steps):.3e}"
    assert max(steps) <= cap, f"step over cap: {max(steps):.4f} > {cap:.4f}"
    if open_lower_bound:
        assert cdf[0] >= reserve - 1e-12, f"open lower reserves < 0.1%: {cdf[0]:.5f}"
    else:
        assert cdf[0] == 0.0
    if open_upper_bound:
        assert cdf[-1] <= 1.0 - reserve + 1e-12, f"open upper reserves < 0.1%: {1 - cdf[-1]:.5f}"
    else:
        assert cdf[-1] == 1.0


# ── Percentile ──────────────────────────────────────────────────────


class TestPercentile:
    def test_valid_percentile(self):
        p = Percentile(percentile=0.5, value=42.0)
        assert p.percentile == 0.5
        assert p.value == 42.0

    def test_percentile_at_zero(self):
        p = Percentile(percentile=0.0, value=0.0)
        assert p.percentile == 0.0

    def test_percentile_at_one(self):
        p = Percentile(percentile=1.0, value=100.0)
        assert p.percentile == 1.0

    def test_percentile_below_zero_raises(self):
        with pytest.raises((ValueError, ValidationError)):
            Percentile(percentile=-0.1, value=0.0)

    def test_percentile_above_one_raises(self):
        with pytest.raises((ValueError, ValidationError)):
            Percentile(percentile=1.5, value=0.0)

    def test_percentile_nan_raises(self):
        with pytest.raises((ValueError, ValidationError)):
            Percentile(percentile=float("nan"), value=0.0)


# ── NumericDistribution validation ──────────────────────────────────


class TestNumericDistributionValidation:
    def test_non_increasing_percentiles_raises(self, linear_scaling):
        percentiles = [
            Percentile(percentile=0.5, value=50),
            Percentile(percentile=0.3, value=30),
        ]
        with pytest.raises(ValueError, match="strictly increasing"):
            NumericDistribution(
                declared_percentiles=percentiles,
                open_upper_bound=linear_scaling.open_upper_bound,
                open_lower_bound=linear_scaling.open_lower_bound,
                upper_bound=linear_scaling.range_max,
                lower_bound=linear_scaling.range_min,
                zero_point=linear_scaling.zero_point,
            )

    def test_non_increasing_values_raises(self, linear_scaling):
        percentiles = [
            Percentile(percentile=0.3, value=90),
            Percentile(percentile=0.5, value=50),
        ]
        with pytest.raises(ValueError, match="strictly increasing"):
            NumericDistribution(
                declared_percentiles=percentiles,
                open_upper_bound=linear_scaling.open_upper_bound,
                open_lower_bound=linear_scaling.open_lower_bound,
                upper_bound=linear_scaling.range_max,
                lower_bound=linear_scaling.range_min,
                zero_point=linear_scaling.zero_point,
            )

    def test_fewer_than_two_percentiles_raises(self, linear_scaling):
        percentiles = [Percentile(percentile=0.5, value=50)]
        with pytest.raises(ValueError, match="at least 2"):
            NumericDistribution(
                declared_percentiles=percentiles,
                open_upper_bound=linear_scaling.open_upper_bound,
                open_lower_bound=linear_scaling.open_lower_bound,
                upper_bound=linear_scaling.range_max,
                lower_bound=linear_scaling.range_min,
                zero_point=linear_scaling.zero_point,
            )

    def test_lower_bound_leq_zero_point_raises(self):
        with pytest.raises(ValueError, match="greater than the zero point"):
            NumericDistribution(
                declared_percentiles=[
                    Percentile(percentile=0.1, value=20),
                    Percentile(percentile=0.9, value=80),
                ],
                open_upper_bound=False,
                open_lower_bound=False,
                upper_bound=100.0,
                lower_bound=0.0,
                zero_point=1.0,
            )

    def test_value_below_zero_point_raises(self):
        with pytest.raises(ValueError, match="less than the zero point"):
            NumericDistribution(
                declared_percentiles=[
                    Percentile(percentile=0.1, value=-5),
                    Percentile(percentile=0.9, value=80),
                ],
                open_upper_bound=False,
                open_lower_bound=False,
                upper_bound=100.0,
                lower_bound=10.0,
                zero_point=1.0,
            )

    def test_percentile_spacing_too_close_raises(self, linear_scaling):
        percentiles = [
            Percentile(percentile=0.1, value=10),
            Percentile(percentile=0.10000001, value=20),
        ]
        with pytest.raises(ValueError, match="too close"):
            NumericDistribution(
                declared_percentiles=percentiles,
                open_upper_bound=linear_scaling.open_upper_bound,
                open_lower_bound=linear_scaling.open_lower_bound,
                upper_bound=linear_scaling.range_max,
                lower_bound=linear_scaling.range_min,
                zero_point=linear_scaling.zero_point,
            )

    def test_strict_validation_disabled_allows_close_spacing(self, linear_scaling):
        percentiles = [
            Percentile(percentile=0.1, value=10),
            Percentile(percentile=0.10000001, value=20),
        ]
        dist = NumericDistribution(
            declared_percentiles=percentiles,
            open_upper_bound=linear_scaling.open_upper_bound,
            open_lower_bound=linear_scaling.open_lower_bound,
            upper_bound=linear_scaling.range_max,
            lower_bound=linear_scaling.range_min,
            zero_point=linear_scaling.zero_point,
            strict_validation=False,
        )
        assert len(dist.declared_percentiles) == 2

    def test_values_far_exceeding_bounds_raises(self, linear_scaling):
        percentiles = [
            Percentile(percentile=0.1, value=10),
            Percentile(percentile=0.9, value=5000),
        ]
        with pytest.raises(ValueError, match="far exceeding the bounds"):
            NumericDistribution(
                declared_percentiles=percentiles,
                open_upper_bound=linear_scaling.open_upper_bound,
                open_lower_bound=linear_scaling.open_lower_bound,
                upper_bound=linear_scaling.range_max,
                lower_bound=linear_scaling.range_min,
                zero_point=linear_scaling.zero_point,
            )

    def test_no_percentiles_within_wiggle_room_raises(self, linear_scaling):
        percentiles = [
            Percentile(percentile=0.1, value=200),
            Percentile(percentile=0.9, value=300),
        ]
        with pytest.raises(ValueError, match="No declared percentiles are within"):
            NumericDistribution(
                declared_percentiles=percentiles,
                open_upper_bound=linear_scaling.open_upper_bound,
                open_lower_bound=linear_scaling.open_lower_bound,
                upper_bound=linear_scaling.range_max,
                lower_bound=linear_scaling.range_min,
                zero_point=linear_scaling.zero_point,
            )

    def test_distribution_too_tall_raises(self) -> None:
        # build 201 percentiles where one step exceeds the max PMF cap (0.2)
        percentiles: list[Percentile] = []
        for i in range(200):
            percentiles.append(Percentile(percentile=i * 0.004, value=float(i)))
        percentiles.append(Percentile(percentile=1.0, value=200.0))
        with pytest.raises(ValueError, match="too concentrated"):
            NumericDistribution(
                declared_percentiles=percentiles,
                open_upper_bound=True,
                open_lower_bound=True,
                upper_bound=200.0,
                lower_bound=0.0,
                zero_point=None,
                cdf_size=201,
                standardize_cdf=True,
                strict_validation=True,
            )


# ── Repeating values ────────────────────────────────────────────────


class TestRepeatingValues:
    def test_repeating_in_bounds_adjusts(self, linear_scaling):
        percentiles = [
            Percentile(percentile=0.3, value=50),
            Percentile(percentile=0.5, value=50),
            Percentile(percentile=0.7, value=70),
        ]
        dist = NumericDistribution(
            declared_percentiles=percentiles,
            open_upper_bound=linear_scaling.open_upper_bound,
            open_lower_bound=linear_scaling.open_lower_bound,
            upper_bound=linear_scaling.range_max,
            lower_bound=linear_scaling.range_min,
            zero_point=linear_scaling.zero_point,
        )
        values = [p.value for p in dist.declared_percentiles]
        assert len(set(values)) == 3

    def test_repeating_above_bound_adjusts(self, linear_scaling):
        percentiles = [
            Percentile(percentile=0.3, value=50),
            Percentile(percentile=0.5, value=120),
            Percentile(percentile=0.7, value=120),
        ]
        dist = NumericDistribution(
            declared_percentiles=percentiles,
            open_upper_bound=linear_scaling.open_upper_bound,
            open_lower_bound=linear_scaling.open_lower_bound,
            upper_bound=linear_scaling.range_max,
            lower_bound=linear_scaling.range_min,
            zero_point=linear_scaling.zero_point,
        )
        values = [p.value for p in dist.declared_percentiles]
        assert len(set(values)) == 3

    def test_repeating_below_bound_adjusts(self, linear_scaling):
        percentiles = [
            Percentile(percentile=0.3, value=-10),
            Percentile(percentile=0.5, value=-10),
            Percentile(percentile=0.7, value=50),
        ]
        dist = NumericDistribution(
            declared_percentiles=percentiles,
            open_upper_bound=linear_scaling.open_upper_bound,
            open_lower_bound=linear_scaling.open_lower_bound,
            upper_bound=linear_scaling.range_max,
            lower_bound=linear_scaling.range_min,
            zero_point=linear_scaling.zero_point,
        )
        values = [p.value for p in dist.declared_percentiles]
        assert len(set(values)) == 3


# ── CDF generation ──────────────────────────────────────────────────


class TestGetCDF:
    def test_get_cdf_returns_correct_size(self, linear_scaling, valid_percentiles_linear):
        cdf = generate_continuous_cdf(
            valid_percentiles_linear,
            linear_scaling.open_upper_bound,
            linear_scaling.open_lower_bound,
            linear_scaling.range_max,
            linear_scaling.range_min,
            linear_scaling.zero_point,
            linear_scaling.cdf_size,
        )
        assert len(cdf) == 201

    def test_get_cdf_values_in_range(self, linear_scaling, valid_percentiles_linear):
        cdf = generate_continuous_cdf(
            valid_percentiles_linear,
            linear_scaling.open_upper_bound,
            linear_scaling.open_lower_bound,
            linear_scaling.range_max,
            linear_scaling.range_min,
            linear_scaling.zero_point,
            linear_scaling.cdf_size,
        )
        assert all(0.0 <= v <= 1.0 for v in cdf)

    def test_get_cdf_monotonically_increasing(self, linear_scaling, valid_percentiles_linear):
        cdf = generate_continuous_cdf(
            valid_percentiles_linear,
            linear_scaling.open_upper_bound,
            linear_scaling.open_lower_bound,
            linear_scaling.range_max,
            linear_scaling.range_min,
            linear_scaling.zero_point,
            linear_scaling.cdf_size,
        )
        for i in range(len(cdf) - 1):
            assert cdf[i] <= cdf[i + 1], f"Not increasing at index {i}: {cdf[i]} > {cdf[i + 1]}"

    def test_get_cdf_with_wide_range(self, linear_scaling, valid_percentiles_wide):
        scaling = linear_scaling.model_copy(update={"range_min": -1000.0, "range_max": 1000.0})
        cdf = generate_continuous_cdf(
            valid_percentiles_wide,
            scaling.open_upper_bound,
            scaling.open_lower_bound,
            scaling.range_max,
            scaling.range_min,
            scaling.zero_point,
            scaling.cdf_size,
        )
        assert len(cdf) == 201
        assert cdf[0] >= 0
        assert cdf[-1] <= 1.0

    def test_get_cdf_log_scale(self, log_scaling, valid_percentiles_log):
        cdf = generate_continuous_cdf(
            valid_percentiles_log,
            log_scaling.open_upper_bound,
            log_scaling.open_lower_bound,
            log_scaling.range_max,
            log_scaling.range_min,
            log_scaling.zero_point,
            log_scaling.cdf_size,
        )
        assert len(cdf) == 201
        assert all(0.0 <= v <= 1.0 for v in cdf)

    def test_get_cdf_open_bounds(self, open_bounds_scaling, valid_percentiles_linear):
        cdf = generate_continuous_cdf(
            valid_percentiles_linear,
            open_bounds_scaling.open_upper_bound,
            open_bounds_scaling.open_lower_bound,
            open_bounds_scaling.range_max,
            open_bounds_scaling.range_min,
            open_bounds_scaling.zero_point,
            open_bounds_scaling.cdf_size,
        )
        assert len(cdf) == 201
        assert 0.0 < cdf[0] < 0.1, f"Expected some mass below lower bound, got {cdf[0]}"
        assert 0.9 < cdf[-1] < 1.0, f"Expected some mass above upper bound, got {cdf[-1]}"

    def test_get_cdf_narrow_range(self, linear_scaling, valid_percentiles_tight):
        scaling = linear_scaling.model_copy(update={"range_min": 40.0, "range_max": 60.0})
        cdf = generate_continuous_cdf(
            valid_percentiles_tight,
            scaling.open_upper_bound,
            scaling.open_lower_bound,
            scaling.range_max,
            scaling.range_min,
            scaling.zero_point,
            scaling.cdf_size,
        )
        assert len(cdf) == 201

    def test_get_cdf_discrete_cdf_size(self, linear_scaling, valid_percentiles_linear):
        cdf = generate_continuous_cdf(
            valid_percentiles_linear,
            linear_scaling.open_upper_bound,
            linear_scaling.open_lower_bound,
            linear_scaling.range_max,
            linear_scaling.range_min,
            linear_scaling.zero_point,
            cdf_size=51,
        )
        assert len(cdf) == 51


# ── Coordinate conversion ───────────────────────────────────────────


class TestCoordinateConversion:
    def test_nominal_to_cdf_location_linear(self, linear_scaling):
        dist = NumericDistribution(
            declared_percentiles=[
                Percentile(percentile=0.1, value=10),
                Percentile(percentile=0.9, value=90),
            ],
            open_upper_bound=linear_scaling.open_upper_bound,
            open_lower_bound=linear_scaling.open_lower_bound,
            upper_bound=linear_scaling.range_max,
            lower_bound=linear_scaling.range_min,
            zero_point=linear_scaling.zero_point,
        )
        loc = dist._nominal_location_to_cdf_location(50)
        assert loc == pytest.approx(0.5)

    def test_nominal_to_cdf_location_linear_min(self, linear_scaling):
        dist = NumericDistribution(
            declared_percentiles=[
                Percentile(percentile=0.1, value=10),
                Percentile(percentile=0.9, value=90),
            ],
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=100.0,
            lower_bound=0.0,
            zero_point=None,
        )
        loc = dist._nominal_location_to_cdf_location(0)
        assert loc == pytest.approx(0.0)

    def test_nominal_to_cdf_location_linear_max(self):
        dist = NumericDistribution(
            declared_percentiles=[
                Percentile(percentile=0.1, value=10),
                Percentile(percentile=0.9, value=90),
            ],
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=100.0,
            lower_bound=0.0,
            zero_point=None,
        )
        loc = dist._nominal_location_to_cdf_location(100)
        assert loc == pytest.approx(1.0)

    def test_nominal_to_cdf_location_log(self, log_scaling):
        dist = NumericDistribution(
            declared_percentiles=[
                Percentile(percentile=0.1, value=100),
                Percentile(percentile=0.9, value=50000),
            ],
            open_upper_bound=log_scaling.open_upper_bound,
            open_lower_bound=log_scaling.open_lower_bound,
            upper_bound=log_scaling.range_max,
            lower_bound=log_scaling.range_min,
            zero_point=log_scaling.zero_point,
        )
        loc = dist._nominal_location_to_cdf_location(1000)
        assert 0.0 < loc < 1.0

    def test_cdf_location_to_nominal_linear_roundtrip(self, linear_scaling):
        dist = NumericDistribution(
            declared_percentiles=[
                Percentile(percentile=0.1, value=10),
                Percentile(percentile=0.9, value=90),
            ],
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=100.0,
            lower_bound=0.0,
            zero_point=None,
        )
        for x in [0, 25, 50, 75, 100]:
            loc = dist._nominal_location_to_cdf_location(x)
            result = dist._cdf_location_to_nominal_location(loc)
            assert result == pytest.approx(x, rel=1e-6)

    def test_cdf_location_to_nominal_log_roundtrip(self, log_scaling):
        dist = NumericDistribution(
            declared_percentiles=[
                Percentile(percentile=0.1, value=100),
                Percentile(percentile=0.9, value=50000),
            ],
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=100000.0,
            lower_bound=10.0,
            zero_point=1.0,
        )
        for x in [50, 500, 5000, 50000]:
            loc = dist._nominal_location_to_cdf_location(x)
            result = dist._cdf_location_to_nominal_location(loc)
            assert result == pytest.approx(x, rel=1e-5)


# ── Standardization ─────────────────────────────────────────────────


class TestStandardizeCDF:
    def test_standardize_produces_valid_cdf(self, linear_scaling):
        dist = NumericDistribution(
            declared_percentiles=[
                Percentile(percentile=0.1, value=10),
                Percentile(percentile=0.9, value=90),
            ],
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=100.0,
            lower_bound=0.0,
            zero_point=None,
        )
        raw_cdf = dist.get_cdf()
        values = [p.percentile for p in raw_cdf]
        assert len(values) == 201
        # First value at lower bound should be near 0 (closed)
        assert values[0] == pytest.approx(0.0, abs=1e-6)
        assert values[-1] == pytest.approx(1.0, abs=1e-6)

    def test_standardize_open_bounds(self, open_bounds_scaling):
        dist = NumericDistribution(
            declared_percentiles=[
                Percentile(percentile=0.1, value=20),
                Percentile(percentile=0.9, value=80),
            ],
            open_upper_bound=True,
            open_lower_bound=True,
            upper_bound=100.0,
            lower_bound=0.0,
            zero_point=None,
        )
        raw_cdf = dist.get_cdf()
        values = [p.percentile for p in raw_cdf]
        assert values[0] > 0.0
        assert values[-1] < 1.0

    def test_standardize_closed_bounds_no_mass_outside(self, linear_scaling):
        dist = NumericDistribution(
            declared_percentiles=[
                Percentile(percentile=0.1, value=20),
                Percentile(percentile=0.9, value=80),
            ],
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=100.0,
            lower_bound=0.0,
            zero_point=None,
        )
        raw_cdf = dist.get_cdf()
        values = [p.percentile for p in raw_cdf]
        assert values[0] == pytest.approx(0.0, abs=1e-6)
        assert values[-1] == pytest.approx(1.0, abs=1e-6)


class TestProjectCdfToValid:
    """The one shared CDF-validity projection (also used by tail humility)."""

    def test_mostly_out_of_range_open_bounds(self):
        # Degenerate forecast: ~98% of mass outside [0,100] leaves inbound mass
        # below the min-step floor. The projection must grow inbound rather
        # than emit a non-monotone CDF.
        cdf = generate_continuous_cdf(
            {17.2: -17.4, 17.4: 106.7, 24.9: 109.4, 38.4: 115.7, 72.6: 117.7},
            True,
            True,
            100.0,
            0.0,
            None,
            101,
        )
        assert_metaculus_valid(cdf, open_lower_bound=True, open_upper_bound=True)

    def test_near_vertical_jump_closed_bounds(self):
        # A near-vertical jump puts >cap mass in one bucket; the projection
        # must spread it under the PMF cap without leaving an over-cap step.
        cdf = generate_continuous_cdf(
            {59.2: -6.5, 60.5: 0.9, 61.4: 112.8}, False, False, 100.0, 0.0, None, 201
        )
        assert_metaculus_valid(cdf, open_lower_bound=False, open_upper_bound=False)

    def test_identity_on_already_valid_cdf(self):
        # A benign, already-valid CDF is returned unchanged up to rounding.
        base = generate_continuous_cdf({10: 20, 50: 50, 90: 80}, True, True, 100.0, 0.0, None, 201)
        again = project_cdf_to_valid(base, open_lower_bound=True, open_upper_bound=True)
        assert again == pytest.approx(base, abs=1e-9)

    @pytest.mark.parametrize("open_lower", [True, False])
    @pytest.mark.parametrize("open_upper", [True, False])
    def test_all_bound_combos_valid(self, open_lower, open_upper):
        cdf = generate_continuous_cdf(
            {10: 15, 30: 40, 50: 55, 70: 70, 90: 92},
            open_upper,
            open_lower,
            100.0,
            0.0,
            None,
            201,
        )
        assert_metaculus_valid(cdf, open_lower_bound=open_lower, open_upper_bound=open_upper)

    def test_tiny_inbound_one_closed_side(self):
        # Direct feed of a near-zero-inbound CDF with a closed side that offers
        # no room: the whole min-step deficit must land on the open side.
        # Previously the split dropped half the deficit and emitted a negative
        # last step. Size 201 is the worst case (most steps to clear the floor).
        import numpy as np

        cdf = list(np.linspace(0.0, 0.002, 201))
        out = project_cdf_to_valid(cdf, open_lower_bound=False, open_upper_bound=True)
        assert_metaculus_valid(out, open_lower_bound=False, open_upper_bound=True)

    def test_all_mass_outside_both_open(self):
        # A flat CDF (all mass above the range) has zero inbound; both sides
        # are open so recovery grows the interval to the floor either way.
        out = project_cdf_to_valid([0.0] * 201, open_lower_bound=True, open_upper_bound=True)
        assert_metaculus_valid(out, open_lower_bound=True, open_upper_bound=True)

    @pytest.mark.parametrize("seed", range(6))
    def test_adversarial_low_inbound_property(self, seed):
        # Squash random monotone CDFs toward an edge / near-flat, across every
        # size and bound combo — the low-inbound regime the earlier property
        # tests (centres in [0,1]) never reached.
        import numpy as np

        rng = np.random.default_rng(seed)
        for _ in range(500):
            n = int(rng.choice([201, 101, 51, 26, 11]))
            lo, uo = bool(rng.integers(0, 2)), bool(rng.integers(0, 2))
            base = np.sort(rng.uniform(0, 1, n))
            base = (base - base[0]) / (base[-1] - base[0] + 1e-12)
            f = np.clip(np.maximum.accumulate(base * rng.uniform(1e-4, 0.01)), 0, 1)
            out = project_cdf_to_valid(f.tolist(), open_lower_bound=lo, open_upper_bound=uo)
            assert len(out) == n
            assert_metaculus_valid(out, open_lower_bound=lo, open_upper_bound=uo)


# ── NumericDefaults ─────────────────────────────────────────────────


class TestNumericDefaults:
    def test_default_cdf_size(self):
        assert NumericDefaults.DEFAULT_CDF_SIZE == 201

    def test_max_pmf_value_with_wiggle(self):
        cap = NumericDefaults.get_max_pmf_value(201, include_wiggle_room=True)
        assert cap == pytest.approx(0.2 * 0.95)

    def test_max_pmf_value_without_wiggle(self):
        cap = NumericDefaults.get_max_pmf_value(201, include_wiggle_room=False)
        assert cap == pytest.approx(0.2)

    def test_max_pmf_value_smaller_cdf_size(self):
        cap = NumericDefaults.get_max_pmf_value(51, include_wiggle_room=False)
        assert cap > 0.2


# ── Edge cases ──────────────────────────────────────────────────────


class TestEdgeCases:
    def test_tiny_range(self):
        cdf = generate_continuous_cdf(
            {10: 10.0, 20: 10.5, 40: 11.0, 60: 11.5, 80: 12.0, 90: 12.5},
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=15.0,
            lower_bound=5.0,
            zero_point=None,
            cdf_size=201,
        )
        assert len(cdf) == 201

    def test_negative_range(self):
        cdf = generate_continuous_cdf(
            {10: -80.0, 20: -60.0, 40: -50.0, 60: -40.0, 80: -20.0, 90: -10.0},
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=0.0,
            lower_bound=-100.0,
            zero_point=None,
            cdf_size=201,
        )
        assert len(cdf) == 201
        assert all(0.0 <= v <= 1.0 for v in cdf)

    def test_log_scale_near_zero_point(self, log_scaling, valid_percentiles_log):
        cdf = generate_continuous_cdf(
            valid_percentiles_log,
            log_scaling.open_upper_bound,
            log_scaling.open_lower_bound,
            log_scaling.range_max,
            log_scaling.range_min,
            log_scaling.zero_point,
            log_scaling.cdf_size,
        )
        assert len(cdf) == 201


# ── open-bound outlier clamping ─────────────────────────────────────


class TestOpenBoundOutlierClamping:
    """Samples believing "the answer is beyond an open bound" must survive.

    Reproduces the 2026-06-28 production failures on Q44215 (Waymo,
    bounds 7M-9M open) and Q44185 (foreign visitors, 4M-6.5M open):
    values 1.5-2.7x above the upper bound killed 4/5 ensemble samples
    and errored three consecutive runs.
    """

    def _waymo_dist(self, percentiles, **overrides):
        kwargs: dict = {
            "declared_percentiles": percentiles,
            "open_upper_bound": True,
            "open_lower_bound": True,
            "upper_bound": 9_000_000.0,
            "lower_bound": 7_000_000.0,
            "zero_point": None,
            "cdf_size": 201,
        }
        kwargs.update(overrides)
        return NumericDistribution(**kwargs)

    def test_far_above_open_upper_bound_survives(self):
        # Actual killed sample from run_2026-06-28_200004.log
        dist = self._waymo_dist(
            [
                Percentile(percentile=0.6, value=14_500_000.0),
                Percentile(percentile=0.8, value=18_000_000.0),
                Percentile(percentile=0.9, value=24_000_000.0),
            ]
        )
        values = [p.value for p in dist.declared_percentiles]
        assert max(values) <= 9_000_000.0 + 0.25 * 2_000_000.0
        # strict ordering must survive compression (CDF interpolation needs it)
        assert all(b > a for a, b in zip(values, values[1:], strict=False))

    def test_entirely_out_of_range_forecast_survives(self):
        # Actual killed sample from run_2026-06-28_210003.log: every
        # percentile above the upper bound ("No declared percentiles
        # are within the range").
        dist = self._waymo_dist(
            [
                Percentile(percentile=0.1, value=13_000_000.0),
                Percentile(percentile=0.2, value=13_800_000.0),
                Percentile(percentile=0.4, value=14_800_000.0),
                Percentile(percentile=0.6, value=16_500_000.0),
                Percentile(percentile=0.8, value=17_500_000.0),
                Percentile(percentile=0.9, value=18_500_000.0),
            ]
        )
        cdf = dist.get_cdf()
        # The belief "above the range" must show up as mass beyond the
        # open upper bound: the CDF tops out well below 1.
        assert cdf[-1].percentile < 0.5

    def test_far_below_open_lower_bound_survives(self):
        dist = self._waymo_dist(
            [
                Percentile(percentile=0.1, value=1_000_000.0),
                Percentile(percentile=0.5, value=2_000_000.0),
                Percentile(percentile=0.9, value=8_000_000.0),
            ]
        )
        assert min(p.value for p in dist.declared_percentiles) >= 7_000_000.0 - 0.25 * 2_000_000.0

    def test_closed_upper_bound_still_rejects(self):
        # With a closed bound the old strictness is correct: the value
        # cannot resolve out there, so the sample is genuinely broken.
        with pytest.raises((ValueError, ValidationError), match="far exceeding|within the range"):
            self._waymo_dist(
                [
                    Percentile(percentile=0.6, value=14_500_000.0),
                    Percentile(percentile=0.8, value=18_000_000.0),
                    Percentile(percentile=0.9, value=24_000_000.0),
                ],
                open_upper_bound=False,
            )

    def test_moderately_out_of_range_unchanged(self):
        # Values inside the wiggle allowance are not touched.
        dist = self._waymo_dist(
            [
                Percentile(percentile=0.2, value=7_500_000.0),
                Percentile(percentile=0.5, value=8_200_000.0),
                Percentile(percentile=0.9, value=9_300_000.0),
            ]
        )
        assert [p.value for p in dist.declared_percentiles] == [
            7_500_000.0,
            8_200_000.0,
            9_300_000.0,
        ]

    def test_submittable_cdf_from_clamped_sample(self):
        # End to end through generate_continuous_cdf: the killed Waymo
        # sample now yields a valid, standardized 201-point CDF.
        cdf = generate_continuous_cdf(
            percentile_values={10: 13_000_000.0, 50: 15_000_000.0, 90: 18_500_000.0},
            open_upper_bound=True,
            open_lower_bound=True,
            upper_bound=9_000_000.0,
            lower_bound=7_000_000.0,
            zero_point=None,
            cdf_size=201,
        )
        assert len(cdf) == 201
        assert all(b - a >= 0 for a, b in zip(cdf, cdf[1:], strict=False))
        assert cdf[-1] < 1.0
