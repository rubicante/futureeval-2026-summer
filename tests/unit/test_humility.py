"""Unit tests for tail-humility post-processing (forecasting/humility.py)."""

from __future__ import annotations

import numpy as np
import pytest

from forecasting.distributions import NumericDefaults
from forecasting.humility import apply_tail_humility, floor_mc, widen_cdf

_MIN_STEP = NumericDefaults.MIN_STEP


def _sharp_cdf(n: int = 201, center: float = 0.5, sigma: float = 0.03) -> list[float]:
    from statistics import NormalDist

    grid = np.linspace(0, 1, n)
    return [NormalDist(center, sigma).cdf(float(x)) for x in grid]


def _validation_cap(n: int) -> float:
    """Metaculus's exact interior-step cap for a size-n CDF (no wiggle)."""
    return NumericDefaults.MAX_NUMERIC_PMF_VALUE * (
        NumericDefaults.DEFAULT_INBOUND_OUTCOME_COUNT / (n - 1)
    )


def assert_metaculus_valid(cdf: list[float], *, lower_open: bool, upper_open: bool) -> None:
    """Assert *cdf* satisfies every Metaculus continuous-CDF constraint."""
    n = len(cdf)
    d = np.diff(cdf)
    assert (d >= _MIN_STEP).all(), f"step below floor: {d.min():.3e}"
    assert (d <= _validation_cap(n)).all(), (
        f"step over cap: {d.max():.4f} > {_validation_cap(n):.4f}"
    )
    reserve = NumericDefaults.MIN_OUTSIDE_OPEN
    if lower_open:
        assert cdf[0] >= reserve - 1e-12, f"open lower reserves < 0.1%: {cdf[0]:.5f}"
    else:
        assert cdf[0] == 0.0, "closed lower bound must be exactly 0.0"
    if upper_open:
        assert cdf[-1] <= 1.0 - reserve + 1e-12, f"open upper reserves < 0.1%: {1 - cdf[-1]:.5f}"
    else:
        assert cdf[-1] == 1.0, "closed upper bound must be exactly 1.0"


class TestWidenCdf:
    def test_weight_zero_is_identity(self):
        cdf = _sharp_cdf()
        assert widen_cdf(cdf, 0.0) == cdf

    def test_monotone_with_min_steps(self):
        out = widen_cdf(_sharp_cdf(), 0.1)
        diffs = np.diff(out)
        assert (diffs >= _MIN_STEP * 0.999).all()
        assert out[0] >= 0.0 and out[-1] <= 1.0

    def test_median_preserved_tails_fattened(self):
        cdf = _sharp_cdf(center=0.5, sigma=0.03)
        out = widen_cdf(cdf, 0.1)
        grid = np.linspace(0, 1, len(cdf))
        median_in = np.interp(0.5, cdf, grid)
        median_out = np.interp(0.5, out, grid)
        assert median_out == pytest.approx(median_in, abs=0.01)
        # More mass far from the median: CDF higher at 3σ below center
        idx = 60  # 0.30 on the grid ≈ 6σ below center
        assert out[idx] > cdf[idx]

    def test_off_center_median(self):
        cdf = _sharp_cdf(center=0.15, sigma=0.02)
        out = widen_cdf(cdf, 0.15)
        assert (np.diff(out) >= _MIN_STEP * 0.999).all()
        assert out[-1] <= 1.0

    def test_discrete_size(self):
        out = widen_cdf(_sharp_cdf(n=25), 0.1)
        assert len(out) == 25
        assert (np.diff(out) > 0).all()

    def test_closed_bounds_pinned_exactly(self):
        # Regression: the Halo Metascore miss (question 44798, 2026-07-20).
        # Widening a closed-both-bounds CDF lifted cdf[0] off 0.0 and cdf[-1]
        # off 1.0, which Metaculus rejects with "CDF Invalid".
        cdf = _sharp_cdf(n=101, center=0.85, sigma=0.05)
        cdf = [(v - cdf[0]) / (cdf[-1] - cdf[0]) for v in cdf]  # start closed
        out = widen_cdf(cdf, 0.1, lower_open=False, upper_open=False)
        assert_metaculus_valid(out, lower_open=False, upper_open=False)

    def test_sharp_closed_bounds_respects_pmf_cap(self):
        # Regression: a sharp forecast on a closed-both-bounds discrete
        # question (Halo's shape) drove an interior step to ~0.95 against a
        # 0.4 cap. The projection must spread it under the cap.
        cdf = _sharp_cdf(n=101, center=0.985, sigma=0.008)
        cdf = [(v - cdf[0]) / (cdf[-1] - cdf[0]) for v in cdf]
        out = widen_cdf(cdf, 0.05, lower_open=False, upper_open=False)
        assert_metaculus_valid(out, lower_open=False, upper_open=False)

    def test_open_upper_stays_below_one(self):
        # Regression: a sharp top-piled forecast on an open upper bound drove
        # cdf[-1] to exactly 1.0 — the mirror of the Halo bug. Open bounds
        # must reserve mass outside the range.
        cdf = _sharp_cdf(n=11, center=0.97, sigma=0.02)
        cdf = [v - cdf[0] for v in cdf]  # closed lower, open upper
        out = widen_cdf(cdf, 0.1, lower_open=False, upper_open=True)
        assert_metaculus_valid(out, lower_open=False, upper_open=True)

    def test_one_closed_one_open_bound(self):
        cdf = _sharp_cdf(n=101, center=0.85, sigma=0.05)
        cdf = [(v - cdf[0]) / (cdf[-1] - cdf[0]) for v in cdf]
        out = widen_cdf(cdf, 0.1, lower_open=True, upper_open=False)
        assert_metaculus_valid(out, lower_open=True, upper_open=False)

    def test_open_bounds_default_is_identity_ish(self):
        # Default (open) behaviour on a benign CDF: endpoints stay interior
        # and the median is preserved; explicit open flags match the default.
        cdf = _sharp_cdf(n=201, center=0.5, sigma=0.06)
        default = widen_cdf(cdf, 0.1)
        explicit = widen_cdf(cdf, 0.1, lower_open=True, upper_open=True)
        assert default == explicit
        assert_metaculus_valid(default, lower_open=True, upper_open=True)

    @pytest.mark.parametrize("seed", range(12))
    def test_property_all_bound_combos_valid(self, seed):
        # The real regression guard: random sharp/edge CDFs across every size
        # and bound combination must always come out Metaculus-valid.
        rng = np.random.default_rng(seed)
        for _ in range(400):
            n = int(rng.choice([201, 101, 51, 26, 11]))
            lo, uo = bool(rng.integers(0, 2)), bool(rng.integers(0, 2))
            center = float(rng.uniform(0.0, 1.0))
            sigma = float(rng.choice([0.006, 0.01, 0.03, 0.15]))
            weight = float(rng.choice([0.05, 0.1, 0.15, 0.25]))
            cdf = _sharp_cdf(n=n, center=center, sigma=sigma)
            if not lo:
                cdf = [v - cdf[0] for v in cdf]
            if not uo and cdf[-1] > 0:
                cdf = [v / cdf[-1] for v in cdf]
            out = widen_cdf(cdf, weight, lower_open=lo, upper_open=uo)
            assert len(out) == n
            assert_metaculus_valid(out, lower_open=lo, upper_open=uo)


class TestFloorMc:
    def test_floors_and_normalizes(self):
        probs = {"a": 0.98, "b": 0.015, "c": 0.005}
        out = floor_mc(probs)
        assert sum(out.values()) == pytest.approx(1.0)
        # No renormalization dip: every option is at or above the floor.
        assert min(out.values()) >= 0.02 - 1e-9
        assert out["a"] > 0.9

    def test_no_dip_below_floor_many_options(self):
        # Renormalizing after flooring used to push options back under 2%.
        probs = {str(i): (0.9 if i == 0 else 0.1 / 19) for i in range(20)}
        out = floor_mc(probs)
        assert sum(out.values()) == pytest.approx(1.0)
        assert min(out.values()) >= 0.02 - 1e-9

    def test_infeasible_floor_falls_back_uniform(self):
        # 60 options can't all reach 2% (60 · 0.02 = 1.2 > 1); stay normalized.
        probs = {str(i): 1.0 / 60 for i in range(60)}
        out = floor_mc(probs)
        assert sum(out.values()) == pytest.approx(1.0)
        assert all(v > 0 for v in out.values())

    def test_already_flat_unchanged_shape(self):
        probs = {"a": 0.5, "b": 0.5}
        out = floor_mc(probs)
        assert out == pytest.approx(probs)


class TestDispatch:
    def test_binary_untouched(self):
        assert apply_tail_humility(0.97, "binary", 0.1) == 0.97

    def test_numeric_widened(self):
        cdf = _sharp_cdf()
        out = apply_tail_humility(cdf, "numeric", 0.1)
        assert out != cdf and len(out) == len(cdf)

    def test_mc_floored(self):
        out = apply_tail_humility({"a": 0.999, "b": 0.001}, "multiple_choice", 0.1)
        assert out["b"] >= 0.019

    def test_weight_zero_noop(self):
        cdf = _sharp_cdf()
        assert apply_tail_humility(cdf, "numeric", 0.0) is cdf
