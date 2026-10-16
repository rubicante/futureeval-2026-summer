from __future__ import annotations

from collections import Counter

import numpy as np
from pydantic import BaseModel, Field, model_validator


def generate_continuous_cdf(
    percentile_values: dict,
    open_upper_bound: bool,
    open_lower_bound: bool,
    upper_bound: float,
    lower_bound: float,
    zero_point: float | None,
    cdf_size: int,
) -> list[float]:
    """
    Returns: list[float]: A list of 201 float values representing the CDF.
    """

    percentiles = []
    for percentile, value in percentile_values.items():
        percentiles.append(Percentile(percentile=percentile / 100, value=value))
    numeric_distribution = NumericDistribution(
        declared_percentiles=percentiles,
        open_upper_bound=open_upper_bound,
        open_lower_bound=open_lower_bound,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        zero_point=zero_point,
        cdf_size=cdf_size,
    )
    cdf_as_objects = numeric_distribution.get_cdf()
    cdf_as_floats = [percentile.percentile for percentile in cdf_as_objects]

    return cdf_as_floats


def cdf_location(
    value: float,
    *,
    range_min: float,
    range_max: float,
    zero_point: float | None = None,
) -> float:
    """Map a real-world value to a [0, 1] position on the CDF x-axis.

    Handles both linear and log (via *zero_point*) scaled questions.
    The returned position may fall outside [0, 1] if *value* is beyond
    the axis bounds.
    """
    if zero_point is not None:
        deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
        v = value if value != zero_point else value + 1e-10
        return float(
            (
                np.log((v - range_min) * (deriv_ratio - 1) + (range_max - range_min))
                - np.log(range_max - range_min)
            )
            / np.log(deriv_ratio)
        )
    return (value - range_min) / (range_max - range_min)


class NumericDefaults:
    DEFAULT_CDF_SIZE = 201  # Discrete questions have fewer points, Numeric will have 201 points
    DEFAULT_INBOUND_OUTCOME_COUNT = DEFAULT_CDF_SIZE - 1
    MAX_NUMERIC_PMF_VALUE = 0.2
    # Metaculus rejects a CDF unless every step is ≥ MIN_STEP; STEP_MARGIN
    # keeps enforced steps a hair above it so the final round-to-10 can never
    # nudge one below the floor.
    MIN_STEP = 5e-05
    STEP_MARGIN = 1e-9
    # Metaculus also requires ≥ 0.1% of probability mass outside each *open*
    # bound (and none outside a closed one). Distinct from MIN_STEP: this is
    # the reserved endpoint mass, not the per-step floor.
    MIN_OUTSIDE_OPEN = 0.001

    @classmethod
    def get_max_pmf_value(cls, cdf_size: int, include_wiggle_room: bool = True) -> float:
        # cap depends on inboundOutcomeCount (0.2 if it is the default 200)
        inbound_outcome_count = cdf_size - 1
        normal_cap = cls.MAX_NUMERIC_PMF_VALUE * (
            cls.DEFAULT_INBOUND_OUTCOME_COUNT / inbound_outcome_count
        )

        if include_wiggle_room:
            return normal_cap * 0.95
        else:
            return normal_cap


def _fit_pmf_to_sum(pmf: np.ndarray, lo: float, hi: float, target: float) -> np.ndarray:
    """Fit *pmf* into the box ``[lo, hi]`` at ``sum == target`` in one solve.

    A uniform shift τ, found by bisection on the monotone
    ``sum(clip(pmf + τ, lo, hi))`` — exact and always terminating whenever the
    box is feasible (``lo·k ≤ target ≤ hi·k``, which the caller guarantees).

    Solving the box *and* the sum jointly is the whole point, and the reason
    this function exists rather than a clamp-then-renormalize: min-step, the
    per-step cap, and sum-to-one are coupled, so enforcing them in sequence
    lets the final step reopen an earlier violation. That coupling is exactly
    the latent min-step bug in the affine standardizer this path replaced —
    its final renormalize could scale a floor-height step back below the
    floor. A single shift avoids it by construction.
    """
    k = len(pmf)
    if target <= lo * k:
        return np.full(k, lo)
    if target >= hi * k:
        return np.full(k, hi)
    tau_lo, tau_hi = lo - float(pmf.max()), hi - float(pmf.min())
    for _ in range(80):
        tau = 0.5 * (tau_lo + tau_hi)
        if np.clip(pmf + tau, lo, hi).sum() < target:
            tau_lo = tau
        else:
            tau_hi = tau
    return np.clip(pmf + 0.5 * (tau_lo + tau_hi), lo, hi)


def project_cdf_to_valid(
    cdf: list[float] | np.ndarray, *, open_lower_bound: bool, open_upper_bound: bool
) -> list[float]:
    """Return a Metaculus-valid gridded CDF for the given bound types.

    The single, authoritative enforcer of CDF submission validity, shared by
    :meth:`NumericDistribution._standardize_cdf` (per-sample generation) and
    the tail-humility widen (``forecasting.humility``). For a length-n input
    the returned CDF is guaranteed to satisfy every constraint Metaculus
    hard-rejects on:

    - strictly increasing, every step in ``[MIN_STEP, cap]`` (``cap`` = the
      per-size PMF ceiling; a step below the floor or above the cap is a hard
      reject, not a warning);
    - closed bound → endpoint exactly 0.0 / 1.0 — mass outside a closed range
      is impossible, and a wrong endpoint here is the 2026-07 "CDF Invalid"
      rejection that lost the Halo question;
    - open bound → at least ``MIN_OUTSIDE_OPEN`` (0.1%, the documented
      Metaculus minimum) reserved outside the range, so the endpoint sits at
      ≤ 1 − 0.001 / ≥ 0.001.

    Behaviour worth knowing for review: on an already-valid CDF the fit is the
    identity up to rounding, so the median is preserved and the common path is
    untouched — the change versus the retired affine method is confined to
    CDFs that were out of spec (a sub-1% median-preserving shift, measured).

    The box ``[step_min, cap]`` is only non-empty when
    ``(n-1)·step_min ≤ inbound ≤ (n-1)·cap``. Feasible by construction for
    closed bounds (inbound = 1); the open-bound branch below restores it.
    """
    f = np.asarray(cdf, dtype=float)
    n = len(f)
    min_step = NumericDefaults.MIN_STEP
    step_min = min_step + NumericDefaults.STEP_MARGIN
    step_max = NumericDefaults.get_max_pmf_value(n)  # per-size cap, with wiggle
    reserve = NumericDefaults.MIN_OUTSIDE_OPEN  # required mass outside an open bound

    lower = 0.0 if not open_lower_bound else max(float(f[0]), reserve)
    upper = 1.0 if not open_upper_bound else min(float(f[-1]), 1.0 - reserve)

    # A forecast that piles most of its mass outside an open bound leaves
    # inbound mass below (n-1)·step_min, so the box has no feasible point and
    # every step can't clear the floor. This is a real degenerate case (found
    # by fuzzing the generation path, not hypothetical): grow inbound back to
    # the floor while never eating into the required outside reserve. The
    # affine method got this for free from its additive uniform tilt.
    min_inbound = (n - 1) * step_min
    lo_floor = 0.0 if not open_lower_bound else reserve
    hi_ceil = 1.0 if not open_upper_bound else 1.0 - reserve
    if upper - lower < min_inbound:
        # Split the shortfall across both sides, but let each side absorb what
        # the other can't take — a closed bound offers no room, so its whole
        # half must land on the open side (missing this was the min-inbound
        # bug: half the deficit was silently dropped, producing a negative
        # last step). hi_ceil − lo_floor ≥ min_inbound for every supported
        # size, so the sides always have enough room between them.
        deficit = min_inbound - (upper - lower)
        room_up, room_down = hi_ceil - upper, lower - lo_floor
        up = min(room_up, deficit / 2)
        down = min(room_down, deficit - up)
        up = min(room_up, deficit - down)
        lower -= down
        upper += up
    inbound = upper - lower
    if inbound < min_inbound:  # unreachable for n ≤ ~20000; widest legal fallback
        lower, upper = lo_floor, hi_ceil
        inbound = upper - lower

    pmf = np.clip(np.diff(f), 0.0, None)
    pmf = np.full(n - 1, inbound / (n - 1)) if pmf.sum() <= 0 else pmf * (inbound / pmf.sum())
    pmf = _fit_pmf_to_sum(pmf, step_min, step_max, inbound)

    out = np.empty(n)
    out[0] = lower
    out[1:] = lower + np.cumsum(pmf)
    out[-1] = upper
    return [float(v) for v in np.round(out, 10)]


class Percentile(BaseModel):
    percentile: float = Field(
        description="A number between 0 and 1 (e.g. '90% of people are age 60 or younger' translates to '0.9')",
    )
    value: float = Field(
        description="The number matching the percentile (e.g. '90% of people are age 60 or younger' translates to '60')",
    )

    @model_validator(mode="after")
    def validate_percentile(self: Percentile) -> Percentile:
        if self.percentile < 0 or self.percentile > 1:
            raise ValueError(f"Percentile must be between 0 and 1, but was {self.percentile}")
        if np.isnan(self.percentile):
            raise ValueError(f"Percentile must be a number, but was {self.percentile}")
        return self


class NumericDistribution(BaseModel):
    declared_percentiles: list[Percentile]
    open_upper_bound: bool
    open_lower_bound: bool
    upper_bound: float
    lower_bound: float
    zero_point: float | None
    cdf_size: int | None = (
        None  # Normal numeric questions have 201 points, but discrete questions have fewer
    )
    standardize_cdf: bool = True
    strict_validation: bool = True
    is_date: bool = False

    @model_validator(mode="after")
    def validate_percentiles(self: NumericDistribution) -> NumericDistribution:
        self._check_percentiles_increasing()
        self._check_log_scaled_fields()
        self.declared_percentiles = self._clamp_open_bound_outliers(self.declared_percentiles)
        percentiles = self.declared_percentiles

        if not self.strict_validation:
            return self

        self._check_percentile_spacing()

        if self.standardize_cdf:
            self._check_too_far_from_bounds(percentiles)
        if self.standardize_cdf and len(percentiles) == self.cdf_size:
            self._check_distribution_too_tall(percentiles)

        self.declared_percentiles = self._check_and_update_repeating_values(percentiles)
        return self

    def _check_percentiles_increasing(self) -> None:
        percentiles = self.declared_percentiles
        for i in range(len(percentiles) - 1):
            if percentiles[i].percentile >= percentiles[i + 1].percentile:
                raise ValueError("Percentiles must be in strictly increasing order")
            if percentiles[i].value > percentiles[i + 1].value:
                raise ValueError("Values must be in strictly increasing order")
        if len(percentiles) < 2:
            raise ValueError("NumericDistribution must have at least 2 percentiles")

    def _check_percentile_spacing(self) -> None:
        percentiles = self.declared_percentiles
        min_step = NumericDefaults.MIN_STEP
        for i in range(len(percentiles) - 1):
            if abs(percentiles[i + 1].percentile - percentiles[i].percentile) < min_step:
                raise ValueError(
                    f"Percentiles at indices {i} and {i + 1} are too close. CDF must be increasing by at least {min_step} at every step. "
                    f"{percentiles[i].percentile} and {percentiles[i + 1].percentile} "
                    f"at values {percentiles[i].value} and {percentiles[i + 1].value}. "
                    "One possible reason is that your prediction is mostly or completely out of the upper/lower "
                    "bound range thus assigning very little probability to any one x-axis value."
                )

    def _check_log_scaled_fields(self) -> None:
        if self.zero_point is not None and self.lower_bound <= self.zero_point:
            raise ValueError(
                f"Lower bound {self.lower_bound} is less than or equal to the zero point {self.zero_point}. "
                "Lower bound must be greater than the zero point."
            )

        for percentile in self.declared_percentiles:
            if self.zero_point is not None and percentile.value < self.zero_point:
                raise ValueError(
                    f"Percentile value {percentile.value} is less than the zero point {self.zero_point}. "
                    "Determining probability less than zero point is currently not supported."
                )

    def _check_and_update_repeating_values(self, percentiles: list[Percentile]) -> list[Percentile]:
        unique_value_count = Counter(percentile.value for percentile in percentiles)
        final_percentiles = []
        for percentile in percentiles:
            value = percentile.value
            count = unique_value_count[value]
            repeated_value = count > 1
            value_in_bounds = self.lower_bound < value < self.upper_bound
            value_above_bound = value >= self.upper_bound
            value_below_bound = value <= self.lower_bound
            epsilon = 1e-10
            if not repeated_value:
                final_percentiles.append(percentile)
            elif value_in_bounds:
                greater_epsilon = 1e-6
                modification = (1 - percentile.percentile) * greater_epsilon
                final_percentiles.append(
                    Percentile(
                        value=value - modification,
                        percentile=percentile.percentile,
                    )
                )
            elif value_above_bound:
                modification = epsilon * percentile.percentile
                final_percentiles.append(
                    Percentile(
                        value=self.upper_bound + modification,
                        percentile=percentile.percentile,
                    )
                )
            elif value_below_bound:
                modification = epsilon * (1 - percentile.percentile)
                final_percentiles.append(
                    Percentile(
                        value=self.lower_bound - modification,
                        percentile=percentile.percentile,
                    )
                )
            else:
                raise ValueError(
                    f"Unexpected state: value {value} is repeated {count} times. Bound is {self.lower_bound} and {self.upper_bound}"
                )
        return final_percentiles

    def _clamp_open_bound_outliers(self, percentiles: list[Percentile]) -> list[Percentile]:
        """Compress values far beyond an *open* bound instead of rejecting them.

        A sample whose percentiles sit beyond an open bound is a legitimate
        "the answer is outside the range" belief; pulling those values to
        just past the bound makes the CDF assign the probability mass beyond
        the bound. Closed bounds keep strict validation downstream — a value
        beyond a closed bound cannot resolve there and marks a broken sample.

        Out-of-envelope values are compressed by rank into a small band past
        the envelope rather than clamped flat: CDF interpolation requires
        strictly increasing values, and a flat clamp would create ties.
        """
        span = self.upper_bound - self.lower_bound
        limit_over = span * 0.2  # compression starts past here
        headroom = span * 0.04  # compressed band; total stays inside the 25% wiggle
        upper_limit = self.upper_bound + limit_over
        lower_limit = self.lower_bound - limit_over

        values = [p.value for p in percentiles]
        if self.open_upper_bound:
            # values are sorted, so the overshoot is a suffix
            over = [i for i, v in enumerate(values) if v > upper_limit]
            for rank, i in enumerate(over):
                values[i] = upper_limit + headroom * (rank + 1) / len(over)
        if self.open_lower_bound:
            under = [i for i, v in enumerate(values) if v < lower_limit]
            for rank, i in enumerate(under):
                values[i] = lower_limit - headroom * (len(under) - rank) / len(under)

        return [
            p if v == p.value else Percentile(value=v, percentile=p.percentile)
            for p, v in zip(percentiles, values, strict=True)
        ]

    def _check_too_far_from_bounds(self, percentiles: list[Percentile]) -> None:
        max_to_min_range = self.upper_bound - self.lower_bound

        wiggle_percent = 0.25
        wiggle_room = max_to_min_range * wiggle_percent
        upper_bound_plus_wiggle_room = self.upper_bound + wiggle_room
        lower_bound_minus_wiggle_room = self.lower_bound - wiggle_room
        percentiles_within_bounds_plus_wiggle_room = [
            percentile
            for percentile in percentiles
            if lower_bound_minus_wiggle_room <= percentile.value <= upper_bound_plus_wiggle_room
        ]
        if len(percentiles_within_bounds_plus_wiggle_room) == 0:
            raise ValueError(
                f"No declared percentiles are within the range of the question +/- {wiggle_percent * 100}%. "
                f"Lower bound: {self.lower_bound}, upper bound: {self.upper_bound}. "
                f"Percentiles: {percentiles}"
            )

        max_to_min_range_buffer = max_to_min_range * 2
        percentiles_far_exceeding_bounds = [
            percentile
            for percentile in percentiles
            if percentile.value < self.lower_bound - max_to_min_range_buffer
            or percentile.value > self.upper_bound + max_to_min_range_buffer
        ]
        if len(percentiles_far_exceeding_bounds) > 0:
            raise ValueError(
                "Some declared percentiles are far exceeding the bounds of the question. "
                f"Lower bound: {self.lower_bound}, upper bound: {self.upper_bound}. "
                f"Percentiles: {percentiles_far_exceeding_bounds}"
            )

    def _check_distribution_too_tall(self, cdf: list[Percentile]) -> None:
        if len(cdf) != self.cdf_size:
            raise ValueError(
                f"CDF size is not the same as the declared percentiles. CDF size: {len(cdf)}, declared percentiles: {self.cdf_size}"
            )
        cap = NumericDefaults.get_max_pmf_value(len(cdf), include_wiggle_room=False)

        for i in range(len(cdf) - 1):
            pmf_value = cdf[i + 1].percentile - cdf[i].percentile
            if pmf_value > cap:
                raise ValueError(
                    f"Distribution is too concentrated. The probability mass between "
                    f"values {cdf[i].value} and {cdf[i + 1].value} is {pmf_value:.4f}, "
                    f"which exceeds the maximum allowed of {cap:.4f}."
                )

    def get_cdf(self) -> list[Percentile]:
        """
        Turns a list of percentiles into a full distribution (201 points, if numeric, otherwise based on discrete values)
        between upper and lower bound (taking into account probability assigned above and below the bounds)
        that is compatible with Metaculus questions.

        cdf stands for 'continuous distribution function'

        At Metaculus CDFs are often represented with 201 points. Each point has:
        - percentile ("X% of values are below this point". This is the y axis of the cdf graph)
        - 'value' or 'nominal location' (The real world number that answers the question)
        - cdf location (a number between 0 and 1 representing where the point is on the cdf x axis, where 0 is range min, and 1 is range max)
        """

        cdf_size = self.cdf_size or NumericDefaults.DEFAULT_CDF_SIZE
        continuous_cdf = []
        cdf_xaxis = []
        cdf_eval_locations = [i / (cdf_size - 1) for i in range(cdf_size)]
        for loc in cdf_eval_locations:
            continuous_cdf.append(self._get_cdf_at(loc))
            cdf_xaxis.append(self._cdf_location_to_nominal_location(loc))

        if self.standardize_cdf:
            continuous_cdf = self._standardize_cdf(continuous_cdf)

        percentiles = [
            Percentile(value=value, percentile=percentile)
            for value, percentile in zip(cdf_xaxis, continuous_cdf, strict=True)
        ]
        assert len(percentiles) == cdf_size

        return percentiles

    @classmethod
    def _percentile_list_to_dict(
        cls, percentiles: list[Percentile], multiply_by_100: bool
    ) -> dict[float, float]:
        return {
            (
                percentile.percentile * 100 if multiply_by_100 else percentile.percentile
            ): percentile.value
            for percentile in percentiles
        }

    @classmethod
    def _dict_to_percentile_list(
        cls, percentile_dict: dict[float, float], divide_by_100: bool
    ) -> list[Percentile]:
        return [
            Percentile(
                percentile=percentile / 100 if divide_by_100 else percentile,
                value=value,
            )
            for percentile, value in percentile_dict.items()
        ]

    def _add_explicit_upper_lower_bound_percentiles(
        self,
        input_percentiles: list[Percentile],
    ) -> list[Percentile]:
        open_upper_bound = self.open_upper_bound
        open_lower_bound = self.open_lower_bound
        range_max = self.upper_bound
        range_min = self.lower_bound

        return_percentiles = self._percentile_list_to_dict(input_percentiles, multiply_by_100=True)
        percentile_max = max(return_percentiles)
        percentile_min = min(return_percentiles)
        range_size = abs(range_max - range_min)
        buffer = 1 if range_size > 100 else 0.01 * range_size

        # Adjust any values that are exactly at the bounds
        for percentile, value in list(return_percentiles.items()):
            if not open_lower_bound and value <= range_min + buffer:
                return_percentiles[percentile] = range_min + buffer
            if not open_upper_bound and value >= range_max - buffer:
                return_percentiles[percentile] = range_max - buffer

        # Set cdf values outside range
        if open_upper_bound:
            if range_max > return_percentiles[percentile_max]:
                halfway_between_max_and_100th_percentile = 100 - (0.5 * (100 - percentile_max))
                return_percentiles[halfway_between_max_and_100th_percentile] = range_max
        else:
            return_percentiles[100] = range_max

        # Set cdf values outside range
        if open_lower_bound:
            if range_min < return_percentiles[percentile_min]:
                halfway_between_min_and_0th_percentile = 0.5 * percentile_min
                return_percentiles[halfway_between_min_and_0th_percentile] = range_min
        else:
            return_percentiles[0] = range_min

        sorted_return_percentiles = dict(sorted(return_percentiles.items()))

        return_list = self._dict_to_percentile_list(sorted_return_percentiles, divide_by_100=True)
        return return_list

    def _nominal_location_to_cdf_location(self, nominal_value: float) -> float:
        return cdf_location(
            nominal_value,
            range_min=self.lower_bound,
            range_max=self.upper_bound,
            zero_point=self.zero_point,
        )

    def _get_cdf_at(self, cdf_location: float) -> float:
        """
        Helper function that takes a cdf location and returns
        the height (percentile) of the cdf at that location, linearly
        interpolating between values
        """
        bounded_percentiles = self._add_explicit_upper_lower_bound_percentiles(
            self.declared_percentiles
        )
        cdf_location_to_percentile_mapping: list[tuple[float, float]] = []
        for percentile in bounded_percentiles:
            height = percentile.percentile
            location = self._nominal_location_to_cdf_location(percentile.value)
            cdf_location_to_percentile_mapping.append((location, height))
        previous = cdf_location_to_percentile_mapping[0]
        for i in range(1, len(cdf_location_to_percentile_mapping)):
            current = cdf_location_to_percentile_mapping[i]
            epsilon = 1e-10
            if previous[0] - epsilon <= cdf_location <= current[0] + epsilon:
                result = previous[1] + (current[1] - previous[1]) * (cdf_location - previous[0]) / (
                    current[0] - previous[0]
                )
                if np.isnan(result):
                    raise ValueError(f"Result is NaN for cdf location {cdf_location}")
                return result
            previous = current
        raise ValueError(f"CDF location Input {cdf_location} cannot be found")

    def _standardize_cdf(self, cdf: list[float] | np.ndarray) -> list[float]:
        """Return a Metaculus-valid version of *cdf* for this question's bounds.

        Delegates to :func:`project_cdf_to_valid`, the single implementation of
        CDF-validity enforcement (min step, PMF cap, boundary mass, rounding).
        """
        return project_cdf_to_valid(
            cdf,
            open_lower_bound=self.open_lower_bound,
            open_upper_bound=self.open_upper_bound,
        )

    def _cdf_location_to_nominal_location(self, cdf_location: float) -> float:
        range_max = self.upper_bound
        range_min = self.lower_bound
        zero_point = self.zero_point

        if zero_point is None:
            scaled_location = range_min + (range_max - range_min) * cdf_location
        else:
            deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
            scaled_location = range_min + (range_max - range_min) * (
                deriv_ratio**cdf_location - 1
            ) / (deriv_ratio - 1)
        if np.isnan(scaled_location):
            raise ValueError(f"Scaled location is NaN for cdf location {cdf_location}")
        return scaled_location
