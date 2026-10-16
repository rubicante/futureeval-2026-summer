"""Tail-humility post-processing for LLM-ensemble forecasts.

Log scoring punishes overconfident misses catastrophically while sharp
correct forecasts gain little from the last bit of confidence. Blending
every continuous CDF with a widened copy of itself — and flooring every
MC option — is cheap insurance: ≈ log(1−weight) when the model is right,
a capped loss instead of a disaster when it's wrong.

Applied post-aggregation behind ``--fc tail=<weight>`` (see
``Strategy.run_from_context``). The quant series path carries its own
mixture and does not pass through here. Binary forecasts are already
clamped to [0.5%, 99.5%] and need nothing.
"""

from __future__ import annotations

import numpy as np

from forecasting.distributions import project_cdf_to_valid

# Widen factor for the background copy — same shape as the quant path's
# humility mixture (docs/financial-series-plan.md §3d).
WIDE_FACTOR = 3.0

# MC options are floored at this probability whenever tail humility is on.
# 2% ≈ log-loss cap of ~−390 instead of unbounded.
MC_FLOOR = 0.02


def widen_cdf(
    cdf: list[float],
    weight: float,
    factor: float = WIDE_FACTOR,
    *,
    lower_open: bool = True,
    upper_open: bool = True,
) -> list[float]:
    """CDF-space mixture of *cdf* with a copy widened ×*factor* about its median.

    Operates in rank space: F_wide(p) = F(m + (p − m)/factor) where m is the
    median's rank position — always an interior interpolation. The mixture is
    then projected onto Metaculus's validity set by
    :func:`forecasting.distributions.project_cdf_to_valid`, which enforces the
    step bounds and pins the endpoints for the question's bound type.

    *lower_open* / *upper_open* describe the bounds: an open bound keeps mass
    reserved outside the range, a closed bound forces the endpoint to exactly
    0.0 / 1.0. Defaults assume open bounds (the common financial-series case)
    for callers that do not know the bound type.
    """
    if weight <= 0:
        return cdf
    f = np.asarray(cdf, dtype=float)
    n = len(f)
    grid = np.linspace(0.0, 1.0, n)
    median_pos = float(np.interp(0.5, f, grid))
    wide = np.interp(median_pos + (grid - median_pos) / factor, grid, f)
    mixed = (1 - weight) * f + weight * wide
    return project_cdf_to_valid(mixed, open_lower_bound=lower_open, open_upper_bound=upper_open)


def floor_mc(probs: dict[str, float], floor: float = MC_FLOOR) -> dict[str, float]:
    """Raise every option to at least *floor*, keeping the sum at 1.

    Flooring-then-renormalizing (the obvious one-liner) re-shrinks the floored
    options below the floor again — e.g. {0.98, 0.01, 0.01} → floor → {0.98,
    0.02, 0.02}/1.02 → 0.0196 < 0.02. Instead, raise the sub-floor options and
    take the excess *only* from the options above the floor, iterating until
    all are ≥ floor. When the requested floor can't fit (floor·n > 1) it is
    first lowered to 1/n, which drives every option to the uniform 1/n.
    """
    if not probs:
        return probs
    keys = list(probs)
    floor = min(floor, 1.0 / len(keys))
    vals = np.clip(np.array([float(probs[k]) for k in keys], dtype=float), 0.0, None)
    vals = np.ones(len(keys)) if vals.sum() <= 0 else vals / vals.sum()
    for _ in range(100):
        below = vals < floor
        if not below.any():
            break
        deficit = float((floor - vals[below]).sum())
        vals[below] = floor
        above = vals > floor
        headroom = vals[above] - floor
        if not above.any() or headroom.sum() <= 0:
            break
        vals[above] -= deficit * (headroom / headroom.sum())
    return {k: float(v) for k, v in zip(keys, vals, strict=True)}


def apply_tail_humility(
    forecast,
    question_type: str,
    weight: float,
    *,
    lower_open: bool = True,
    upper_open: bool = True,
):
    """Dispatch by question type; returns the forecast unchanged when N/A.

    *lower_open* / *upper_open* describe the question's bounds so the CDF
    widen can re-pin closed endpoints (see :func:`widen_cdf`). They are
    ignored for MC and binary.
    """
    if weight <= 0:
        return forecast
    if question_type in ("numeric", "date", "discrete") and isinstance(forecast, list):
        return widen_cdf(forecast, weight, lower_open=lower_open, upper_open=upper_open)
    if question_type == "multiple_choice" and isinstance(forecast, dict):
        return floor_mc(forecast)
    return forecast
