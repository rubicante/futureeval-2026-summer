"""Forecasting primitives — parse, aggregate, trim, format.

These are question-type-specific but strategy-agnostic plumbing functions.
"""

from __future__ import annotations

import datetime as dt
import difflib
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

import numpy as np

from common.models import ForecastQuestion, NumericScaling, QuestionType
from forecasting.distributions import generate_continuous_cdf
from llm.gateway import Gateway, LLMRequest

from .extraction import BinaryExtraction, DateExtraction, NumericExtraction, OptionPrediction
from .prompts import (
    BINARY_PROMPT_TEMPLATE,
    DATE_PROMPT_TEMPLATE,
    MULTIPLE_CHOICE_PROMPT_TEMPLATE,
    NUMERIC_PROMPT_TEMPLATE,
)

T = TypeVar("T")


@dataclass(frozen=True)
class ParseDeps:
    """Dependency bundle for parse functions — constructed once per forecast run."""

    gateway: Gateway
    scaling: NumericScaling | None = None
    options: list[str] | None = None


@dataclass(frozen=True)
class Primitive(Generic[T]):
    """Immutable, typed, one instance per question type.

    The type parameter ``T`` ties the return of ``parse`` to the input of
    ``aggregate`` and ``trim`` — they must agree.
    """

    prompt_template: str
    parse: Callable[[str, ParseDeps], Awaitable[T]]
    aggregate: Callable[[list[T]], T]
    trim: Callable[[list[T]], list[int]]
    dispersion: Callable[[list[T]], float]
    supervisor_threshold: float
    output_format_hint: str
    aggregate_weighted: Callable[[list[T], list[float]], T] | None = None


# ═══════════════════════════════════════════════════════════════════
#  Parse functions
# ═══════════════════════════════════════════════════════════════════


async def parse_binary_probability(rationale: str, deps: ParseDeps) -> float:
    extraction = await deps.gateway.extract_structured(
        LLMRequest(prompt=rationale, purpose="extract"), BinaryExtraction
    )
    return min(99.5, max(0.5, extraction.probability_percent)) / 100


async def parse_numeric_cdf(rationale: str, deps: ParseDeps) -> list[float]:
    scaling = deps.scaling
    if scaling is None:
        raise ValueError("Numeric question requires scaling")
    extraction = await deps.gateway.extract_structured(
        LLMRequest(prompt=rationale, purpose="extract"), NumericExtraction
    )
    percentiles = {
        10: extraction.percentile_10,
        20: extraction.percentile_20,
        40: extraction.percentile_40,
        60: extraction.percentile_60,
        80: extraction.percentile_80,
        90: extraction.percentile_90,
    }
    return generate_continuous_cdf(
        percentiles,
        scaling.open_upper_bound,
        scaling.open_lower_bound,
        scaling.range_max,
        scaling.range_min,
        scaling.zero_point,
        scaling.cdf_size,
    )


async def parse_date_cdf(rationale: str, deps: ParseDeps) -> list[float]:
    scaling = deps.scaling
    if scaling is None:
        raise ValueError("Date question requires scaling")
    extraction = await deps.gateway.extract_structured(
        LLMRequest(prompt=rationale, purpose="extract"), DateExtraction
    )
    percentiles: dict[int, float] = {}
    for key in (10, 20, 40, 60, 80, 90):
        date_str = getattr(extraction, f"percentile_{key}")
        try:
            ts = dt.datetime.fromisoformat(date_str).replace(tzinfo=dt.UTC).timestamp()
            percentiles[key] = ts
        except Exception as e:
            raise ValueError(f"Failed to parse date percentile {key}='{date_str}'") from e
    return generate_continuous_cdf(
        percentiles,
        scaling.open_upper_bound,
        scaling.open_lower_bound,
        scaling.range_max,
        scaling.range_min,
        scaling.zero_point,
        scaling.cdf_size,
    )


# "Option_A" / "option B" / "(C)" / bare "d" — the placeholder shapes models
# emit when they copy a lettered format illustration instead of option text.
_MCQ_PLACEHOLDER_RE = re.compile(r"^\(?(?:option[\s_-]*)?([a-z])\)?\.?$", re.IGNORECASE)


def _placeholder_index(key: str) -> int | None:
    match = _MCQ_PLACEHOLDER_RE.match(key.strip())
    return ord(match.group(1).casefold()) - ord("a") if match else None


def _align_mcq_options(raw: dict[str, float], options: list[str]) -> dict[str, float]:
    """Re-key extraction output onto the question's canonical option names.

    Exact (casefolded) match first, fuzzy fallback for paraphrases, then
    lettered placeholders ("Option_A") mapped by position — the prompt asks
    for probabilities in canonical option order, so A ↔ options[0]. Missing
    options filled at the floor. Raises only when nothing matches at all.
    """
    leftover = dict(raw)
    aligned: dict[str, float] = {}
    for opt in options:
        key = next((k for k in leftover if k.casefold().strip() == opt.casefold().strip()), None)
        if key is not None:
            aligned[opt] = leftover.pop(key)
    for opt in options:
        if opt in aligned:
            continue
        close = difflib.get_close_matches(opt, list(leftover), n=1, cutoff=0.6)
        if not close:
            of = opt.casefold().strip()
            close = [k for k in leftover if of in k.casefold() or k.casefold().strip() in of][:1]
        if close:
            aligned[opt] = leftover.pop(close[0])
    for idx, opt in enumerate(options):
        if opt in aligned:
            continue
        key = next((k for k in leftover if _placeholder_index(k) == idx), None)
        if key is not None:
            aligned[opt] = leftover.pop(key)
    if not aligned:
        raise ValueError(f"Extracted option names {list(raw)} match none of the options {options}")
    return {opt: aligned.get(opt, 0.001) for opt in options}


async def parse_multiple_choice(rationale: str, deps: ParseDeps) -> dict[str, float]:
    prompt = rationale
    if deps.options:
        # The extraction schema demands exact option names, so the extractor
        # must be shown what they are — the rationale alone may only contain
        # placeholders like "Option_A".
        option_list = "\n".join(f"- {opt}" for opt in deps.options)
        prompt = (
            f"The question's options, in order (extracted names must match these exactly; "
            f'map any lettered placeholders like "Option_A" to them by position):\n'
            f"{option_list}\n\n{rationale}"
        )
    extraction = await deps.gateway.extract_structured(
        LLMRequest(prompt=prompt, purpose="extract"), list[OptionPrediction]
    )
    raw = {p.name: p.probability for p in extraction}
    if deps.options:
        raw = _align_mcq_options(raw, deps.options)
    clamped = {k: max(0.001, min(0.999, v)) for k, v in raw.items()}
    total = sum(clamped.values())
    return {k: v / total for k, v in clamped.items()}


# ═══════════════════════════════════════════════════════════════════
#  Aggregate functions
# ═══════════════════════════════════════════════════════════════════


def aggregate_scalar_median(values: list[float]) -> float:
    return float(np.median(values))


def aggregate_array_median(arrays: list[list[float]]) -> list[float]:
    return np.median(np.array(arrays), axis=0).tolist()


def aggregate_dict_mean(dicts: list[dict[str, float]]) -> dict[str, float]:
    if not dicts:
        return {}
    options = list(dicts[0].keys())
    return {opt: sum(d.get(opt, 0.0) for d in dicts) / len(dicts) for opt in options}


# ═══════════════════════════════════════════════════════════════════
#  Weighted aggregate functions
# ═══════════════════════════════════════════════════════════════════


def _weighted_median_scalar(values: list[float], weights: list[float]) -> float:
    """Weighted median — sort by value, walk cumulative weight to the 0.5 mark.

    When the cumulative weight lands exactly at the half-total mark between
    two values, the two straddling values are averaged (matching the
    interpolation behaviour of the unweighted median for even counts).
    """
    if not values:
        raise ValueError("empty values")
    pairs = sorted(zip(values, weights, strict=True), key=lambda x: x[0])
    total_weight = sum(weights)
    if total_weight == 0:
        return float(np.median(values))
    half = total_weight / 2

    cum = 0.0
    for i, (v, w) in enumerate(pairs):
        if cum + w > half:
            return float(v)
        if cum + w == half:
            if i + 1 < len(pairs):
                return float((v + pairs[i + 1][0]) / 2)
            return float(v)
        cum += w

    return float(pairs[-1][0])


def aggregate_weighted_scalar_median(values: list[float], weights: list[float]) -> float:
    return _weighted_median_scalar(values, weights)


def aggregate_weighted_scalar_mean(values: list[float], weights: list[float]) -> float:
    total = sum(weights)
    if total == 0:
        return float(np.mean(values))
    return float(np.dot(values, weights) / total)


def aggregate_weighted_array_median(arrays: list[list[float]], weights: list[float]) -> list[float]:
    arr = np.array(arrays)
    n_rows, n_cols = arr.shape
    result: list[float] = []
    for col in range(n_cols):
        col_vals = arr[:, col].tolist()
        result.append(_weighted_median_scalar(col_vals, weights))
    return result


def aggregate_weighted_dict_mean(
    dicts: list[dict[str, float]], weights: list[float]
) -> dict[str, float]:
    if not dicts:
        return {}
    total_weight = sum(weights)
    if total_weight == 0:
        return aggregate_dict_mean(dicts)
    options = list(dicts[0].keys())
    return {
        opt: sum(d.get(opt, 0.0) * w for d, w in zip(dicts, weights, strict=True)) / total_weight
        for opt in options
    }


# ═══════════════════════════════════════════════════════════════════
#  Trim functions
# ═══════════════════════════════════════════════════════════════════


def trim_scalar_outliers(values: list[float]) -> list[int]:
    if len(values) <= 3:
        return list(range(len(values)))
    consensus = float(np.median(values))
    distances = [abs(v - consensus) for v in values]
    median_dist = float(np.median(distances))
    if median_dist == 0:
        return list(range(len(values)))
    return [i for i, d in enumerate(distances) if d <= 3.0 * median_dist]


def trim_array_outliers(arrays: list[list[float]]) -> list[int]:
    if len(arrays) <= 3:
        return list(range(len(arrays)))
    consensus = np.median(np.array(arrays), axis=0).tolist()
    distances = [float(np.sum(np.abs(np.array(a) - np.array(consensus)))) for a in arrays]
    median_dist = float(np.median(distances))
    if median_dist == 0:
        return list(range(len(arrays)))
    return [i for i, d in enumerate(distances) if d <= 3.0 * median_dist]


def trim_dict_outliers(dicts: list[dict[str, float]]) -> list[int]:
    if len(dicts) <= 3:
        return list(range(len(dicts)))
    options = sorted(dicts[0].keys())
    consensus = {opt: float(np.mean([d.get(opt, 0.0) for d in dicts])) for opt in options}
    distances = [
        float(np.sum([abs(d.get(opt, 0.0) - consensus[opt]) for opt in options])) for d in dicts
    ]
    median_dist = float(np.median(distances))
    if median_dist == 0:
        return list(range(len(dicts)))
    return [i for i, dist in enumerate(distances) if dist <= 3.0 * median_dist]


# ═══════════════════════════════════════════════════════════════════
#  Dispersion functions — unit-free ensemble disagreement metrics
# ═══════════════════════════════════════════════════════════════════


def _cdf_median_position(cdf: list[float]) -> float:
    """Position (0–1) where the CDF crosses 0.5. Uses linear interpolation.

    Returns a unit-free scalar that is comparable across CDFs of any
    real-world range.  For a 201-point CDF, crossing at index 100 gives
    position 0.5.
    """
    n = len(cdf)
    if n < 2:
        return 0.5
    for i in range(n):
        if cdf[i] >= 0.5:
            if i == 0:
                return 0.0
            lo, hi = cdf[i - 1], cdf[i]
            if hi == lo:
                return i / (n - 1)
            frac = (0.5 - lo) / (hi - lo)
            return (i - 1 + frac) / (n - 1)
    return 1.0


def _dispersion_binary(values: list[float]) -> float:
    """Standard deviation of binary probabilities — range [0, 0.5]."""
    if len(values) < 2:
        return 0.0
    return float(np.std(values))


def _dispersion_cdf(values: list[list[float]]) -> float:
    """Std of CDF median positions — unit-free, range roughly [0, ~0.5].

    Extracts the 0.5-crossing position from each CDF (via
    ``_cdf_median_position``) and returns the standard deviation of those
    positions.  This catches central-tendency disagreement without
    depending on the real-world units of the quantity.
    """
    if len(values) < 2:
        return 0.0
    positions = [_cdf_median_position(cdf) for cdf in values]
    return float(np.std(positions))


def _mcq_tvd(a: dict[str, float], b: dict[str, float]) -> float:
    """Total variation distance between two probability distributions."""
    options = sorted(set(a.keys()) | set(b.keys()))
    return 0.5 * sum(abs(a.get(opt, 0.0) - b.get(opt, 0.0)) for opt in options)


def _dispersion_mcq(values: list[dict[str, float]]) -> float:
    """Mean pairwise total variation distance — range [0, 1].

    Catches both magnitude disagreement (different confidence on the same
    winner) and identity disagreement (different forecasters pick different
    winners).
    """
    n = len(values)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += _mcq_tvd(values[i], values[j])
            count += 1
    return total / count


# ═══════════════════════════════════════════════════════════════════
#  CDF / display helpers
# ═══════════════════════════════════════════════════════════════════


def _cdf_quantile(cdf: list[float], scaling: NumericScaling, q: float) -> float:
    """Extract the real-world value at a given CDF probability ``q`` (0–1)."""
    cdf_size = len(cdf)
    idx = next((i for i, p in enumerate(cdf) if p >= q), cdf_size - 1)
    loc = idx / (cdf_size - 1)
    if scaling.zero_point is not None:
        deriv_ratio = (scaling.range_max - scaling.zero_point) / (
            scaling.range_min - scaling.zero_point
        )
        value = scaling.range_min + (scaling.range_max - scaling.range_min) * (
            (deriv_ratio**loc - 1) / (deriv_ratio - 1)
        )
    else:
        value = scaling.range_min + loc * (scaling.range_max - scaling.range_min)
    return value


def _timestamp_to_date(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, tz=dt.UTC).strftime("%Y-%m-%d")


def _build_date_bound_messages(scaling: NumericScaling) -> tuple[str, str]:
    upper_date = _timestamp_to_date(scaling.range_max)
    lower_date = _timestamp_to_date(scaling.range_min)
    if scaling.open_upper_bound:
        upper_msg = f"The question creator thinks the date is likely not later than {upper_date}."
    else:
        upper_msg = f"The outcome can not be later than {upper_date}."
    if scaling.open_lower_bound:
        lower_msg = f"The question creator thinks the date is likely not earlier than {lower_date}."
    else:
        lower_msg = f"The outcome can not be earlier than {lower_date}."
    return lower_msg, upper_msg


# ═══════════════════════════════════════════════════════════════════
#  Context enrichment
# ═══════════════════════════════════════════════════════════════════


def _enrich_context(question: ForecastQuestion, ctx: dict[str, Any]) -> dict[str, Any]:
    """Add question-type-specific derived fields required by prompt templates."""
    qtype = question.question_type
    scaling = question.scaling

    if (
        qtype in (QuestionType.NUMERIC, QuestionType.DISCRETE, QuestionType.DATE)
        and scaling is not None
    ):
        if qtype == QuestionType.DATE:
            lower_msg, upper_msg = _build_date_bound_messages(scaling)
            ctx["lower_bound_message"] = lower_msg
            ctx["upper_bound_message"] = upper_msg
        else:
            upper_msg = (
                "" if scaling.open_upper_bound else f"Outcome cannot exceed {scaling.range_max}."
            )
            lower_msg = (
                "" if scaling.open_lower_bound else f"Outcome cannot be below {scaling.range_min}."
            )
            ctx["lower_bound_message"] = lower_msg
            ctx["upper_bound_message"] = upper_msg
            ctx["units"] = question.unit or "Not stated (please infer this)"

    return ctx


# ═══════════════════════════════════════════════════════════════════
#  Format forecast
# ═══════════════════════════════════════════════════════════════════


def format_forecast(forecast: object, question: ForecastQuestion) -> str:
    qtype = question.question_type
    if qtype in (QuestionType.NUMERIC, QuestionType.DISCRETE, QuestionType.DATE):
        cdf = cast("list[float]", forecast)
        scaling = cast("NumericScaling", question.scaling)
        if qtype == QuestionType.DATE:
            median = _cdf_quantile(cdf, scaling, 0.5)
            p10 = _cdf_quantile(cdf, scaling, 0.1)
            p90 = _cdf_quantile(cdf, scaling, 0.9)
            return (
                f"Median: {_timestamp_to_date(median)}  "
                f"(10th–90th: {_timestamp_to_date(p10)} – {_timestamp_to_date(p90)})"
            )
        unit_str = f" {question.unit}" if question.unit else ""
        median = _cdf_quantile(cdf, scaling, 0.5)
        p10 = _cdf_quantile(cdf, scaling, 0.1)
        p90 = _cdf_quantile(cdf, scaling, 0.9)
        return f"Median: {median:.2f}{unit_str}  (10th–90th: {p10:.2f} – {p90:.2f}{unit_str})"
    if qtype == QuestionType.BINARY:
        return f"Median Probability: {forecast:.2%}"
    if qtype == QuestionType.MULTIPLE_CHOICE:
        probs = cast("dict[str, float]", forecast)
        parts = [f"{opt} {prob:.0%}" for opt, prob in probs.items()]
        return "Average: " + " | ".join(parts)
    return str(forecast)


# ═══════════════════════════════════════════════════════════════════
#  Primitive registry — one entry per question type
# ═══════════════════════════════════════════════════════════════════

PRIMITIVES: dict[str, Primitive[Any]] = {
    "binary": Primitive[float](
        prompt_template=BINARY_PROMPT_TEMPLATE,
        parse=parse_binary_probability,
        aggregate=aggregate_scalar_median,
        aggregate_weighted=aggregate_weighted_scalar_median,
        trim=trim_scalar_outliers,
        dispersion=_dispersion_binary,  # type: ignore[arg-type]
        supervisor_threshold=0.15,
        output_format_hint=(
            'The last thing you write is your final answer as: "Probability: ZZ%", 0-100'
        ),
    ),
    "numeric": Primitive[list[float]](
        prompt_template=NUMERIC_PROMPT_TEMPLATE,
        parse=parse_numeric_cdf,
        aggregate=aggregate_array_median,
        aggregate_weighted=aggregate_weighted_array_median,
        trim=trim_array_outliers,
        dispersion=_dispersion_cdf,  # type: ignore[arg-type]
        supervisor_threshold=0.12,
        output_format_hint=(
            "The last thing you write is your final answer as:\n"
            '"'
            "Percentile 10: XX\n"
            "Percentile 20: XX\n"
            "Percentile 40: XX\n"
            "Percentile 60: XX\n"
            "Percentile 80: XX\n"
            "Percentile 90: XX\n"
            '"'
        ),
    ),
    "discrete": Primitive[list[float]](
        prompt_template=NUMERIC_PROMPT_TEMPLATE,
        parse=parse_numeric_cdf,
        aggregate=aggregate_array_median,
        aggregate_weighted=aggregate_weighted_array_median,
        trim=trim_array_outliers,
        dispersion=_dispersion_cdf,  # type: ignore[arg-type]
        supervisor_threshold=0.12,
        output_format_hint=(
            "The last thing you write is your final answer as:\n"
            '"'
            "Percentile 10: XX\n"
            "Percentile 20: XX\n"
            "Percentile 40: XX\n"
            "Percentile 60: XX\n"
            "Percentile 80: XX\n"
            "Percentile 90: XX\n"
            '"'
        ),
    ),
    "date": Primitive[list[float]](
        prompt_template=DATE_PROMPT_TEMPLATE,
        parse=parse_date_cdf,
        aggregate=aggregate_array_median,
        aggregate_weighted=aggregate_weighted_array_median,
        trim=trim_array_outliers,
        dispersion=_dispersion_cdf,  # type: ignore[arg-type]
        supervisor_threshold=0.12,
        output_format_hint=(
            "The last thing you write is your final answer as:\n"
            '"'
            "Percentile 10: YYYY-MM-DD (oldest date)\n"
            "Percentile 20: YYYY-MM-DD\n"
            "Percentile 40: YYYY-MM-DD\n"
            "Percentile 60: YYYY-MM-DD\n"
            "Percentile 80: YYYY-MM-DD\n"
            "Percentile 90: YYYY-MM-DD (newest date)\n"
            '"'
        ),
    ),
    "multiple_choice": Primitive[dict[str, float]](
        prompt_template=MULTIPLE_CHOICE_PROMPT_TEMPLATE,
        parse=parse_multiple_choice,
        aggregate=aggregate_dict_mean,
        aggregate_weighted=aggregate_weighted_dict_mean,
        trim=trim_dict_outliers,
        dispersion=_dispersion_mcq,  # type: ignore[arg-type]
        supervisor_threshold=0.18,
        output_format_hint=(
            "The last thing you write is your final probabilities for the N options"
            " in this order {options} as:\n"
            "Option_A: Probability_A\n"
            "Option_B: Probability_B\n"
            "...\n"
            "Option_N: Probability_N"
        ),
    ),
}
