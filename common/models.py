"""Shared domain models.

Covers question structures, eval-capture records, and run-config snapshots.
"""

from __future__ import annotations

import datetime
import enum
from datetime import datetime as Datetime  # avoid shadowing the module for UTC
from typing import Any

from pydantic import BaseModel, Field


class QuestionType(enum.StrEnum):
    BINARY = "binary"
    NUMERIC = "numeric"
    DISCRETE = "discrete"
    MULTIPLE_CHOICE = "multiple_choice"
    DATE = "date"


class NumericScaling(BaseModel):
    range_min: float
    range_max: float
    zero_point: float | None = None
    open_lower_bound: bool = False
    open_upper_bound: bool = False
    cdf_size: int = 201
    nominal_min: float | None = None
    nominal_max: float | None = None
    continuous_range: list | None = None


class ForecastQuestion(BaseModel):
    title: str
    body: str  # description + resolution criteria + fine print, in order
    question_type: QuestionType
    options: list[str] | None = None
    scaling: NumericScaling | None = None
    unit: str | None = None
    resolve_time: str | None = None
    community_prediction: str | None = None
    community_prediction_raw: dict[str, Any] | None = None
    url: str | None = None


def build_question_context(question: ForecastQuestion) -> dict[str, Any]:
    result = question.model_dump()
    # The raw aggregations dict never feeds prompts and is captured as its
    # own field on CaptureRecord — keep it out of the forecast context.
    # The formatted string is likewise unused by any template (and empty
    # while tournament questions are open — CP is hidden until close).
    result.pop("community_prediction_raw", None)
    result.pop("community_prediction", None)
    result.setdefault("summary_report", "")
    result.setdefault("hours_remaining", "")
    result["today"] = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
    if not result.get("resolve_time"):
        result["resolve_time"] = "unknown"
    if question.resolve_time:
        try:
            resolve_dt = datetime.datetime.fromisoformat(question.resolve_time)
            if resolve_dt.tzinfo is None:
                resolve_dt = resolve_dt.replace(tzinfo=datetime.UTC)
            remaining = resolve_dt - datetime.datetime.now(datetime.UTC)
            hours = round(remaining.total_seconds() / 3600)
            if hours >= 0:
                result["hours_remaining"] = hours
        except ValueError:
            pass
    return result


# ── eval-capture models ──────────────────────────────────────────────

CURRENT_RUN_CONFIG_VERSION = 1


class RunConfig(BaseModel):
    """Snapshot of the run configuration for capture records.

    A projection of the argparse SimpleNamespace — typed, serializable,
    immutable in the capture.  The original argparse path stays untouched.
    """

    schema_version: int
    research_strategy: str  # "planned_research" | "no_research"
    research_provider: str  # "exa" | "asknews"
    forecast_strategy: str  # "ensemble"
    forecast_config: dict[str, Any] = {}
    research_config: dict[str, Any] = {}
    forecast_model: str
    decompose_model: str
    forecast_models: dict[str, str] = {}

    def migrate(self) -> RunConfig:
        """Upgrade to CURRENT_RUN_CONFIG_VERSION.  Idempotent — safe to call on
        already-current records."""
        if self.schema_version >= CURRENT_RUN_CONFIG_VERSION:
            return self
        data = self.model_dump()
        # Placeholder: future v1→v2 migration goes here.
        data["schema_version"] = CURRENT_RUN_CONFIG_VERSION
        return RunConfig(**data)


class CaptureRecord(BaseModel):
    """Immutable capture record — one JSONL row per (question, run).

    Append-only.  Never edit a record once written.
    """

    format_version: int = 1
    code_version: str | None = None
    run_id: str
    question_id: int
    snapshot_timestamp: Datetime

    run_config: RunConfig
    context: dict[str, Any]
    raw_results: list[tuple[Any, str]]
    sample_models: list[str] | None = None
    final_forecast: Any
    community_prediction: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Raw API aggregations dict (not the formatted str from ForecastQuestion). "
            "Null while tournament questions are open — CP is hidden until close."
        ),
    )
    cost: dict[str, Any] | None = None  # CostLedger per-question tally (tokens + est. USD)


class ResolutionRecord(BaseModel):
    """Resolved outcome for one question — loss-agnostic.

    Stores the raw outcome plus the axis spec so any scoring function
    (log-score, CRPS, Brier) can be computed later without re-fetching
    from the API.
    """

    question_id: int
    resolved_at: Datetime | None
    resolution: Any
    question_type: str
    scaling: dict[str, Any] | None = None
    options: list[str] | None = None
