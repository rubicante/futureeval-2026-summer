"""Metaculus ↔ canonical model adapter.

Converts Metaculus API response dicts into platform-agnostic
ForecastQuestion instances, and canonical forecast values back
into Metaculus API submission payloads.
"""

from __future__ import annotations

from common.models import ForecastQuestion, NumericScaling, QuestionType


class MetaculusAdapter:
    """Convert between Metaculus API shapes and canonical forecast models."""

    @staticmethod
    def _format_community_prediction(
        aggregations: dict, q_type: str, options: list[str] | None = None
    ) -> str | None:
        """Format the recency-weighted community prediction as a human-readable string."""
        latest = (aggregations.get("recency_weighted", {}) or {}).get("latest")
        if not latest:
            return None
        if q_type == "binary":
            p = latest.get("probability_yes")
            if p is not None:
                return f"{p:.0%} Yes"
        elif q_type in ("numeric", "discrete", "date"):
            cdf = latest.get("continuous_cdf")
            if cdf and len(cdf) == 201:
                idx = next((i for i, p in enumerate(cdf) if p >= 0.5), 100)
                pct = idx / (len(cdf) - 1) * 100
                return f"median at {pct:.0f}th percentile of CDF range"
            return "community CDF available"
        elif q_type == "multiple_choice":
            cat = latest.get("probability_yes_per_category")
            if cat and options:
                parts = [f"{opt}: {cat.get(opt, 0):.0%}" for opt in options]
                return ", ".join(parts)
        return None

    @staticmethod
    def question_from_api(post_details: dict) -> ForecastQuestion:
        """Build a canonical ForecastQuestion from a Metaculus post details dict."""

        post_id = post_details["id"]
        q = post_details["question"]
        q_type = q["type"]
        scaling_raw = q.get("scaling")
        if (
            scaling_raw
            and scaling_raw.get("range_min") is not None
            and scaling_raw.get("range_max") is not None
        ):
            cdf_size = scaling_raw["inbound_outcome_count"] + 1 if q_type == "discrete" else 201
            scaling = NumericScaling(
                range_min=scaling_raw["range_min"],
                range_max=scaling_raw["range_max"],
                zero_point=scaling_raw.get("zero_point"),
                open_lower_bound=q.get("open_lower_bound") or False,
                open_upper_bound=q.get("open_upper_bound") or False,
                cdf_size=cdf_size,
                nominal_min=scaling_raw.get("nominal_min"),
                nominal_max=scaling_raw.get("nominal_max"),
                continuous_range=scaling_raw.get("continuous_range"),
            )
        else:
            scaling = None

        body_parts = [
            q.get("description", ""),
            q.get("resolution_criteria", ""),
            q.get("fine_print", ""),
        ]
        body = "\n\n".join(p for p in body_parts if p)

        options = q.get("options")
        aggregations: dict = q.get("aggregations", {})
        community_prediction = MetaculusAdapter._format_community_prediction(
            aggregations, q_type, options
        )

        return ForecastQuestion(
            title=q["title"],
            body=body,
            question_type=QuestionType(q_type),
            options=options,
            scaling=scaling,
            unit=q.get("unit"),
            resolve_time=q.get("scheduled_resolve_time"),
            community_prediction=community_prediction,
            community_prediction_raw=aggregations or None,
            url=f"https://www.metaculus.com/questions/{post_id}/",
        )

    @staticmethod
    def forecast_to_payload(
        forecast: float | dict[str, float] | list[float],
        question_type: str,
    ) -> dict:
        """Convert a forecast value into a Metaculus API submission payload."""
        if question_type == "binary":
            return {
                "probability_yes": forecast,
                "probability_yes_per_category": None,
                "continuous_cdf": None,
            }
        if question_type == "multiple_choice":
            return {
                "probability_yes": None,
                "probability_yes_per_category": forecast,
                "continuous_cdf": None,
            }
        return {
            "probability_yes": None,
            "probability_yes_per_category": None,
            "continuous_cdf": forecast,
        }
