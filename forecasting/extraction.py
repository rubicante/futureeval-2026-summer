"""Pydantic extraction schemas for structured LLM output parsing."""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field, field_validator


class BinaryExtraction(BaseModel):
    probability_percent: float = Field(ge=0, le=100)


class NumericExtraction(BaseModel):
    percentile_10: float
    percentile_20: float
    percentile_40: float
    percentile_60: float
    percentile_80: float
    percentile_90: float


class DateExtraction(BaseModel):
    percentile_10: str
    percentile_20: str
    percentile_40: str
    percentile_60: str
    percentile_80: str
    percentile_90: str

    @field_validator(
        "percentile_10",
        "percentile_20",
        "percentile_40",
        "percentile_60",
        "percentile_80",
        "percentile_90",
        mode="after",
    )
    @classmethod
    def _validate_date_format(cls, v: str) -> str:
        # Accept "YYYY-MM-DD" or full ISO-8601 (with time, optional Z)
        try:
            # fromisoformat handles YYYY-MM-DD and YYYY-MM-DDTHH:MM:SS etc.
            # Replace trailing Z for UTC compatibility.
            dt.datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError as e:
            raise ValueError(
                f"Date field must be a valid ISO date (YYYY-MM-DD or ISO-8601), got: {v}"
            ) from e
        return v


class OptionPrediction(BaseModel):
    name: str = Field(description="The option name. Must match one of the valid options exactly.")
    probability: float = Field(description="Probability between 0 and 1")
