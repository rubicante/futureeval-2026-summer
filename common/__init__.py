"""Common / shared domain models.

This package holds types that are used across the forecasting, research,
and platform adapter layers (e.g. ForecastQuestion and helpers).
No external dependencies or side effects belong here.
"""

from .models import (
    ForecastQuestion,
    NumericScaling,
    QuestionType,
    build_question_context,
)

__all__ = [
    "ForecastQuestion",
    "NumericScaling",
    "QuestionType",
    "build_question_context",
]
