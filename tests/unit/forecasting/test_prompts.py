"""Unit tests for forecasting.prompts — template rendering."""

from __future__ import annotations

import re

from common.models import ForecastQuestion, NumericScaling, QuestionType, build_question_context
from forecasting.primitives import _enrich_context
from forecasting.prompts import (
    BINARY_PROMPT_TEMPLATE,
    DATE_PROMPT_TEMPLATE,
    MULTIPLE_CHOICE_PROMPT_TEMPLATE,
    NUMERIC_PROMPT_TEMPLATE,
)


class TestBinaryPrompt:
    def test_renders_with_context(self):
        ctx = {
            "title": "Will it rain tomorrow?",
            "body": "Test body text.",
            "summary_report": "No research available.",
            "today": "2025-01-15",
            "resolve_time": "2025-02-01T00:00:00Z",
        }
        result = BINARY_PROMPT_TEMPLATE.format(**ctx)
        assert "Will it rain tomorrow?" in result
        assert "Test body text." in result
        assert "No research available." in result
        assert "Probability: ZZ%" in result

    def test_renders_without_summary_report(self):
        result = BINARY_PROMPT_TEMPLATE.format(
            title="Test",
            body="Body",
            summary_report="No report",
            today="2025-01-15",
            resolve_time="unknown",
        )
        assert "Test" in result


class TestNumericPrompt:
    def test_renders_with_full_context(self):
        ctx = {
            "title": "What will be the GDP?",
            "body": "Test body.",
            "units": "USD",
            "today": "2025-01-15",
            "resolve_time": "2025-02-01T00:00:00Z",
            "summary_report": "Research notes.",
            "lower_bound_message": "Outcome cannot be below 0.",
            "upper_bound_message": "Outcome cannot exceed 100.",
        }
        result = NUMERIC_PROMPT_TEMPLATE.format(**ctx)
        assert "What will be the GDP?" in result
        assert "USD" in result
        assert "Outcome cannot be below 0." in result
        assert "Outcome cannot exceed 100." in result
        assert "Percentile 10:" in result
        assert "Percentile 90:" in result

    def test_renders_without_bound_messages(self):
        result = NUMERIC_PROMPT_TEMPLATE.format(
            title="Test",
            body="Body",
            units="units",
            today="2025-01-15",
            resolve_time="unknown",
            summary_report="Report",
            lower_bound_message="",
            upper_bound_message="",
        )
        assert "Test" in result


class TestDatePrompt:
    def test_renders_with_full_context(self):
        ctx = {
            "title": "When will AI surpass humans?",
            "body": "Test body.",
            "summary_report": "Research notes.",
            "today": "2025-01-15",
            "resolve_time": "2026-01-01T00:00:00Z",
            "lower_bound_message": "The outcome can not be earlier than 2025-01-01.",
            "upper_bound_message": "The outcome can not be later than 2100-01-01.",
        }
        result = DATE_PROMPT_TEMPLATE.format(**ctx)
        assert "When will AI surpass humans?" in result
        assert "2025-01-15" in result
        assert "Percentile 10:" in result
        assert "YYYY-MM-DD" in result


class TestMultipleChoicePrompt:
    def test_renders_with_options(self):
        ctx = {
            "title": "Who will win?",
            "body": "Test body.",
            "summary_report": "Research notes.",
            "options": ["Democrat", "Republican", "Other"],
            "today": "2025-01-15",
            "resolve_time": "2025-02-01T00:00:00Z",
        }
        result = MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(**ctx)
        assert "Who will win?" in result
        assert "Democrat" in result
        assert "Republican" in result
        assert "Other" in result


# ── Mechanical completeness: every {key} in every template is in the context ──


def _template_keys(template: str) -> set[str]:
    return set(re.findall(r"\{(\w+)\}", template))


_TEMPLATE_TYPE_MAP: list[tuple[str, QuestionType]] = [
    (BINARY_PROMPT_TEMPLATE, QuestionType.BINARY),
    (NUMERIC_PROMPT_TEMPLATE, QuestionType.NUMERIC),
    (DATE_PROMPT_TEMPLATE, QuestionType.DATE),
    (MULTIPLE_CHOICE_PROMPT_TEMPLATE, QuestionType.MULTIPLE_CHOICE),
]


def _make_representative_question(qtype: QuestionType) -> ForecastQuestion:
    """Return a minimal ForecastQuestion that exercises the full context pipeline."""
    if qtype == QuestionType.BINARY:
        return ForecastQuestion(
            title="Test Q",
            body="Test body.",
            question_type=QuestionType.BINARY,
            resolve_time="2030-01-01T00:00:00Z",
        )
    if qtype in (QuestionType.NUMERIC, QuestionType.DISCRETE):
        return ForecastQuestion(
            title="Test Q",
            body="Test body.",
            question_type=qtype,
            scaling=NumericScaling(range_min=0.0, range_max=100.0),
            unit="USD",
            resolve_time="2030-01-01T00:00:00Z",
        )
    if qtype == QuestionType.DATE:
        return ForecastQuestion(
            title="Test Q",
            body="Test body.",
            question_type=QuestionType.DATE,
            scaling=NumericScaling(range_min=1700000000.0, range_max=2000000000.0),
            resolve_time="2030-01-01T00:00:00Z",
        )
    return ForecastQuestion(
        title="Test Q",
        body="Test body.",
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=["A", "B", "C"],
        resolve_time="2030-01-01T00:00:00Z",
    )


def test_all_template_keys_are_in_context():
    """Every ``{key}`` in every forecast template is produced by the context pipeline
    for the matching question type."""
    for template, qtype in _TEMPLATE_TYPE_MAP:
        question = _make_representative_question(qtype)
        ctx = build_question_context(question)
        ctx = _enrich_context(question, ctx)

        needed = _template_keys(template)
        available = set(ctx.keys())
        missing = needed - available

        assert not missing, (
            f"{qtype.value}: template references keys {sorted(missing)} "
            f"not in context dict (available: {sorted(available)})"
        )
