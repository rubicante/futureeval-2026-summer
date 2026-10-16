"""Unit tests for common.models — domain models and helpers."""

from __future__ import annotations

import datetime
from unittest import mock

from common.models import (
    ForecastQuestion,
    NumericScaling,
    QuestionType,
    ResolutionRecord,
    build_question_context,
)


class TestQuestionType:
    def test_valid_values(self):
        assert QuestionType.BINARY.value == "binary"
        assert QuestionType.NUMERIC.value == "numeric"
        assert QuestionType.DISCRETE.value == "discrete"
        assert QuestionType.MULTIPLE_CHOICE.value == "multiple_choice"
        assert QuestionType.DATE.value == "date"

    def test_from_string(self):
        assert QuestionType("binary") == QuestionType.BINARY


class TestNumericScaling:
    def test_defaults(self):
        s = NumericScaling(range_min=0.0, range_max=100.0)
        assert s.range_min == 0.0
        assert s.range_max == 100.0
        assert s.zero_point is None
        assert s.open_lower_bound is False
        assert s.open_upper_bound is False
        assert s.cdf_size == 201

    def test_with_zero_point(self):
        s = NumericScaling(range_min=10.0, range_max=1000.0, zero_point=1.0)
        assert s.zero_point == 1.0

    def test_open_bounds(self):
        s = NumericScaling(
            range_min=0.0, range_max=100.0, open_lower_bound=True, open_upper_bound=True
        )
        assert s.open_lower_bound is True
        assert s.open_upper_bound is True

    def test_custom_cdf_size(self):
        s = NumericScaling(range_min=0.0, range_max=100.0, cdf_size=51)
        assert s.cdf_size == 51


class TestForecastQuestion:
    def test_minimal_construction(self):
        q = ForecastQuestion(
            title="Test?",
            body="Test body.",
            question_type=QuestionType.BINARY,
        )
        assert q.title == "Test?"
        assert q.body == "Test body."
        assert q.options is None
        assert q.scaling is None
        assert q.unit is None
        assert q.resolve_time is None
        assert q.community_prediction is None
        assert q.url is None

    def test_full_construction(self):
        q = ForecastQuestion(
            title="Test?",
            body="Test body.",
            question_type=QuestionType.NUMERIC,
            options=["A", "B"],
            scaling=NumericScaling(range_min=0.0, range_max=100.0),
            unit="meters",
            resolve_time="2030-01-01T00:00:00Z",
            community_prediction="50% chance",
            url="https://example.com/q/1",
        )
        assert q.unit == "meters"
        assert q.resolve_time == "2030-01-01T00:00:00Z"
        assert q.community_prediction == "50% chance"
        assert q.url == "https://example.com/q/1"
        assert q.scaling is not None

    def test_scaling_optional(self):
        q = ForecastQuestion(
            title="Binary Q",
            body="Body",
            question_type=QuestionType.BINARY,
        )
        assert q.scaling is None


class TestBuildQuestionContext:
    @mock.patch("common.models.datetime")
    def test_future_resolve_time_with_timezone(self, mock_dt, sample_binary_question):
        mock_dt.datetime.now.return_value = datetime.datetime(
            2025, 1, 1, 0, 0, 0, tzinfo=datetime.UTC
        )
        mock_dt.UTC = datetime.UTC
        fromiso_original = datetime.datetime.fromisoformat

        def fake_fromisoformat(s):
            if s == "2100-01-02T00:01:00Z":
                return datetime.datetime(2100, 1, 2, 0, 1, 0, tzinfo=datetime.UTC)
            return fromiso_original(s)

        mock_dt.datetime.fromisoformat = fake_fromisoformat

        ctx = build_question_context(sample_binary_question)
        assert ctx["title"] == "Will humans go extinct before 2100?"
        assert isinstance(ctx["hours_remaining"], int)
        assert ctx["hours_remaining"] >= 0

    def test_empty_summary_report_default(self, sample_binary_question):
        ctx = build_question_context(sample_binary_question)
        assert ctx["summary_report"] == ""

    def test_empty_hours_remaining_default(self, sample_binary_question):
        q = sample_binary_question.model_copy(update={"resolve_time": None})
        ctx = build_question_context(q)
        assert ctx["hours_remaining"] == ""

    def test_invalid_resolve_time_ignored(self, sample_binary_question):
        q = sample_binary_question.model_copy(update={"resolve_time": "not-a-date"})
        ctx = build_question_context(q)
        assert ctx["hours_remaining"] == ""

    def test_past_resolve_time_sets_empty(self, sample_binary_question):
        q = sample_binary_question.model_copy(update={"resolve_time": "2020-01-01T00:00:00Z"})
        ctx = build_question_context(q)
        assert ctx["hours_remaining"] == ""

    def test_resolve_time_missing_timezone(self, sample_binary_question):
        q = sample_binary_question.model_copy(update={"resolve_time": "2100-01-02T00:01:00"})
        ctx = build_question_context(q)
        assert ctx["hours_remaining"] != ""

    def test_current_time_falls_back(self, sample_binary_question):
        ctx = build_question_context(sample_binary_question)
        assert "hours_remaining" in ctx


class TestResolutionRecord:
    def test_binary_resolution(self):
        rec = ResolutionRecord(
            question_id=123,
            resolved_at=None,
            resolution="yes",
            question_type="binary",
        )
        assert rec.question_type == "binary"
        assert rec.resolution == "yes"
        assert rec.scaling is None
        assert rec.options is None

    def test_numeric_resolution_with_scaling(self):
        rec = ResolutionRecord(
            question_id=456,
            resolved_at=None,
            resolution=42.5,
            question_type="numeric",
            scaling={"range_min": 0.0, "range_max": 100.0, "cdf_size": 201},
        )
        assert rec.scaling is not None
        assert rec.scaling["range_min"] == 0.0

    def test_mcq_resolution_with_options(self):
        rec = ResolutionRecord(
            question_id=789,
            resolved_at=None,
            resolution="B",
            question_type="multiple_choice",
            options=["A", "B", "C"],
        )
        assert rec.options == ["A", "B", "C"]

    def test_round_trip(self):
        rec = ResolutionRecord(
            question_id=1,
            resolved_at=None,
            resolution=3.14,
            question_type="numeric",
            scaling={"range_min": 0.0, "range_max": 10.0},
            options=None,
        )
        loaded = ResolutionRecord.model_validate_json(rec.model_dump_json())
        assert loaded == rec
