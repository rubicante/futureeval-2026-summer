"""Unit tests for common.capture — CaptureRecord model."""

from __future__ import annotations

from datetime import UTC, datetime

from common.models import CaptureRecord, RunConfig


class TestCaptureRecord:
    def _run_config(self) -> RunConfig:
        return RunConfig(
            schema_version=1,
            research_strategy="planned_research",
            research_provider="exa",
            forecast_strategy="ensemble",
            forecast_model="gpt-5.4-mini",
            decompose_model="gpt-5.4-mini",
        )

    def test_minimal_record(self):
        now = datetime.now(UTC)
        record = CaptureRecord(
            run_id="2026-06-04_120000",
            question_id=12345,
            snapshot_timestamp=now,
            run_config=self._run_config(),
            context={"title": "Test Q", "body": "body", "summary_report": "research"},
            raw_results=[(0.73, "rationale text")],
            final_forecast=0.73,
        )
        assert record.format_version == 1
        assert record.code_version is None
        assert record.run_id == "2026-06-04_120000"
        assert record.question_id == 12345
        assert record.snapshot_timestamp == now
        assert record.context["title"] == "Test Q"
        assert len(record.raw_results) == 1
        assert record.raw_results[0] == (0.73, "rationale text")
        assert record.final_forecast == 0.73
        assert record.community_prediction is None

    def test_with_community_prediction(self):
        cp = {"recency_weighted": {"latest": {"probability_yes": 0.19}}}
        record = CaptureRecord(
            run_id="2026-06-04_120000",
            question_id=12345,
            snapshot_timestamp=datetime.now(UTC),
            run_config=self._run_config(),
            context={},
            raw_results=[(0.5, "rationale")],
            final_forecast=0.5,
            community_prediction=cp,
        )
        assert record.community_prediction == cp

    def test_with_code_version(self):
        record = CaptureRecord(
            code_version="abc123def",
            run_id="2026-06-04_120000",
            question_id=12345,
            snapshot_timestamp=datetime.now(UTC),
            run_config=self._run_config(),
            context={},
            raw_results=[(0.5, "rationale")],
            final_forecast=0.5,
        )
        assert record.code_version == "abc123def"

    def test_round_trip(self):
        now = datetime.now(UTC)
        record = CaptureRecord(
            code_version="abc123",
            run_id="2026-06-04_120000",
            question_id=12345,
            snapshot_timestamp=now,
            run_config=self._run_config(),
            context={"title": "Q", "body": "", "summary_report": ""},
            raw_results=[(0.42, "rationale")],
            final_forecast=0.42,
            community_prediction={"recency_weighted": {"latest": {"probability_yes": 0.19}}},
        )
        json_str = record.model_dump_json()
        loaded = CaptureRecord.model_validate_json(json_str)
        assert loaded == record

    def test_format_version_default(self):
        record = CaptureRecord(
            run_id="2026-06-04_120000",
            question_id=12345,
            snapshot_timestamp=datetime.now(UTC),
            run_config=self._run_config(),
            context={},
            raw_results=[(0.5, "rationale")],
            final_forecast=0.5,
        )
        assert record.format_version == 1
