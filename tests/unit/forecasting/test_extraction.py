"""Unit tests for forecasting.extraction — structured output schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from forecasting.extraction import (
    BinaryExtraction,
    DateExtraction,
    NumericExtraction,
    OptionPrediction,
)


class TestBinaryExtraction:
    def test_valid_probability(self):
        b = BinaryExtraction(probability_percent=42)
        assert b.probability_percent == 42

    def test_probability_zero(self):
        b = BinaryExtraction(probability_percent=0)
        assert b.probability_percent == 0

    def test_probability_hundred(self):
        b = BinaryExtraction(probability_percent=100)
        assert b.probability_percent == 100

    def test_probability_negative_raises(self):
        with pytest.raises(ValidationError):
            BinaryExtraction(probability_percent=-1)

    def test_probability_above_100_raises(self):
        with pytest.raises(ValidationError):
            BinaryExtraction(probability_percent=101)

    def test_probability_float_allowed(self):
        b = BinaryExtraction(probability_percent=42.5)
        assert b.probability_percent == 42.5


class TestNumericExtraction:
    def test_valid_percentiles(self):
        n = NumericExtraction(
            percentile_10=10.0,
            percentile_20=20.0,
            percentile_40=40.0,
            percentile_60=60.0,
            percentile_80=80.0,
            percentile_90=90.0,
        )
        assert n.percentile_10 == 10.0
        assert n.percentile_20 == 20.0
        assert n.percentile_90 == 90.0

    def test_negative_values_allowed(self):
        n = NumericExtraction(
            percentile_10=-100.0,
            percentile_20=-80.0,
            percentile_40=-60.0,
            percentile_60=-40.0,
            percentile_80=-20.0,
            percentile_90=-10.0,
        )
        assert n.percentile_10 == -100.0

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            NumericExtraction(percentile_10=10.0)


class TestDateExtraction:
    def test_valid_ymd_dates(self):
        d = DateExtraction(
            percentile_10="2025-01-01",
            percentile_20="2025-03-01",
            percentile_40="2025-06-01",
            percentile_60="2025-09-01",
            percentile_80="2026-01-01",
            percentile_90="2026-06-01",
        )
        assert d.percentile_10 == "2025-01-01"

    def test_valid_iso_with_time(self):
        d = DateExtraction(
            percentile_10="2025-01-01T00:00:00",
            percentile_20="2025-03-01T00:00:00",
            percentile_40="2025-06-01T00:00:00",
            percentile_60="2025-09-01T00:00:00",
            percentile_80="2026-01-01T00:00:00",
            percentile_90="2026-06-01T00:00:00",
        )
        assert d.percentile_10 == "2025-01-01T00:00:00"

    def test_valid_iso_with_z(self):
        d = DateExtraction(
            percentile_10="2025-01-01T00:00:00Z",
            percentile_20="2025-03-01T00:00:00Z",
            percentile_40="2025-06-01T00:00:00Z",
            percentile_60="2025-09-01T00:00:00Z",
            percentile_80="2026-01-01T00:00:00Z",
            percentile_90="2026-06-01T00:00:00Z",
        )
        assert d.percentile_10 == "2025-01-01T00:00:00Z"

    def test_invalid_date_raises(self):
        with pytest.raises(ValidationError):
            DateExtraction(
                percentile_10="not-a-date",
                percentile_20="2025-03-01",
                percentile_40="2025-06-01",
                percentile_60="2025-09-01",
                percentile_80="2026-01-01",
                percentile_90="2026-06-01",
            )

    def test_invalid_month_raises(self):
        with pytest.raises(ValidationError):
            DateExtraction(
                percentile_10="2025-13-01",
                percentile_20="2025-03-01",
                percentile_40="2025-06-01",
                percentile_60="2025-09-01",
                percentile_80="2026-01-01",
                percentile_90="2026-06-01",
            )


class TestOptionPrediction:
    def test_valid_prediction(self):
        op = OptionPrediction(name="Democrat", probability=0.45)
        assert op.name == "Democrat"
        assert op.probability == 0.45

    def test_probability_zero(self):
        op = OptionPrediction(name="A", probability=0.0)
        assert op.probability == 0.0

    def test_probability_one(self):
        op = OptionPrediction(name="A", probability=1.0)
        assert op.probability == 1.0

    def test_missing_name_raises(self):
        with pytest.raises(ValidationError):
            OptionPrediction(name=None, probability=0.5)  # type: ignore[arg-type]
