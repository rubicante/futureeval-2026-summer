"""Fixtures for forecasting unit tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def valid_percentiles_linear() -> dict:
    return {
        10: 20.0,
        20: 35.0,
        40: 48.0,
        60: 55.0,
        80: 70.0,
        90: 85.0,
    }


@pytest.fixture
def valid_percentiles_wide() -> dict:
    return {
        10: -500.0,
        20: -200.0,
        40: 0.0,
        60: 200.0,
        80: 500.0,
        90: 800.0,
    }


@pytest.fixture
def valid_percentiles_tight() -> dict:
    return {
        10: 48.0,
        20: 49.0,
        40: 49.5,
        60: 50.0,
        80: 50.5,
        90: 51.0,
    }


@pytest.fixture
def valid_percentiles_log() -> dict:
    return {
        10: 100.0,
        20: 500.0,
        40: 2000.0,
        60: 5000.0,
        80: 20000.0,
        90: 50000.0,
    }
