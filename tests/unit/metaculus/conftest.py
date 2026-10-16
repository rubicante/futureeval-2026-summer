"""Fixtures for metaculus adapter unit tests — loaded from fixtures.json."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_FIXTURES = json.loads((Path(__file__).parent / "fixtures.json").read_text())


@pytest.fixture
def binary_api_post() -> dict:
    return _FIXTURES["binary_api_post"]


@pytest.fixture
def numeric_api_post() -> dict:
    post = dict(_FIXTURES["numeric_api_post"])
    post["question"]["aggregations"]["recency_weighted"]["latest"]["continuous_cdf"] = [
        min(i / 200, 1.0) for i in range(201)
    ]
    return post


@pytest.fixture
def log_numeric_api_post() -> dict:
    return _FIXTURES["log_numeric_api_post"]


@pytest.fixture
def mc_api_post() -> dict:
    return _FIXTURES["mc_api_post"]


@pytest.fixture
def minimal_api_post() -> dict:
    return _FIXTURES["minimal_api_post"]
