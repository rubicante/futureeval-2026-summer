"""Fixtures for llm unit tests."""

from __future__ import annotations

import pytest

from llm.gateway import LLMRequest, ModelConfig


@pytest.fixture
def sample_request() -> LLMRequest:
    return LLMRequest(prompt="What is 2+2?", purpose="test")


@pytest.fixture
def sample_request_with_system() -> LLMRequest:
    return LLMRequest(prompt="What is 2+2?", purpose="test", system_prompt="You are a math tutor.")


@pytest.fixture
def sample_request_reasoning() -> LLMRequest:
    return LLMRequest(prompt="Solve a hard problem", purpose="test", reasoning_effort="high")


@pytest.fixture
def sample_extract_request() -> LLMRequest:
    return LLMRequest(prompt="Probability: 73%", purpose="extract")


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(model="test-model", provider="mock")
