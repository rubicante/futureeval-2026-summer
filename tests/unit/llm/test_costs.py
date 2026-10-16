"""Unit tests for llm.costs — CostLedger accounting."""

from __future__ import annotations

import pytest

from llm.costs import CostLedger, _warned_missing_prices, current_question


class TestCostLedger:
    def test_known_model_pricing(self):
        ledger = CostLedger()
        # gpt-5.5: $5/M input, $0.5/M cached, $30/M output
        ledger.record(
            purpose="forecast",
            model="gpt-5.5",
            prompt_tokens=1_000_000,
            cached_tokens=500_000,
            completion_tokens=100_000,
        )
        expected = (500_000 * 5.0 + 500_000 * 0.5 + 100_000 * 30.0) / 1e6
        assert ledger.total_cost_usd() == pytest.approx(expected)

    def test_unknown_model_costs_zero(self, caplog):
        _warned_missing_prices.clear()
        ledger = CostLedger()
        ledger.record(
            purpose="forecast",
            model="mystery-model",
            prompt_tokens=1000,
            cached_tokens=0,
            completion_tokens=1000,
        )
        assert ledger.total_cost_usd() == 0.0
        assert "mystery-model" in caplog.text
        assert "PRICES_PER_M" in caplog.text

    def test_unknown_model_warns_once_per_model(self, caplog):
        _warned_missing_prices.clear()
        ledger = CostLedger()
        for _ in range(3):
            ledger.record(
                purpose="forecast",
                model="mystery-model",
                prompt_tokens=100,
                cached_tokens=0,
                completion_tokens=10,
            )
        assert caplog.text.count("mystery-model") == 1

    def test_per_question_attribution(self):
        ledger = CostLedger()
        token = current_question.set(42)
        try:
            ledger.record(
                purpose="forecast",
                model="gpt-5.5",
                prompt_tokens=100,
                cached_tokens=0,
                completion_tokens=10,
            )
        finally:
            current_question.reset(token)
        ledger.record(
            purpose="extract",
            model="gpt-5.4-mini",
            prompt_tokens=100,
            cached_tokens=0,
            completion_tokens=10,
        )
        q = ledger.question_cost(42)
        assert q is not None
        assert q["calls"] == 1
        assert q["prompt_tokens"] == 100
        assert ledger.question_cost(99) is None

    def test_summary_lists_purpose_and_total(self):
        ledger = CostLedger()
        ledger.record(
            purpose="forecast",
            model="gpt-5.5",
            prompt_tokens=10,
            cached_tokens=0,
            completion_tokens=10,
        )
        summary = ledger.format_summary()
        assert "forecast" in summary
        assert "total $" in summary

    def test_empty_summary(self):
        assert "no LLM usage" in CostLedger().format_summary()
