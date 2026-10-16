"""Unit tests for common.run_config — RunConfig model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from common.models import CURRENT_RUN_CONFIG_VERSION, RunConfig


class TestRunConfig:
    def test_default_values(self):
        cfg = RunConfig(
            schema_version=1,
            research_strategy="planned_research",
            research_provider="exa",
            forecast_strategy="ensemble",
            forecast_model="gpt-5.4-mini",
            decompose_model="gpt-5.4-mini",
        )
        assert cfg.schema_version == 1
        assert cfg.research_strategy == "planned_research"
        assert cfg.research_provider == "exa"
        assert cfg.forecast_strategy == "ensemble"
        assert cfg.forecast_config == {}
        assert cfg.research_config == {}
        assert cfg.forecast_model == "gpt-5.4-mini"
        assert cfg.decompose_model == "gpt-5.4-mini"

    def test_schema_version_required(self):
        with pytest.raises(ValidationError):
            RunConfig(
                research_strategy="planned_research",
                research_provider="exa",
                forecast_strategy="ensemble",
                forecast_model="gpt-5.4-mini",
                decompose_model="gpt-5.4-mini",
            )

    def test_migrate_noop(self):
        cfg = RunConfig(
            schema_version=CURRENT_RUN_CONFIG_VERSION,
            research_strategy="planned_research",
            research_provider="exa",
            forecast_strategy="ensemble",
            forecast_model="gpt-5.5",
            decompose_model="gpt-5.5",
        )
        migrated = cfg.migrate()
        assert migrated is cfg

    def test_migrate_idempotent(self):
        cfg = RunConfig(
            schema_version=CURRENT_RUN_CONFIG_VERSION,
            research_strategy="planned_research",
            research_provider="exa",
            forecast_strategy="ensemble",
            forecast_model="gpt-5.5",
            decompose_model="gpt-5.5",
        )
        assert cfg.migrate() == cfg.migrate()

    def test_with_config_dicts(self):
        cfg = RunConfig(
            schema_version=1,
            research_strategy="planned_research",
            research_provider="asknews",
            forecast_strategy="ensemble",
            forecast_config={"runs": 3, "ov_runs": 3},
            research_config={"searcher": "exa", "queries": 5},
            forecast_model="gpt-5.5",
            decompose_model="gpt-5.5",
        )
        assert cfg.forecast_config["runs"] == 3
        assert cfg.forecast_config["ov_runs"] == 3
        assert cfg.research_config["queries"] == 5

    def test_round_trip(self):
        cfg = RunConfig(
            schema_version=1,
            research_strategy="planned_research",
            research_provider="exa",
            forecast_strategy="ensemble",
            forecast_config={"runs": 5},
            research_config={},
            forecast_model="gpt-5.4-mini",
            decompose_model="gpt-5.4-mini",
        )
        json_str = cfg.model_dump_json()
        loaded = RunConfig.model_validate_json(json_str)
        assert loaded == cfg

    def test_unknown_key_in_config_dicts_preserved(self):
        cfg = RunConfig(
            schema_version=1,
            research_strategy="planned_research",
            research_provider="exa",
            forecast_strategy="ensemble",
            forecast_config={"runs": 5, "future_knob": "some_value"},
            research_config={},
            forecast_model="gpt-5.4-mini",
            decompose_model="gpt-5.4-mini",
        )
        loaded = RunConfig.model_validate_json(cfg.model_dump_json())
        assert loaded.forecast_config["future_knob"] == "some_value"
