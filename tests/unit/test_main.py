from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from main import _parse_config_entries, parse_args


class TestParseConfigEntries:
    def test_empty(self):
        assert _parse_config_entries([]) == {}

    def test_int(self):
        assert _parse_config_entries(["runs=5"]) == {"runs": 5}

    def test_negative_int(self):
        assert _parse_config_entries(["value=-3"]) == {"value": -3}

    def test_float(self):
        assert _parse_config_entries(["threshold=0.15"]) == {"threshold": 0.15}

    def test_bool_true(self):
        assert _parse_config_entries(["flag=true"]) == {"flag": True}

    def test_bool_false(self):
        assert _parse_config_entries(["flag=false"]) == {"flag": False}

    def test_string(self):
        assert _parse_config_entries(["searcher=exa"]) == {"searcher": "exa"}

    def test_repeatable(self):
        assert _parse_config_entries(["runs=5", "ov_runs=3"]) == {"runs": 5, "ov_runs": 3}

    def test_last_wins_on_duplicate(self):
        assert _parse_config_entries(["runs=5", "runs=10"]) == {"runs": 10}

    def test_missing_equals_raises(self):
        import argparse

        with pytest.raises(argparse.ArgumentTypeError):
            _parse_config_entries(["missingequals"])


class TestParseArgsDefaults:
    def test_submit_defaults_to_false(self):
        sys.argv = ["main.py", "-t", "x"]
        config = parse_args()
        assert config.submit is False

    def test_tournament_id_is_required(self):
        sys.argv = ["main.py"]
        with pytest.raises(SystemExit):
            parse_args()

    def test_forecast_config_defaults_to_empty(self):
        sys.argv = ["main.py", "-t", "x"]
        config = parse_args()
        assert config.forecast_config_entries == []

    def test_research_config_defaults_to_empty(self):
        sys.argv = ["main.py", "-t", "x"]
        config = parse_args()
        assert config.research_config_entries == []

    def test_reforecast_defaults_to_false(self):
        sys.argv = ["main.py", "-t", "x"]
        config = parse_args()
        assert config.reforecast is False

    def test_disable_research_defaults_to_false(self):
        sys.argv = ["main.py", "-t", "x"]
        config = parse_args()
        assert config.disable_research is False

    def test_strategy_default(self):
        sys.argv = ["main.py", "-t", "x"]
        config = parse_args()
        assert config.strategy == "ensemble"

    def test_returns_simple_namespace(self):
        sys.argv = ["main.py", "-t", "x"]
        config = parse_args()
        assert isinstance(config, SimpleNamespace)


class TestParseArgsCustom:
    def test_submit_flag(self):
        sys.argv = ["main.py", "-s", "-t", "x"]
        config = parse_args()
        assert config.submit is True

    def test_tournament_id(self):
        sys.argv = ["main.py", "-t", "12345"]
        config = parse_args()
        assert config.tournament_id == "12345"

    def test_forecast_config_single(self):
        sys.argv = ["main.py", "-t", "x", "--fc", "runs=3"]
        config = parse_args()
        assert config.forecast_config_entries == ["runs=3"]

    def test_forecast_config_repeatable(self):
        sys.argv = ["main.py", "-t", "x", "--fc", "runs=5", "--fc", "ov_runs=3"]
        config = parse_args()
        assert config.forecast_config_entries == ["runs=5", "ov_runs=3"]

    def test_research_config_single(self):
        sys.argv = ["main.py", "-t", "x", "--rc", "searcher=exa"]
        config = parse_args()
        assert config.research_config_entries == ["searcher=exa"]

    def test_reforecast_flag(self):
        sys.argv = ["main.py", "-t", "x", "--reforecast"]
        config = parse_args()
        assert config.reforecast is True

    def test_disable_research_flag(self):
        sys.argv = ["main.py", "-t", "x", "--disable-research"]
        config = parse_args()
        assert config.disable_research is True

    def test_combined_flags(self):
        sys.argv = [
            "main.py",
            "-t",
            "x",
            "-s",
            "--reforecast",
            "--fc",
            "runs=10",
            "--rc",
            "searcher=exa",
        ]
        config = parse_args()
        assert config.submit is True
        assert config.reforecast is True
        assert config.forecast_config_entries == ["runs=10"]
        assert config.research_config_entries == ["searcher=exa"]

    def test_strategy_custom(self):
        sys.argv = ["main.py", "-t", "x", "--strategy", "ensemble"]
        config = parse_args()
        assert config.strategy == "ensemble"

    def test_invalid_strategy_raises(self):
        sys.argv = ["main.py", "-t", "x", "--strategy", "invalid"]
        with pytest.raises(SystemExit):
            parse_args()
