"""Integration tests for main.py — orchestration with mocked services."""

from __future__ import annotations

import logging
from dataclasses import replace
from unittest import mock

import pytest

from metaculus.client import MetaculusClient


@pytest.fixture
def orchestrator_config(llm_config):
    return llm_config


@pytest.fixture
def mock_client():
    client = mock.MagicMock(spec=MetaculusClient)
    client.close = mock.MagicMock()
    return client


# ── forecast_individual_question ────────────────────────────────────


class TestForecastIndividualQuestion:
    async def test_binary_question(
        self, mock_client, api_post_details_binary, orchestrator_config, patch_forecast_deps
    ):
        from main import forecast_individual_question

        mock_client.get_post_details.return_value = api_post_details_binary
        mock_client.forecast_already_made.return_value = False

        with patch_forecast_deps() as gateway:
            orchestrator_config = replace(orchestrator_config, gateway=gateway)
            result = await forecast_individual_question(
                mock_client, 43332, 43327, orchestrator_config
            )

        assert "Forecast:" in result.summary
        assert not mock_client.post_question_prediction.called

    async def test_skips_if_already_forecast(
        self, mock_client, api_post_details_binary, orchestrator_config
    ):
        from main import forecast_individual_question

        mock_client.get_post_details.return_value = api_post_details_binary
        mock_client.forecast_already_made.return_value = True

        result = await forecast_individual_question(mock_client, 43332, 43327, orchestrator_config)

        assert "Skipped" in result.summary

    async def test_submits_when_enabled(
        self, mock_client, api_post_details_binary, orchestrator_config, patch_forecast_deps
    ):
        from main import forecast_individual_question

        orchestrator_config = replace(orchestrator_config, submit=True)
        mock_client.get_post_details.return_value = api_post_details_binary
        mock_client.forecast_already_made.return_value = False

        with patch_forecast_deps() as gateway:
            orchestrator_config = replace(orchestrator_config, gateway=gateway)
            await forecast_individual_question(mock_client, 43332, 43327, orchestrator_config)

        assert mock_client.post_question_prediction.called

    async def test_reforecast_overrides_skip(
        self, mock_client, api_post_details_binary, orchestrator_config, patch_forecast_deps
    ):
        from main import forecast_individual_question

        orchestrator_config = replace(orchestrator_config, reforecast=True)
        mock_client.get_post_details.return_value = api_post_details_binary
        mock_client.forecast_already_made.return_value = True

        with patch_forecast_deps() as gateway:
            orchestrator_config = replace(orchestrator_config, gateway=gateway)
            result = await forecast_individual_question(
                mock_client, 43332, 43327, orchestrator_config
            )

        assert "Skipped" not in result.summary

    async def test_date_question(
        self, mock_client, api_post_details_date, orchestrator_config, patch_forecast_deps
    ):
        from main import forecast_individual_question

        mock_client.get_post_details.return_value = api_post_details_date
        mock_client.forecast_already_made.return_value = False

        async def fake_call_llm(request):
            return (
                "Percentile 10: 2026-03-15\n"
                "Percentile 20: 2026-09-01\n"
                "Percentile 40: 2027-03-01\n"
                "Percentile 60: 2027-09-01\n"
                "Percentile 80: 2028-03-01\n"
                "Percentile 90: 2028-09-01"
            )

        async def fake_extract(request, output_type):
            from forecasting.extraction import DateExtraction

            return DateExtraction(
                percentile_10="2026-03-15",
                percentile_20="2026-09-01",
                percentile_40="2027-03-01",
                percentile_60="2027-09-01",
                percentile_80="2028-03-01",
                percentile_90="2028-09-01",
            )

        with patch_forecast_deps(call_llm=fake_call_llm, extract=fake_extract) as gateway:
            orchestrator_config = replace(orchestrator_config, gateway=gateway)
            result = await forecast_individual_question(
                mock_client, 43324, 43324, orchestrator_config
            )

        assert "Forecast:" in result.summary
        assert not mock_client.post_question_prediction.called

    async def test_discrete_question(
        self, mock_client, api_post_details_discrete, orchestrator_config, patch_forecast_deps
    ):
        from main import forecast_individual_question

        mock_client.get_post_details.return_value = api_post_details_discrete
        mock_client.forecast_already_made.return_value = False

        async def fake_call_llm(request):
            return (
                "Percentile 10: 2\n"
                "Percentile 20: 4\n"
                "Percentile 40: 6\n"
                "Percentile 60: 8\n"
                "Percentile 80: 10\n"
                "Percentile 90: 12"
            )

        async def fake_extract(request, output_type):
            from forecasting.extraction import NumericExtraction

            return NumericExtraction(
                percentile_10=2,
                percentile_20=4,
                percentile_40=6,
                percentile_60=8,
                percentile_80=10,
                percentile_90=12,
            )

        with patch_forecast_deps(call_llm=fake_call_llm, extract=fake_extract) as gateway:
            orchestrator_config = replace(orchestrator_config, gateway=gateway)
            result = await forecast_individual_question(
                mock_client, 43321, 43321, orchestrator_config
            )

        assert "Forecast:" in result.summary
        assert not mock_client.post_question_prediction.called


# ── forecast_questions ──────────────────────────────────────────────


class TestForecastQuestions:
    async def test_all_succeed(
        self, mock_client, api_post_details_binary, orchestrator_config, patch_forecast_deps
    ):
        from main import forecast_questions

        mock_client.get_post_details.return_value = api_post_details_binary
        mock_client.forecast_already_made.return_value = False

        with patch_forecast_deps():
            await forecast_questions(
                mock_client,
                [(43332, 43327), (43331, 43326)],
                orchestrator_config,
            )

    async def test_error_on_one_raises_runtime_error(
        self, mock_client, api_post_details_binary, orchestrator_config, patch_forecast_deps
    ):
        from main import forecast_questions

        call_count = 0

        def get_details(post_id):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return api_post_details_binary
            raise RuntimeError("API error")

        mock_client.get_post_details.side_effect = get_details
        mock_client.forecast_already_made.return_value = False

        with patch_forecast_deps():
            with pytest.raises(RuntimeError, match="Errors were encountered"):
                await forecast_questions(
                    mock_client,
                    [(43332, 43327), (43331, 43326)],
                    orchestrator_config,
                )

    async def test_error_logs_contain_url_format(
        self, mock_client, api_post_details_binary, orchestrator_config, patch_forecast_deps
    ):
        from main import forecast_questions

        call_count = 0

        def get_details(post_id):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return api_post_details_binary
            raise RuntimeError("API error")

        mock_client.get_post_details.side_effect = get_details
        mock_client.forecast_already_made.return_value = False

        with patch_forecast_deps():
            with pytest.raises(RuntimeError):
                with self._capture_log("main", logging.ERROR) as log_output:
                    await forecast_questions(
                        mock_client,
                        [(43332, 43327), (43331, 43326)],
                        orchestrator_config,
                    )

        error_lines = log_output.getvalue()
        assert "Error:" in error_lines
        assert "RuntimeError" in error_lines
        assert "API error" in error_lines
        assert "https://www.metaculus.com/questions/43326/" in error_lines

    @staticmethod
    def _capture_log(logger_name: str, level: int):
        import io

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(level)
        logger = logging.getLogger(logger_name)
        logger.addHandler(handler)

        class CaptureCtx:
            def __enter__(self):
                return stream

            def __exit__(self, *args):
                logger.removeHandler(handler)
                handler.close()

        return CaptureCtx()


# ── Submission flow (with real API shapes) ──────────────────────────


class TestSubmissionFlow:
    async def test_submits_binary_forecast_with_correct_payload(
        self, mock_client, api_post_details_binary, orchestrator_config, patch_forecast_deps
    ):
        from main import forecast_individual_question

        orchestrator_config = replace(orchestrator_config, submit=True)
        mock_client.get_post_details.return_value = api_post_details_binary
        mock_client.forecast_already_made.return_value = False

        async def fake_call_llm(request):
            return "Rationale\nProbability: 42%"

        async def fake_extract(request, output_type):
            from forecasting.extraction import BinaryExtraction

            return BinaryExtraction(probability_percent=42)

        with patch_forecast_deps(call_llm=fake_call_llm, extract=fake_extract) as gateway:
            orchestrator_config = replace(orchestrator_config, gateway=gateway)
            await forecast_individual_question(mock_client, 43332, 43327, orchestrator_config)

        assert mock_client.post_question_prediction.called
        call_args = mock_client.post_question_prediction.call_args
        question_id, payload = call_args[0]
        assert question_id == 43332
        assert payload["probability_yes"] == pytest.approx(0.42)
        assert payload["probability_yes_per_category"] is None
        assert payload["continuous_cdf"] is None

    async def test_submits_numeric_cdf_forecast_with_correct_payload(
        self, mock_client, api_post_details_numeric, orchestrator_config, patch_forecast_deps
    ):
        from main import forecast_individual_question

        orchestrator_config = replace(orchestrator_config, submit=True)
        mock_client.get_post_details.return_value = api_post_details_numeric
        mock_client.forecast_already_made.return_value = False

        percentile_text = (
            "Percentile 10: -30\nPercentile 20: -25\nPercentile 40: -20\n"
            "Percentile 60: -15\nPercentile 80: -10\nPercentile 90: -5"
        )

        async def fake_call_llm(request):
            return f"Analysis.\n\n{percentile_text}"

        async def fake_extract(request, output_type):
            from forecasting.extraction import NumericExtraction

            return NumericExtraction(
                percentile_10=-30,
                percentile_20=-25,
                percentile_40=-20,
                percentile_60=-15,
                percentile_80=-10,
                percentile_90=-5,
            )

        with patch_forecast_deps(call_llm=fake_call_llm, extract=fake_extract) as gateway:
            orchestrator_config = replace(orchestrator_config, gateway=gateway)
            await forecast_individual_question(mock_client, 43325, 43323, orchestrator_config)

        assert mock_client.post_question_prediction.called
        _, payload = mock_client.post_question_prediction.call_args[0]
        assert payload["probability_yes"] is None
        assert payload["probability_yes_per_category"] is None
        assert isinstance(payload["continuous_cdf"], list)
        assert len(payload["continuous_cdf"]) == 201

    async def test_submits_mc_forecast_with_correct_payload(
        self, mock_client, api_post_details_mc, orchestrator_config, patch_forecast_deps
    ):
        from main import forecast_individual_question

        orchestrator_config = replace(orchestrator_config, submit=True)
        mock_client.get_post_details.return_value = api_post_details_mc
        mock_client.forecast_already_made.return_value = False

        options = api_post_details_mc["question"]["options"]
        probs = {opt: 1.0 / len(options) for opt in options}

        async def fake_call_llm(request):
            return "\n".join(f"{opt}: {probs[opt]}" for opt in options)

        async def fake_extract(request, output_type):
            from forecasting.extraction import OptionPrediction

            return [OptionPrediction(name=opt, probability=probs[opt]) for opt in options]

        with patch_forecast_deps(call_llm=fake_call_llm, extract=fake_extract) as gateway:
            orchestrator_config = replace(orchestrator_config, gateway=gateway)
            await forecast_individual_question(mock_client, 43331, 43326, orchestrator_config)

        assert mock_client.post_question_prediction.called
        _, payload = mock_client.post_question_prediction.call_args[0]
        assert payload["probability_yes"] is None
        assert isinstance(payload["probability_yes_per_category"], dict)
        assert payload["continuous_cdf"] is None

    def test_forecast_already_made_after_submission(self, api_post_details_binary_with_forecast):
        assert MetaculusClient.forecast_already_made(api_post_details_binary_with_forecast) is True

        forecast_values = api_post_details_binary_with_forecast["question"]["my_forecasts"][
            "latest"
        ]["forecast_values"]
        assert forecast_values is not None
        assert len(forecast_values) == 2

    async def test_full_roundtrip_post_details_submitted_shape(
        self, mock_client, api_post_details_binary, orchestrator_config, patch_forecast_deps
    ):
        from main import forecast_individual_question

        orchestrator_config = replace(orchestrator_config, submit=True)
        orchestrator_config = replace(orchestrator_config, disable_research=True)

        mock_client.get_post_details.return_value = api_post_details_binary
        mock_client.forecast_already_made.return_value = False

        async def fake_call_llm(request):
            return "Rationale\nProbability: 55%"

        async def fake_extract(request, output_type):
            from forecasting.extraction import BinaryExtraction

            return BinaryExtraction(probability_percent=55)

        with patch_forecast_deps(call_llm=fake_call_llm, extract=fake_extract) as gateway:
            orchestrator_config = replace(orchestrator_config, gateway=gateway)
            result = await forecast_individual_question(
                mock_client, 43332, 43327, orchestrator_config
            )

        assert "Forecast:" in result.summary
        assert "Posted:" in result.summary
        assert mock_client.post_question_prediction.called


class TestForecastIndividualQuestionEdgeCases:
    async def test_unsupported_question_type_raises(
        self, mock_client, api_post_details_binary, orchestrator_config
    ):
        import json

        from main import forecast_individual_question

        details_copy = json.loads(json.dumps(api_post_details_binary))
        details_copy["question"]["type"] = "unsupported_type"

        mock_client.get_post_details.return_value = details_copy
        mock_client.forecast_already_made.return_value = False

        with pytest.raises(ValueError, match="not a valid QuestionType"):
            await forecast_individual_question(mock_client, 1, 1, orchestrator_config)


# ── main ─────────────────────────────────────────────────────────────


class TestMain:
    @pytest.fixture
    def config(self):
        c = mock.MagicMock()
        c.submit = False
        c.strategy = "ensemble"
        c.tournament_id = ""
        c.reforecast = False
        c.disable_research = False
        c.limit = 0
        c.live = False
        c.forecast_config_entries = ["runs=5"]
        c.research_config_entries = []
        return c

    async def test_client_created_and_closed(self, config):
        from main import main

        fake_client = mock.AsyncMock()

        with (
            mock.patch("main.dotenv.load_dotenv"),
            mock.patch("main.Path.mkdir"),
            mock.patch("main.logging.basicConfig"),
            mock.patch("main.MetaculusClient", return_value=fake_client),
            mock.patch("main.forecast_questions", return_value=None),
            mock.patch(
                "main.build_default_gateway",
                return_value=(
                    mock.MagicMock(),
                    "gpt-5.4-mini",
                    "gpt-5.4-mini",
                    {"forecast": "gpt-5.4-mini"},
                ),
            ),
        ):
            await main(config)

        fake_client.close.assert_called_once()

    async def test_dotenv_is_loaded(self, config):
        from main import main

        with (
            mock.patch("main.dotenv.load_dotenv") as mock_load_dotenv,
            mock.patch("main.Path.mkdir"),
            mock.patch("main.logging.basicConfig"),
            mock.patch("main.MetaculusClient", return_value=mock.AsyncMock()),
            mock.patch("main.forecast_questions", return_value=None),
            mock.patch(
                "main.build_default_gateway",
                return_value=(
                    mock.MagicMock(),
                    "gpt-5.4-mini",
                    "gpt-5.4-mini",
                    {"forecast": "gpt-5.4-mini"},
                ),
            ),
        ):
            await main(config)

        mock_load_dotenv.assert_called_once()

    async def test_runs_directory_created(self, config):
        from main import main

        with (
            mock.patch("main.dotenv.load_dotenv"),
            mock.patch("main.Path.mkdir") as mock_mkdir,
            mock.patch("main.logging.basicConfig"),
            mock.patch("main.MetaculusClient", return_value=mock.AsyncMock()),
            mock.patch("main.forecast_questions", return_value=None),
            mock.patch(
                "main.build_default_gateway",
                return_value=(
                    mock.MagicMock(),
                    "gpt-5.4-mini",
                    "gpt-5.4-mini",
                    {"forecast": "gpt-5.4-mini"},
                ),
            ),
        ):
            await main(config)

        mock_mkdir.assert_called_once_with(exist_ok=True)

    async def test_gateway_config_is_set(self, config):
        from main import main

        with (
            mock.patch("main.dotenv.load_dotenv"),
            mock.patch("main.Path.mkdir"),
            mock.patch("main.logging.basicConfig"),
            mock.patch("main.MetaculusClient", return_value=mock.AsyncMock()),
            mock.patch("main.forecast_questions", return_value=None),
            mock.patch(
                "main.build_default_gateway",
                return_value=(
                    mock.MagicMock(),
                    "gpt-5.4-mini",
                    "gpt-5.4-mini",
                    {"forecast": "gpt-5.4-mini"},
                ),
            ),
        ):
            await main(config)

        assert hasattr(config, "gateway")
        assert config.gateway is not None

    async def test_client_closed_even_on_error(self, config):
        from main import main

        fake_client = mock.AsyncMock()

        with (
            mock.patch("main.dotenv.load_dotenv"),
            mock.patch("main.Path.mkdir"),
            mock.patch("main.logging.basicConfig"),
            mock.patch("main.MetaculusClient", return_value=fake_client),
            mock.patch("main.forecast_questions", side_effect=RuntimeError("boom")),
            mock.patch(
                "main.build_default_gateway",
                return_value=(
                    mock.MagicMock(),
                    "gpt-5.4-mini",
                    "gpt-5.4-mini",
                    {"forecast": "gpt-5.4-mini"},
                ),
            ),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                await main(config)

        fake_client.close.assert_called_once()

    async def test_tournament_questions_fetched_when_id_set(self, config):
        from main import main

        config.tournament_id = "123"

        fake_client = mock.AsyncMock()
        fake_client.get_open_question_ids_from_tournament.return_value = [
            (1, 10),
            (2, 20),
        ]

        with (
            mock.patch("main.dotenv.load_dotenv"),
            mock.patch("main.Path.mkdir"),
            mock.patch("main.logging.basicConfig"),
            mock.patch("main.MetaculusClient", return_value=fake_client),
            mock.patch("main.forecast_questions", return_value=None),
            mock.patch("main.asyncio.to_thread", wraps=lambda fn, *args, **kw: fn(*args, **kw)),
            mock.patch(
                "main.build_default_gateway",
                return_value=(
                    mock.MagicMock(),
                    "gpt-5.4-mini",
                    "gpt-5.4-mini",
                    {"forecast": "gpt-5.4-mini"},
                ),
            ),
        ):
            await main(config)

        fake_client.get_open_question_ids_from_tournament.assert_called_once_with("123")

    async def test_tournament_questions_not_fetched_when_id_empty(self, config):
        from main import main

        fake_client = mock.AsyncMock()

        with (
            mock.patch("main.dotenv.load_dotenv"),
            mock.patch("main.Path.mkdir"),
            mock.patch("main.logging.basicConfig"),
            mock.patch("main.MetaculusClient", return_value=fake_client),
            mock.patch("main.forecast_questions", return_value=None),
            mock.patch(
                "main.build_default_gateway",
                return_value=(
                    mock.MagicMock(),
                    "gpt-5.4-mini",
                    "gpt-5.4-mini",
                    {"forecast": "gpt-5.4-mini"},
                ),
            ),
        ):
            await main(config)

        fake_client.get_open_question_ids_from_tournament.assert_not_called()

    async def test_forecast_questions_called_with_open_ids(self, config):
        from main import main

        config.tournament_id = "123"

        fake_client = mock.AsyncMock()
        fake_client.get_open_question_ids_from_tournament.return_value = [
            (1, 10),
            (2, 20),
        ]

        with (
            mock.patch("main.dotenv.load_dotenv"),
            mock.patch("main.Path.mkdir"),
            mock.patch("main.logging.basicConfig"),
            mock.patch("main.MetaculusClient", return_value=fake_client),
            mock.patch("main.forecast_questions", return_value=None) as mock_fc,
            mock.patch("main.asyncio.to_thread", wraps=lambda fn, *args, **kw: fn(*args, **kw)),
            mock.patch(
                "main.build_default_gateway",
                return_value=(
                    mock.MagicMock(),
                    "gpt-5.4-mini",
                    "gpt-5.4-mini",
                    {"forecast": "gpt-5.4-mini"},
                ),
            ),
        ):
            await main(config)

        mock_fc.assert_called_once_with(fake_client, [(1, 10), (2, 20)], config)

    async def test_forecast_questions_called_with_empty_list_when_no_tournament(self, config):
        from main import main

        fake_client = mock.AsyncMock()

        with (
            mock.patch("main.dotenv.load_dotenv"),
            mock.patch("main.Path.mkdir"),
            mock.patch("main.logging.basicConfig"),
            mock.patch("main.MetaculusClient", return_value=fake_client),
            mock.patch("main.forecast_questions", return_value=None) as mock_fc,
            mock.patch(
                "main.build_default_gateway",
                return_value=(
                    mock.MagicMock(),
                    "gpt-5.4-mini",
                    "gpt-5.4-mini",
                    {"forecast": "gpt-5.4-mini"},
                ),
            ),
        ):
            await main(config)

        mock_fc.assert_called_once_with(fake_client, [], config)
