"""Unit tests for metaculus.adapter — API response ↔ domain model conversion."""

from __future__ import annotations

from common.models import QuestionType
from metaculus.adapter import MetaculusAdapter


class TestFormatCommunityPrediction:
    def test_binary_with_prediction(self, binary_api_post):
        q = binary_api_post["question"]
        result = MetaculusAdapter._format_community_prediction(
            q["aggregations"], q["type"], q.get("options")
        )
        assert result is not None
        assert "19%" in result
        assert "Yes" in result

    def test_binary_no_prediction(self):
        result = MetaculusAdapter._format_community_prediction(
            {"recency_weighted": {"latest": {}}}, "binary"
        )
        assert result is None

    def test_numeric_with_cdf(self):
        aggregations = {
            "recency_weighted": {
                "latest": {"continuous_cdf": [min(i / 200, 1.0) for i in range(201)]}
            }
        }
        result = MetaculusAdapter._format_community_prediction(aggregations, "numeric")
        assert result is not None
        assert "median at" in result or "CDF available" in result

    def test_multiple_choice_with_categories(self):
        aggregations = {
            "recency_weighted": {
                "latest": {
                    "probability_yes_per_category": {
                        "A": 0.6,
                        "B": 0.4,
                    }
                }
            }
        }
        result = MetaculusAdapter._format_community_prediction(
            aggregations, "multiple_choice", ["A", "B"]
        )
        assert result is not None
        assert "A:" in result
        assert "B:" in result

    def test_no_aggregations(self):
        result = MetaculusAdapter._format_community_prediction({}, "binary")
        assert result is None

    def test_empty_recency_weighted(self):
        result = MetaculusAdapter._format_community_prediction({"recency_weighted": None}, "binary")
        assert result is None


class TestQuestionFromAPI:
    def test_binary_question(self, binary_api_post):
        q = MetaculusAdapter.question_from_api(binary_api_post)
        assert q.title == "Will humans go extinct before 2100?"
        assert q.question_type == QuestionType.BINARY
        assert q.scaling is None
        assert "Test description text." in q.body
        assert "Resolves Yes if no humans alive." in q.body
        assert "Fine print details." in q.body
        assert q.url == "https://www.metaculus.com/questions/578/"
        assert q.resolve_time == "2100-01-02T00:01:00Z"
        assert q.community_prediction is not None
        assert q.community_prediction_raw is not None
        assert q.community_prediction_raw["recency_weighted"]["latest"]["probability_yes"] == 0.19

    def test_numeric_question(self, numeric_api_post):
        q = MetaculusAdapter.question_from_api(numeric_api_post)
        assert q.question_type == QuestionType.NUMERIC
        assert q.scaling is not None
        assert q.scaling.range_min == 0.0
        assert q.scaling.range_max == 5.0
        assert q.scaling.zero_point is None
        assert q.scaling.cdf_size == 201
        assert q.unit == "°C"
        assert q.url == "https://www.metaculus.com/questions/1234/"

    def test_log_scaled_numeric_question(self, log_numeric_api_post):
        q = MetaculusAdapter.question_from_api(log_numeric_api_post)
        assert q.scaling is not None
        assert q.scaling.zero_point == 1.0
        assert q.scaling.range_min == 10.0
        assert q.scaling.range_max == 100000.0

    def test_multiple_choice_question(self, mc_api_post):
        q = MetaculusAdapter.question_from_api(mc_api_post)
        assert q.question_type == QuestionType.MULTIPLE_CHOICE
        assert q.options == ["Democrat", "Republican", "Other"]
        assert q.scaling is None
        assert q.url == "https://www.metaculus.com/questions/4321/"

    def test_minimal_question(self, minimal_api_post):
        q = MetaculusAdapter.question_from_api(minimal_api_post)
        assert q.title == "Minimal question"
        assert q.question_type == QuestionType.BINARY
        assert q.scaling is None
        assert q.body == ""
        assert q.unit is None
        assert q.resolve_time is None
        assert q.url == "https://www.metaculus.com/questions/1/"

    def test_open_bounds(self):
        post = {
            "id": 1,
            "title": "Open bounds test",
            "question": {
                "id": 1,
                "title": "Open bounds test",
                "type": "numeric",
                "description": "",
                "resolution_criteria": "",
                "fine_print": "",
                "options": [],
                "open_lower_bound": True,
                "open_upper_bound": True,
                "scaling": {
                    "range_min": 0.0,
                    "range_max": 100.0,
                    "zero_point": None,
                    "inbound_outcome_count": 200,
                },
                "unit": None,
                "scheduled_resolve_time": None,
                "aggregations": {},
            },
        }
        q = MetaculusAdapter.question_from_api(post)
        assert q.scaling is not None
        assert q.scaling.open_lower_bound is True
        assert q.scaling.open_upper_bound is True

    def test_discrete_question_cdf_size(self):
        post = {
            "id": 1,
            "title": "Discrete test",
            "question": {
                "id": 1,
                "title": "Discrete test",
                "type": "discrete",
                "description": "",
                "resolution_criteria": "",
                "fine_print": "",
                "options": [],
                "open_lower_bound": False,
                "open_upper_bound": False,
                "scaling": {
                    "range_min": 0.0,
                    "range_max": 100.0,
                    "zero_point": None,
                    "inbound_outcome_count": 50,
                },
                "unit": None,
                "scheduled_resolve_time": None,
                "aggregations": {},
            },
        }
        q = MetaculusAdapter.question_from_api(post)
        assert q.scaling is not None
        assert q.scaling.cdf_size == 51

    def test_numeric_question_no_scaling(self):
        post = {
            "id": 1,
            "title": "No scaling",
            "question": {
                "id": 1,
                "title": "No scaling",
                "type": "numeric",
                "description": "",
                "resolution_criteria": "",
                "fine_print": "",
                "options": [],
                "open_lower_bound": False,
                "open_upper_bound": False,
                "scaling": None,
                "unit": None,
                "scheduled_resolve_time": None,
                "aggregations": {},
            },
        }
        q = MetaculusAdapter.question_from_api(post)
        assert q.scaling is None

    def test_scaling_missing_range_min(self):
        post = {
            "id": 1,
            "title": "Partial scaling",
            "question": {
                "id": 1,
                "title": "Partial scaling",
                "type": "numeric",
                "description": "",
                "resolution_criteria": "",
                "fine_print": "",
                "options": [],
                "open_lower_bound": False,
                "open_upper_bound": False,
                "scaling": {
                    "range_min": None,
                    "range_max": 100.0,
                },
                "unit": None,
                "scheduled_resolve_time": None,
                "aggregations": {},
            },
        }
        q = MetaculusAdapter.question_from_api(post)
        assert q.scaling is None


class TestForecastToPayload:
    def test_binary(self):
        payload = MetaculusAdapter.forecast_to_payload(0.73, "binary")
        assert payload == {
            "probability_yes": 0.73,
            "probability_yes_per_category": None,
            "continuous_cdf": None,
        }

    def test_multiple_choice(self):
        probs = {"A": 0.6, "B": 0.4}
        payload = MetaculusAdapter.forecast_to_payload(probs, "multiple_choice")
        assert payload == {
            "probability_yes": None,
            "probability_yes_per_category": probs,
            "continuous_cdf": None,
        }

    def test_numeric(self):
        cdf = [i / 200 for i in range(201)]
        payload = MetaculusAdapter.forecast_to_payload(cdf, "numeric")
        assert payload == {
            "probability_yes": None,
            "probability_yes_per_category": None,
            "continuous_cdf": cdf,
        }

    def test_discrete(self):
        cdf = [i / 50 for i in range(51)]
        payload = MetaculusAdapter.forecast_to_payload(cdf, "discrete")
        assert payload["continuous_cdf"] == cdf

    def test_date(self):
        cdf = [i / 200 for i in range(201)]
        payload = MetaculusAdapter.forecast_to_payload(cdf, "date")
        assert payload["continuous_cdf"] == cdf

    def test_binary_extreme_probabilities(self):
        for p in [0.0, 1.0, 0.5]:
            payload = MetaculusAdapter.forecast_to_payload(p, "binary")
            assert payload["probability_yes"] == p

    def test_roundtrip_binary(self, binary_api_post):
        question = MetaculusAdapter.question_from_api(binary_api_post)
        forecast = 0.42
        payload = MetaculusAdapter.forecast_to_payload(forecast, question.question_type)
        assert payload["probability_yes"] == forecast
        assert payload["probability_yes_per_category"] is None
        assert payload["continuous_cdf"] is None
