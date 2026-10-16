"""Adapter round-trip tests using real API shapes from docs/."""

from __future__ import annotations

import pytest

from common.models import QuestionType
from metaculus.adapter import MetaculusAdapter


class TestQuestionFromApiAllTypes:
    def test_binary_question_from_real_shape(self, api_post_details_binary):
        q = MetaculusAdapter.question_from_api(api_post_details_binary)
        assert q.question_type == QuestionType.BINARY
        assert q.title == api_post_details_binary["title"]
        assert q.scaling is None
        assert q.options is None or q.options == []
        assert q.url is not None
        assert "metaculus.com/questions" in q.url
        assert q.body != ""
        assert "description" not in q.body

    def test_numeric_question_from_real_shape(self, api_post_details_numeric):
        q = MetaculusAdapter.question_from_api(api_post_details_numeric)
        assert q.question_type == QuestionType.NUMERIC
        assert q.scaling is not None
        assert q.scaling.range_min == -35
        assert q.scaling.range_max == 0
        assert q.scaling.open_lower_bound is True
        assert q.scaling.open_upper_bound is True
        assert q.scaling.cdf_size == 201
        assert q.unit == "%"
        assert q.scaling.nominal_min == -35
        assert q.scaling.nominal_max == 0
        assert q.scaling.continuous_range is not None
        assert len(q.scaling.continuous_range) == 201

    def test_discrete_question_from_real_shape(self, api_post_details_discrete):
        q = MetaculusAdapter.question_from_api(api_post_details_discrete)
        assert q.question_type == QuestionType.DISCRETE
        assert q.scaling is not None
        assert q.scaling.range_min == -0.5
        assert q.scaling.range_max == 15.5
        assert q.scaling.cdf_size == 17  # 16 options + 1
        assert q.scaling.nominal_min == 0
        assert q.scaling.nominal_max == 15
        assert q.scaling.continuous_range is not None
        assert len(q.scaling.continuous_range) == 17

    def test_date_question_from_real_shape(self, api_post_details_date):
        q = MetaculusAdapter.question_from_api(api_post_details_date)
        assert q.question_type == QuestionType.DATE
        assert q.scaling is not None
        assert q.scaling.range_min == 1653782400  # Unix timestamp
        assert q.scaling.range_max == 1811462400
        assert q.scaling.open_lower_bound is False
        assert q.scaling.open_upper_bound is True
        assert q.scaling.cdf_size == 201
        assert q.scaling.continuous_range is not None
        assert len(q.scaling.continuous_range) == 201
        assert isinstance(q.scaling.continuous_range[0], str)
        assert q.scaling.continuous_range[0].startswith("2022")

    def test_multiple_choice_question_from_real_shape(self, api_post_details_mc):
        q = MetaculusAdapter.question_from_api(api_post_details_mc)
        assert q.question_type == QuestionType.MULTIPLE_CHOICE
        assert q.scaling is None
        assert q.options is not None
        assert len(q.options) >= 3
        assert isinstance(q.options[0], str)
        assert q.url is not None

    def test_five_real_questions_all_produce_valid_question_type(
        self,
        api_post_details_binary,
        api_post_details_numeric,
        api_post_details_discrete,
        api_post_details_date,
        api_post_details_mc,
    ):
        for details in [
            api_post_details_binary,
            api_post_details_numeric,
            api_post_details_discrete,
            api_post_details_date,
            api_post_details_mc,
        ]:
            q = MetaculusAdapter.question_from_api(details)
            assert q.question_type in QuestionType
            assert q.title != ""
            assert q.url is not None


class TestRoundTripForecastToPayload:
    def test_binary_roundtrip(self, api_post_details_binary):
        question = MetaculusAdapter.question_from_api(api_post_details_binary)
        forecast = 0.42
        payload = MetaculusAdapter.forecast_to_payload(forecast, question.question_type)
        assert payload == {
            "probability_yes": 0.42,
            "probability_yes_per_category": None,
            "continuous_cdf": None,
        }

    def test_numeric_roundtrip(self, api_post_details_numeric):
        question = MetaculusAdapter.question_from_api(api_post_details_numeric)
        cdf = [i / 200 for i in range(201)]
        payload = MetaculusAdapter.forecast_to_payload(cdf, question.question_type)
        assert payload["probability_yes"] is None
        assert payload["probability_yes_per_category"] is None
        assert payload["continuous_cdf"] == cdf

    def test_discrete_roundtrip(self, api_post_details_discrete):
        question = MetaculusAdapter.question_from_api(api_post_details_discrete)
        cdf = [i / 16 for i in range(17)]
        payload = MetaculusAdapter.forecast_to_payload(cdf, question.question_type)
        assert payload["continuous_cdf"] == cdf

    def test_date_roundtrip(self, api_post_details_date):
        question = MetaculusAdapter.question_from_api(api_post_details_date)
        cdf = [i / 200 for i in range(201)]
        payload = MetaculusAdapter.forecast_to_payload(cdf, question.question_type)
        assert payload["continuous_cdf"] is not None
        assert len(payload["continuous_cdf"]) == 201

    def test_mc_roundtrip(self, api_post_details_mc):
        question = MetaculusAdapter.question_from_api(api_post_details_mc)
        options = question.options or []
        probs = {opt: 1.0 / len(options) for opt in options}
        payload = MetaculusAdapter.forecast_to_payload(probs, question.question_type)
        assert payload["probability_yes"] is None
        assert payload["probability_yes_per_category"] == probs
        assert payload["continuous_cdf"] is None
        assert sum(payload["probability_yes_per_category"].values()) == pytest.approx(1.0)


class TestCommunityPredictionFormatting:
    def test_numeric_community_prediction_from_real_shape(self, api_post_details_numeric):
        q = MetaculusAdapter.question_from_api(api_post_details_numeric)
        assert q.community_prediction is not None
        assert "CDF available" in q.community_prediction or "median at" in q.community_prediction

    def test_date_community_prediction_from_real_shape(self, api_post_details_date):
        q = MetaculusAdapter.question_from_api(api_post_details_date)
        assert q.community_prediction is not None

    def test_binary_no_community_prediction_from_test_data(self, api_post_details_binary):
        q = MetaculusAdapter.question_from_api(api_post_details_binary)
        assert q.community_prediction is None

    def test_mc_no_community_prediction_from_test_data(self, api_post_details_mc):
        q = MetaculusAdapter.question_from_api(api_post_details_mc)
        assert q.community_prediction is None


class TestBodyConcatenation:
    def test_binary_body_includes_description_and_criteria(self, api_post_details_binary):
        q = MetaculusAdapter.question_from_api(api_post_details_binary)
        desc = api_post_details_binary["question"].get("description", "")
        criteria = api_post_details_binary["question"].get("resolution_criteria", "")
        if desc:
            assert desc in q.body
        if criteria:
            assert criteria in q.body

    def test_numeric_body_includes_description(self, api_post_details_numeric):
        q = MetaculusAdapter.question_from_api(api_post_details_numeric)
        desc = api_post_details_numeric["question"].get("description", "")
        if desc:
            assert desc in q.body

    def test_fine_print_included(self, api_post_details_binary):
        fp = api_post_details_binary["question"].get("fine_print", "")
        if fp:
            q = MetaculusAdapter.question_from_api(api_post_details_binary)
            assert fp in q.body
