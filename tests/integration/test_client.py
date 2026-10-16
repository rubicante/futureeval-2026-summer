"""Integration tests for metaculus.client — MetaculusClient with mocked HTTP transport."""

from __future__ import annotations

import os
from unittest import mock

import pytest
from pytest_httpx import HTTPXMock

from metaculus.client import MetaculusClient


class TestInit:
    async def test_missing_token_raises(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="METACULUS_TOKEN"):
                MetaculusClient()

    async def test_token_from_env(self):
        with mock.patch.dict(os.environ, {"METACULUS_TOKEN": "x"}, clear=True):
            c = MetaculusClient()
            assert c.token == "x"
            await c.close()

    async def test_token_from_arg(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            c = MetaculusClient(token="abc123")
            assert c.token == "abc123"
            await c.close()

    async def test_default_timeout_does_not_raise(self):
        with mock.patch.dict(os.environ, {"METACULUS_TOKEN": "x"}):
            c = MetaculusClient()
            await c.close()

    async def test_custom_timeout_does_not_raise(self):
        with mock.patch.dict(os.environ, {"METACULUS_TOKEN": "x"}):
            c = MetaculusClient(timeout=60)
            await c.close()


class TestClose:
    async def test_close_prevents_subsequent_calls(self, metaculus_client):
        await metaculus_client.close()
        with pytest.raises(RuntimeError):
            await metaculus_client.get_post_details(1)


class TestListPostsFromTournament:
    async def test_returns_response_json(self, httpx_mock: HTTPXMock, metaculus_client):
        httpx_mock.add_response(
            url="https://www.metaculus.com/api/posts/?limit=200&offset=0&tournaments=bot-testing-area",
            json={"results": [], "next": None, "previous": None},
        )
        result = await metaculus_client.list_posts_from_tournament("bot-testing-area")
        assert result == {"results": [], "next": None, "previous": None}

    async def test_with_custom_count_and_offset(self, httpx_mock: HTTPXMock, metaculus_client):
        httpx_mock.add_response(
            url="https://www.metaculus.com/api/posts/?limit=10&offset=20&tournaments=cup",
            json={"results": [{"id": 1}], "next": None, "previous": None},
        )
        result = await metaculus_client.list_posts_from_tournament("cup", offset=20, count=10)
        assert len(result["results"]) == 1

    async def test_non_success_raises(self, httpx_mock: HTTPXMock, metaculus_client):
        httpx_mock.add_response(
            url="https://www.metaculus.com/api/posts/?limit=200&offset=0&tournaments=bad",
            status_code=401,
        )
        with pytest.raises(RuntimeError):
            await metaculus_client.list_posts_from_tournament("bad")

    async def test_server_error_retries_then_raises(
        self, httpx_mock: HTTPXMock, metaculus_client, monkeypatch
    ):
        monkeypatch.setattr(type(metaculus_client), "_RETRY_BASE_DELAY", 0.0)
        for _ in range(5):  # initial attempt + 4 retries
            httpx_mock.add_response(
                url="https://www.metaculus.com/api/posts/?limit=200&offset=0&tournaments=bad",
                status_code=503,
            )
        with pytest.raises(RuntimeError):
            await metaculus_client.list_posts_from_tournament("bad")

    async def test_server_error_retry_recovers(
        self, httpx_mock: HTTPXMock, metaculus_client, monkeypatch
    ):
        monkeypatch.setattr(type(metaculus_client), "_RETRY_BASE_DELAY", 0.0)
        url = "https://www.metaculus.com/api/posts/?limit=200&offset=0&tournaments=flaky"
        httpx_mock.add_response(url=url, status_code=429)
        httpx_mock.add_response(url=url, json={"results": [], "next": None, "previous": None})
        result = await metaculus_client.list_posts_from_tournament("flaky")
        assert result["results"] == []


class TestGetOpenQuestionIdsFromTournament:
    async def test_single_page(
        self, httpx_mock: HTTPXMock, base_url, metaculus_client, api_paginated_result_item
    ):
        item1 = api_paginated_result_item
        item2 = dict(item1)
        item2["id"] = 2
        item2["question"] = dict(item2["question"])
        item2["question"]["id"] = 2
        import json

        results = [json.loads(json.dumps(item1)), json.loads(json.dumps(item2))]
        for r in results:
            r["question"]["status"] = "open"
        httpx_mock.add_response(
            url=f"{base_url}/posts/?limit=50&offset=0&tournaments=test",
            json={"results": results, "next": None, "previous": None},
        )
        ids = await metaculus_client.get_open_question_ids_from_tournament("test")
        assert len(ids) == 2
        assert set(ids) == {
            (results[0]["question"]["id"], results[0]["id"]),
            (results[1]["question"]["id"], results[1]["id"]),
        }

    async def test_orders_by_soonest_close_time(
        self, httpx_mock: HTTPXMock, base_url, metaculus_client, api_paginated_result_item
    ):
        import json

        late = json.loads(json.dumps(api_paginated_result_item))
        soon = json.loads(json.dumps(api_paginated_result_item))
        soon["id"] = 2
        soon["question"]["id"] = 2
        for r, close in ((late, "2026-06-12T18:00:00Z"), (soon, "2026-06-12T16:00:00Z")):
            r["question"]["status"] = "open"
            r["question"]["scheduled_close_time"] = close
        httpx_mock.add_response(
            url=f"{base_url}/posts/?limit=50&offset=0&tournaments=test",
            json={"results": [late, soon], "next": None, "previous": None},
        )
        ids = await metaculus_client.get_open_question_ids_from_tournament("test")
        assert ids[0][0] == 2  # soonest close first

    async def test_filters_closed_questions(
        self, httpx_mock: HTTPXMock, base_url, metaculus_client, api_paginated_result_item
    ):
        import json

        item = json.loads(json.dumps(api_paginated_result_item))
        item["question"]["status"] = "closed"
        httpx_mock.add_response(
            url=f"{base_url}/posts/?limit=50&offset=0&tournaments=test",
            json={"results": [item], "next": None, "previous": None},
        )
        ids = await metaculus_client.get_open_question_ids_from_tournament("test")
        assert ids == []

    async def test_skips_missing_question(self, httpx_mock: HTTPXMock, base_url, metaculus_client):
        httpx_mock.add_response(
            url=f"{base_url}/posts/?limit=50&offset=0&tournaments=test",
            json={"results": [{"id": 1, "question": None}], "next": None, "previous": None},
        )
        ids = await metaculus_client.get_open_question_ids_from_tournament("test")
        assert ids == []

    async def test_pagination_multi_page(
        self, httpx_mock: HTTPXMock, base_url, metaculus_client, api_paginated_result_item
    ):
        import json

        item = json.loads(json.dumps(api_paginated_result_item))

        # Page 0: 50 results (full page)
        p0 = []
        for i in range(50):
            it = json.loads(json.dumps(item))
            it["id"] = i
            it["question"]["id"] = i
            it["question"]["status"] = "open"
            p0.append(it)
        httpx_mock.add_response(
            url=f"{base_url}/posts/?limit=50&offset=0&tournaments=test",
            json={
                "results": p0,
                "next": f"{base_url}/posts/?limit=50&offset=50&tournaments=test",
                "previous": None,
            },
        )
        # Page 1: 10 results (short page — terminates)
        p1 = []
        for i in range(10):
            it = json.loads(json.dumps(item))
            it["id"] = 50 + i
            it["question"]["id"] = 50 + i
            it["question"]["status"] = "open"
            p1.append(it)
        httpx_mock.add_response(
            url=f"{base_url}/posts/?limit=50&offset=50&tournaments=test",
            json={"results": p1, "next": None, "previous": None},
        )
        ids = await metaculus_client.get_open_question_ids_from_tournament("test")
        assert len(ids) == 60

    async def test_safety_cap_stops_after_100_pages(
        self, httpx_mock: HTTPXMock, base_url, metaculus_client, api_paginated_result_item
    ):
        import json

        item = json.loads(json.dumps(api_paginated_result_item))
        for offset in range(0, 5000, 50):
            p = []
            for i in range(50):
                it = json.loads(json.dumps(item))
                it["id"] = offset + i
                it["question"]["id"] = offset + i
                it["question"]["status"] = "open"
                p.append(it)
            httpx_mock.add_response(
                url=f"{base_url}/posts/?limit=50&offset={offset}&tournaments=test",
                json={
                    "results": p,
                    "next": f"{base_url}/posts/?limit=50&offset={offset + 50}&tournaments=test",
                    "previous": None,
                },
            )
        ids = await metaculus_client.get_open_question_ids_from_tournament("test")
        assert len(ids) == 100 * 50


class TestGetPostDetails:
    async def test_returns_json(
        self, httpx_mock: HTTPXMock, base_url, metaculus_client, api_post_details_binary
    ):
        httpx_mock.add_response(
            url=f"{base_url}/posts/578/",
            json=api_post_details_binary,
        )
        result = await metaculus_client.get_post_details(578)
        assert result["id"] == api_post_details_binary["id"]
        assert result["question"]["type"] == "binary"

    async def test_404_raises(self, httpx_mock: HTTPXMock, base_url, metaculus_client):
        httpx_mock.add_response(url=f"{base_url}/posts/99999/", status_code=404)
        with pytest.raises(RuntimeError):
            await metaculus_client.get_post_details(99999)

    async def test_auth_error_raises(self, httpx_mock: HTTPXMock, base_url, metaculus_client):
        httpx_mock.add_response(url=f"{base_url}/posts/578/", status_code=401)
        with pytest.raises(RuntimeError):
            await metaculus_client.get_post_details(578)


class TestPostQuestionPrediction:
    async def test_binary_payload(
        self, httpx_mock: HTTPXMock, base_url, metaculus_client, mock_forecast_submit
    ):
        mock_forecast_submit()
        forecast_payload = {
            "probability_yes": 0.42,
            "probability_yes_per_category": None,
            "continuous_cdf": None,
        }
        await metaculus_client.post_question_prediction(123, forecast_payload)

    async def test_numeric_cdf_payload(
        self, httpx_mock: HTTPXMock, base_url, metaculus_client, mock_forecast_submit
    ):
        mock_forecast_submit()
        cdf = [i / 200 for i in range(201)]
        forecast_payload = {
            "probability_yes": None,
            "probability_yes_per_category": None,
            "continuous_cdf": cdf,
        }
        await metaculus_client.post_question_prediction(456, forecast_payload)

    async def test_error_raises(
        self, httpx_mock: HTTPXMock, base_url, metaculus_client, mock_forecast_submit
    ):
        mock_forecast_submit(status_code=400)
        with pytest.raises(RuntimeError):
            await metaculus_client.post_question_prediction(
                123,
                {
                    "probability_yes": 0.5,
                    "probability_yes_per_category": None,
                    "continuous_cdf": None,
                },
            )


class TestPostQuestionComment:
    async def test_posts_comment(
        self, httpx_mock: HTTPXMock, base_url, metaculus_client, mock_comment_post
    ):
        mock_comment_post()
        await metaculus_client.post_question_comment(578, "Test comment text")

    async def test_error_raises(
        self, httpx_mock: HTTPXMock, base_url, metaculus_client, mock_comment_post
    ):
        mock_comment_post(status_code=400)
        with pytest.raises(RuntimeError):
            await metaculus_client.post_question_comment(578, "Bad comment")


class TestForecastAlreadyMade:
    def test_no_forecast(self):
        details = {
            "question": {
                "my_forecasts": {
                    "latest": None,
                }
            }
        }
        assert MetaculusClient.forecast_already_made(details) is False

    def test_forecast_present(self):
        details = {
            "question": {
                "my_forecasts": {
                    "latest": {
                        "forecast_values": [0.3, 0.7],
                    }
                }
            }
        }
        assert MetaculusClient.forecast_already_made(details) is True

    def test_missing_my_forecasts(self):
        details = {"question": {}}
        assert MetaculusClient.forecast_already_made(details) is False

    def test_null_forecast_values(self):
        details = {
            "question": {
                "my_forecasts": {
                    "latest": {
                        "forecast_values": None,
                    }
                }
            }
        }
        assert MetaculusClient.forecast_already_made(details) is False

    def test_from_real_api_shape(self, api_post_details_binary):
        assert MetaculusClient.forecast_already_made(api_post_details_binary) is False

    def test_from_api_shape_with_forecast(self, api_post_details_date):
        assert MetaculusClient.forecast_already_made(api_post_details_date) is True


class TestGetResolvedQuestionDetails:
    @staticmethod
    def _resolved_list_response(results: list[dict]) -> dict:
        return {"results": results, "next": None, "previous": None}

    @staticmethod
    def _list_item(question_id: int, post_id: int, status: str) -> dict:
        return {"id": post_id, "question": {"id": question_id, "status": status}}

    async def test_returns_resolved_details(self, httpx_mock: HTTPXMock, metaculus_client):
        httpx_mock.add_response(
            url="https://www.metaculus.com/api/posts/?limit=50&offset=0&tournaments=42",
            json=self._resolved_list_response([self._list_item(100, 200, "resolved")]),
        )
        httpx_mock.add_response(
            url="https://www.metaculus.com/api/posts/200/",
            json={
                "id": 200,
                "actual_resolve_time": "2025-01-01T00:00:00Z",
                "question": {
                    "id": 100,
                    "type": "binary",
                    "resolution": "yes",
                    "scaling": None,
                },
            },
        )
        resolved = await metaculus_client.get_resolved_question_details("42")
        assert len(resolved) == 1
        detail = resolved[0]
        assert detail["question"]["id"] == 100
        assert detail["question"]["resolution"] == "yes"

    async def test_skips_known_ids(self, httpx_mock: HTTPXMock, metaculus_client):
        httpx_mock.add_response(
            url="https://www.metaculus.com/api/posts/?limit=50&offset=0&tournaments=42",
            json=self._resolved_list_response(
                [
                    self._list_item(100, 200, "resolved"),
                    self._list_item(101, 201, "resolved"),
                ]
            ),
        )
        httpx_mock.add_response(
            url="https://www.metaculus.com/api/posts/201/",
            json={
                "id": 201,
                "actual_resolve_time": None,
                "question": {
                    "id": 101,
                    "type": "numeric",
                    "resolution": 42.0,
                    "scaling": {"range_min": 0.0, "range_max": 100.0},
                },
            },
        )
        resolved = await metaculus_client.get_resolved_question_details("42", known_ids={100})
        assert len(resolved) == 1
        assert resolved[0]["question"]["id"] == 101

    async def test_skips_non_resolved(self, httpx_mock: HTTPXMock, metaculus_client):
        httpx_mock.add_response(
            url="https://www.metaculus.com/api/posts/?limit=50&offset=0&tournaments=42",
            json=self._resolved_list_response(
                [
                    self._list_item(100, 200, "open"),
                    self._list_item(101, 201, "resolved"),
                ]
            ),
        )
        httpx_mock.add_response(
            url="https://www.metaculus.com/api/posts/201/",
            json={
                "id": 201,
                "actual_resolve_time": None,
                "question": {
                    "id": 101,
                    "type": "binary",
                    "resolution": "no",
                    "scaling": None,
                },
            },
        )
        resolved = await metaculus_client.get_resolved_question_details("42")
        assert len(resolved) == 1
        assert resolved[0]["question"]["id"] == 101
