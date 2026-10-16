"""Metaculus API client — question listing, forecast submission, comments."""

from __future__ import annotations

import asyncio
import logging
import os

import httpx

logger = logging.getLogger(__name__)


class MetaculusClient:
    """Thin wrapper around the Metaculus REST API."""

    def __init__(self, token: str | None = None, timeout: int = 30) -> None:
        self.token = token or os.getenv("METACULUS_TOKEN")
        if not self.token:
            raise ValueError(
                "METACULUS_TOKEN is not set. Set the environment variable or pass token=... to MetaculusClient."
            )
        self.base_url = "https://www.metaculus.com/api"
        self._headers = {"Authorization": f"Token {self.token}"}
        self._client = httpx.AsyncClient(timeout=timeout)
        self._detail_gate = asyncio.Lock()
        self._next_detail_at = 0.0

    async def close(self) -> None:
        """Close the underlying httpx client."""
        await self._client.aclose()

    # Cloudflare fronts the API: bursts of GETs get challenge pages with
    # these statuses. Backoff-and-retry is the only recovery.
    _RETRYABLE_STATUSES = (403, 429, 502, 503, 504)
    _RETRY_BASE_DELAY = 3.0  # seconds; doubled each attempt
    _DETAIL_PACING = 2.0  # seconds between per-post detail fetches

    # 4 retries (3/6/12/24s): 28 ladders have reached the 12s rung in prod
    # and recovered on the last attempt — one more rung turns an unlucky
    # streak into a slow success instead of a silently skipped question.
    async def _get_json(self, url: str, params: dict | None = None, retries: int = 4) -> dict:
        for attempt in range(retries + 1):
            response = await self._client.get(url, headers=self._headers, params=params)
            if response.is_success:
                return response.json()
            if response.status_code in self._RETRYABLE_STATUSES and attempt < retries:
                delay = self._RETRY_BASE_DELAY * 2**attempt
                logger.warning(
                    "GET %s returned %s; retrying in %.0fs", url, response.status_code, delay
                )
                await asyncio.sleep(delay)
                continue
            raise RuntimeError(response.text)
        raise AssertionError("unreachable")

    FORECAST_TYPES = ("binary", "multiple_choice", "numeric", "discrete", "date")

    # ── posting ──────────────────────────────────────────────────────

    async def post_question_comment(self, post_id: int, comment_text: str) -> None:
        """Post a comment on the question page as the bot user."""
        response = await self._client.post(
            f"{self.base_url}/comments/create/",
            json={
                "text": comment_text,
                "parent": None,
                "included_forecast": True,
                "is_private": True,
                "on_post": post_id,
            },
            headers=self._headers,
        )
        if not response.is_success:
            raise RuntimeError(response.text)

    async def post_question_prediction(self, question_id: int, forecast_payload: dict) -> None:
        """Post a forecast on a question."""
        url = f"{self.base_url}/questions/forecast/"
        response = await self._client.post(
            url,
            json=[{"question": question_id, **forecast_payload}],
            headers=self._headers,
        )
        logger.debug("Prediction Post status code: %s", response.status_code)
        if not response.is_success:
            raise RuntimeError(response.text)

    # ── reading ──────────────────────────────────────────────────────

    async def list_posts_from_tournament(
        self,
        tournament_id: int | str,
        offset: int = 0,
        count: int = 200,
    ) -> dict:
        """List {count} posts from {tournament_id}.

        Returns the API response envelope with ``results``, ``count``,
        ``next``, and ``previous`` keys.
        """
        url_qparams = {
            "limit": count,
            "offset": offset,
            # "order_by": "-hotness",
            # "forecast_type": "date",
            # "forecast_type": ",".join(self.FORECAST_TYPES),
            "tournaments": [tournament_id],
            # Note: client-side filtering below handles 'open' status.
            # The server-side statuses filter is intentionally omitted —
            # it can return stale results or miss posts whose status
            # transitions during the open window.
            # "include_description": "true",
        }
        url = f"{self.base_url}/posts/"
        return await self._get_json(url, params=url_qparams)

    async def get_open_question_ids_from_tournament(
        self,
        tournament_id: int | str,
    ) -> list[tuple[int, int]]:
        """Return [(question_id, post_id), …] for every open question,
        ordered by soonest ``scheduled_close_time`` first."""
        all_results: list[dict] = []
        offset = 0
        page_size = 50
        # Metaculus /posts/ often returns a non-null "next" even on the final
        # partial page (seen in live responses), so terminate on short page.
        for _ in range(100):  # hard safety cap against infinite loops
            posts = await self.list_posts_from_tournament(
                tournament_id, offset=offset, count=page_size
            )
            results = posts.get("results") or []
            all_results.extend(results)
            if len(results) < page_size:
                break
            if posts.get("next") is None:
                break
            offset += page_size

        logger.info(
            "Fetched %d total posts from tournament %s",
            len(all_results),
            tournament_id,
        )

        open_questions: list[tuple[str, int, int]] = []
        for post in all_results:
            question = post.get("question")
            if question is None:
                logger.debug(
                    "Skipping post %s (no question): %s",
                    post["id"],
                    post.get("title", "")[:80],
                )
                continue
            if question.get("status") != "open":
                logger.debug(
                    "Skipping post %s (status=%s): %s",
                    post["id"],
                    question.get("status"),
                    question.get("title", "")[:80],
                )
                continue
            logger.debug(
                "ID: %s  Q: %s  Closes: %s",
                question["id"],
                question["title"],
                question["scheduled_close_time"],
            )
            open_questions.append(
                (question.get("scheduled_close_time") or "", question["id"], post["id"])
            )
        open_questions.sort(key=lambda t: (t[0] == "", t))  # unknown close time last
        return [(qid, pid) for _, qid, pid in open_questions]

    async def get_post_details(self, post_id: int, *, with_cp: bool = False) -> dict:
        """Get all details about a post.

        ``with_cp=True`` includes aggregations and, when authenticated, the
        bot's own ``my_forecasts`` with per-question ``score_data``.
        """
        # The per-post endpoint is rate-limited far harder than list
        # endpoints: at 0.6s pacing every request in a run 429'd through
        # all retries (2026-07-07); 2s has been reliable. Enforced here,
        # across all callers of this method, so none can forget it.
        async with self._detail_gate:
            now = asyncio.get_running_loop().time()
            if now < self._next_detail_at:
                await asyncio.sleep(self._next_detail_at - now)
            self._next_detail_at = asyncio.get_running_loop().time() + self._DETAIL_PACING
        url = f"{self.base_url}/posts/{post_id}/"
        logger.debug("Getting details for %s", url)
        return await self._get_json(url, params={"with_cp": "true"} if with_cp else None)

    async def get_resolved_question_details(
        self,
        tournament_id: int | str,
        *,
        known_ids: set[int] | None = None,
        statuses: tuple[str, ...] = ("resolved",),
        with_cp: bool = False,
    ) -> list[dict]:
        """Return full post detail dicts for every question in *statuses*.

        Paginates tournament posts, filters by question status, fetches the
        full detail for each (includes axis spec and aggregations), and
        optionally skips *known_ids* (idempotency).
        """
        resolved_posts: list[dict] = []
        offset = 0
        page_size = 50
        for _ in range(100):  # hard safety cap
            posts = await self.list_posts_from_tournament(
                tournament_id, offset=offset, count=page_size
            )
            results = posts.get("results") or []
            for post in results:
                question = post.get("question")
                if question is None:
                    continue
                if question.get("status") not in statuses:
                    continue
                qid = question["id"]
                if known_ids and qid in known_ids:
                    continue
                try:
                    detail = await self.get_post_details(post["id"], with_cp=with_cp)
                except RuntimeError:
                    # Skip posts that stay throttled/failing so one bad post
                    # doesn't discard the whole pass; idempotent reruns
                    # (known_ids) pick them up next time.
                    logger.warning("skipping post %s: detail fetch failed", post["id"])
                    continue
                resolved_posts.append(detail)
            if len(results) < page_size:
                break
            if posts.get("next") is None:
                break
            offset += page_size
        return resolved_posts

    @staticmethod
    def forecast_already_made(post_details: dict) -> bool:
        """Check whether a forecast has already been submitted.

        Looks at question.my_forecasts.latest.forecast_values:
          Binary:  [p_no, p_yes]
          Numeric: [cdf_1, …, cdf_201]
          Multiple Choice: [p_option_1, …]
        """
        try:
            fv = post_details["question"]["my_forecasts"]["latest"]["forecast_values"]
            return fv is not None
        except (KeyError, TypeError):
            return False
