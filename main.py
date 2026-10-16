"""Scuttlebot entry point.

Run with:
    python main.py
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import dotenv

from common.config import DEFAULT_FORECAST_CONFIG, DEFAULT_RESEARCH_CONFIG, BotConfig
from common.models import CURRENT_RUN_CONFIG_VERSION, CaptureRecord, RunConfig
from forecasting import PRIMITIVES, EnsembleHealth, StrategyResult, format_forecast, strategy_for
from forecasting.quant import (
    QuantGateFlagged,
    match_quant_series,
    quant_shadow_info,
    quant_strategy_result,
)
from llm.costs import LEDGER, current_question
from llm.presets import build_default_gateway
from metaculus import MetaculusAdapter, MetaculusClient
from research.quant_assist import quant_modes

logger = logging.getLogger(__name__)


def _get_code_version() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _parse_config_entries(entries: list[str]) -> dict[str, Any]:
    """Parse ``KEY=VALUE`` entries into a typed dict.

    Auto-types values: ``int`` → ``float`` → ``"true"/"false"`` → ``str``.
    """
    cfg: dict[str, Any] = {}
    for entry in entries:
        if "=" not in entry:
            raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {entry!r}")
        key, _, value = entry.partition("=")
        if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            cfg[key] = int(value)
        else:
            try:
                cfg[key] = float(value)
            except ValueError:
                if value.lower() in ("true", "false"):
                    cfg[key] = value.lower() == "true"
                else:
                    cfg[key] = value
    return cfg


@dataclass
class ForecastResult:
    """Return type for ``forecast_individual_question`` — summary text plus health."""

    summary: str
    health: EnsembleHealth | None


# ═══════════════════════════════════════════════════════════════════
#  Orchestrator
# ═══════════════════════════════════════════════════════════════════


async def forecast_individual_question(
    client: MetaculusClient,
    question_id: int,
    post_id: int,
    config: BotConfig,
) -> ForecastResult:
    """Forecast a single question, post results if configured."""
    current_question.set(question_id)  # attribute LLM spend to this question
    post_details = await client.get_post_details(post_id)
    question = MetaculusAdapter.question_from_api(post_details)
    title = question.title
    question_type = question.question_type.value

    summary = f"\nQuestion: {title}\n"
    summary += f"URL: {question.url}\n"

    if question.options:
        summary += f"options: {question.options}\n"

    if client.forecast_already_made(post_details) and not config.reforecast:
        summary += "Skipped: Forecast already made\n"
        return ForecastResult(summary=summary, health=None)

    # Look up the question type primitive and dispatch through strategy
    prim = PRIMITIVES.get(question_type)
    if prim is None:
        raise ValueError(f"Unsupported question type: {question_type}")
    strategy = strategy_for(config.strategy, disable_research=config.disable_research)

    # Quant path for recurring financial series. --fc quant= takes
    # comma-separated flags: "series" (deterministic forecast replaces the
    # ensemble on matched questions, behind a widen-only LLM sanity gate),
    # "shadow" (ensemble submits; quant percentiles ride along in the
    # capture ctx), "assist" (resolution-source stats injected into every
    # detected question's research context — applied inside Strategy.run).
    modes = quant_modes(config.forecast_config.get("quant"))
    quant_spec = None
    if modes & {"series", "shadow"} and question_type == "numeric":
        quant_spec = match_quant_series(question.title, question.body)

    if quant_spec is not None and "series" in modes:
        try:
            result: StrategyResult = await quant_strategy_result(
                question, quant_spec, gateway=config.gateway
            )
            summary += f"Quant series: {quant_spec.name}\n"
        except QuantGateFlagged as flagged:
            logger.info("quant gate flagged (%s); using LLM strategy", flagged)
            result = await strategy.run(question, config, prim)
        except Exception:
            logger.warning(
                "quant path failed for %s; falling back to LLM strategy",
                quant_spec.name,
                exc_info=True,
            )
            result = await strategy.run(question, config, prim)
    else:
        result = await strategy.run(question, config, prim)
        if quant_spec is not None and "shadow" in modes:
            shadow = await quant_shadow_info(question, quant_spec)
            if shadow is not None:
                result.ctx["_quant_shadow"] = shadow
                summary += f"Quant shadow: {quant_spec.name}\n"
    forecast = result.forecast
    comment = result.comment
    summary += f"Forecast: {format_forecast(forecast, question)}\n"
    if result.health:
        summary += f"{result.health.format_one_line()}\n"
    logger.info(f"\nQuestion: {title}\nURL: {question.url}\n{comment}")

    if config.submit:
        forecast_payload = MetaculusAdapter.forecast_to_payload(forecast, question_type)
        await client.post_question_prediction(question_id, forecast_payload)
        summary += "Posted: Forecast was posted to Metaculus.\n"

    _write_capture(config, question_id, result, question.community_prediction_raw)

    return ForecastResult(summary=summary, health=result.health)


def _write_capture(
    config: BotConfig,
    question_id: int,
    result: StrategyResult,
    community_prediction_raw: dict[str, Any] | None,
) -> None:
    run_cfg = config.run_cfg
    if run_cfg is None or config.run_id is None:
        return

    capture = CaptureRecord(
        code_version=config.code_version,
        run_id=config.run_id,
        question_id=question_id,
        snapshot_timestamp=datetime.datetime.now(datetime.UTC),
        run_config=run_cfg,
        context=result.ctx,
        raw_results=result.raw_results,
        sample_models=result.sample_models,
        final_forecast=result.forecast,
        community_prediction=community_prediction_raw,
        cost=LEDGER.question_cost(question_id),
    )
    captures_dir = Path("runs")
    captures_dir.mkdir(exist_ok=True)
    captures_path = captures_dir / f"captures_{config.run_id}.jsonl"
    with captures_path.open("a") as f:
        f.write(capture.model_dump_json() + "\n")


# One hung search or LLM call must not eat the 1.5–3h tournament window.
QUESTION_TIMEOUT_SECONDS = 900


async def forecast_questions(
    client: MetaculusClient,
    open_question_id_post_id: list[tuple[int, int]],
    config: BotConfig,
) -> None:
    """Forecast every question concurrently, gather results, report errors.

    The input list is ordered soonest-closing-first; questions run
    concurrently (the gateway semaphore bounds LLM concurrency) with a
    per-question timeout so one stall can't forfeit the rest of the batch.
    """

    async def one(qid: int, pid: int) -> ForecastResult:
        async with asyncio.timeout(QUESTION_TIMEOUT_SECONDS):
            return await forecast_individual_question(client, qid, pid, config)

    forecast_results = await asyncio.gather(
        *[one(qid, pid) for qid, pid in open_question_id_post_id],
        return_exceptions=True,
    )
    logger.info("\n%s\nForecast Summaries\n%s", "#" * 100, "#" * 100)

    errors: list[BaseException] = []
    healths: list[EnsembleHealth] = []
    for (qid, pid), result in zip(open_question_id_post_id, forecast_results, strict=True):
        if isinstance(result, BaseException):
            logger.error(
                "Post %s Question %s:\nError: %s %s\nURL: https://www.metaculus.com/questions/%s/",
                pid,
                qid,
                result.__class__.__name__,
                result,
                pid,
            )
            errors.append(result)
        else:
            logger.info(result.summary)
            if result.health:
                healths.append(result.health)

    # ── aggregate ensemble health across run ────────────────────────
    if healths:
        total_planned = 0
        total_survived = 0
        by_family: dict[str, tuple[int, int]] = {}
        for h in healths:
            total_planned += h.planned
            total_survived += h.survived
            for family, (p, s) in h.by_family.items():
                tp, ts = by_family.get(family, (0, 0))
                by_family[family] = (tp + p, ts + s)
        aggregate = EnsembleHealth(
            planned=total_planned,
            survived=total_survived,
            by_family=by_family,
        )
        logger.info("Run health: %s", aggregate.format_one_line())

    if errors:
        logger.error("-----------------------------------------------\nErrors:\n")
        error_message = f"Errors were encountered: {errors}"
        logger.error(error_message)
        raise RuntimeError(error_message)


# ═══════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════


async def main(config: BotConfig) -> None:
    """Run the forecasting bot."""
    dotenv.load_dotenv()
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_file = runs_dir / f"run_{timestamp}.log"
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[stream_handler, file_handler],
    )
    logging.getLogger("forecasting").setLevel(logging.DEBUG)
    logging.getLogger("research").setLevel(logging.DEBUG)
    logging.getLogger("metaculus").setLevel(logging.DEBUG)

    logger.info(
        "Config: live=%s limit=%d submit=%s strategy=%s reforecast=%s "
        "disable_research=%s tournament_id=%s forecast_config=%s research_config=%s "
        "code_version=%s",
        config.live,
        config.limit,
        config.submit,
        config.strategy,
        config.reforecast,
        config.disable_research,
        config.tournament_id or "none",
        config.forecast_config,
        config.research_config,
        config.code_version or "none",
    )

    client = MetaculusClient()
    try:
        open_ids: list[tuple[int, int]] = []
        if config.tournament_id:
            open_ids = await client.get_open_question_ids_from_tournament(config.tournament_id)

        if config.tournament_id and not open_ids:
            logger.info("No open questions found")
            return

        if config.limit:
            open_ids = open_ids[: config.limit]

        await forecast_questions(client, open_ids, config)
    finally:
        await client.close()
        logger.info(LEDGER.format_summary())


def parse_args() -> SimpleNamespace:
    config = SimpleNamespace()
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--submit", action="store_true", default=False)
    parser.add_argument("-t", "--tournament-id", type=str, required=True)
    parser.add_argument("--reforecast", action="store_true", default=False)
    parser.add_argument("--disable-research", action="store_true", default=False)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Use cheap model tiers (live tiers are the default)",
    )
    # Deprecated no-op: live is now the default. Kept so existing cron
    # invocations don't break; remove once the crontab is updated.
    parser.add_argument("--live", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--strategy", choices=["ensemble"], default="ensemble")
    parser.add_argument("-l", "--limit", type=int, default=0)
    parser.add_argument(
        "--fc",
        "--forecast-config",
        dest="forecast_config_entries",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Forecast-strategy knobs (repeatable, e.g. --fc tail=0); "
        "defaults: " + " ".join(f"{k}={v}" for k, v in DEFAULT_FORECAST_CONFIG.items()),
    )
    parser.add_argument(
        "--rc",
        "--research-config",
        dest="research_config_entries",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Research-strategy knobs (repeatable, e.g. --rc plan=legacy); "
        "defaults: " + " ".join(f"{k}={v}" for k, v in DEFAULT_RESEARCH_CONFIG.items()),
    )
    parser.parse_args(namespace=config)
    return config


if __name__ == "__main__":
    dotenv.load_dotenv()
    args = parse_args()
    if args.live:
        print(
            "--live is deprecated (live tiers are now the default); "
            "use --dry-run for cheap tiers",
            file=sys.stderr,
        )
    live = not args.dry_run
    gateway, forecast_model, decompose_model, forecast_models = build_default_gateway(live)
    forecast_config = {
        **DEFAULT_FORECAST_CONFIG,
        **_parse_config_entries(args.forecast_config_entries),
    }
    research_config = {
        **DEFAULT_RESEARCH_CONFIG,
        **_parse_config_entries(args.research_config_entries),
    }
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    config = BotConfig(
        gateway=gateway,
        live=live,
        submit=args.submit,
        limit=args.limit,
        tournament_id=args.tournament_id,
        reforecast=args.reforecast,
        disable_research=args.disable_research,
        strategy=args.strategy,
        forecast_config=forecast_config,
        research_config=research_config,
        run_id=timestamp,
        code_version=_get_code_version(),
        run_cfg=RunConfig(
            schema_version=CURRENT_RUN_CONFIG_VERSION,
            research_strategy=("no_research" if args.disable_research else "planned_research"),
            research_provider="exa",
            forecast_strategy=args.strategy,
            forecast_config=forecast_config,
            research_config=research_config,
            forecast_model=forecast_model,
            decompose_model=decompose_model,
            forecast_models=forecast_models,
        ),
    )
    asyncio.run(main(config))
