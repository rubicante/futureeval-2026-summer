# scuttlebot

A forecasting bot for Metaculus's
[Summer FutureEval 2026 bot tournament](https://www.metaculus.com/tournament/summer-futureeval-2026/).

A cron job runs `main.py` every 20 minutes to forecast open questions.

This repository is a release snapshot of the bot as it ran in that
tournament; see [`NOTICE`](NOTICE) for provenance.

## How it forecasts

An LLM decomposes the question into search queries: one per resolution
clause, a recency sweep, and a base-rate reference class. The queries go
to Exa and AskNews. One follow-up round covers gaps found in the first
results. The forecast is the median of samples from OpenAI, Anthropic,
and Gemini models. Errors decorrelate across families, and no family
holds a majority of samples.

Two paths specialize. Recurring financial-series questions (index
spreads, commodity prices, VIX, Treasury yields) are forecast from the
empirical volatility of the underlying series instead of LLM-elicited
distributions. When Polymarket or Manifold has a matching market, its
price is added to the research context as evidence.

Every run appends capture records: research context, per-sample model
attribution, code version, cost. Behavior changes ship behind config
knobs so score movements can be attributed to them later.

## Map

- `main.py`: forecast a tournament's open questions; `--submit` posts them
- `ARCHITECTURE.md`: standing constraints and design rationale
- `NOTICE`: provenance for this release snapshot

All entry points take `--help`. API keys go in `.env`.

## License

MIT
