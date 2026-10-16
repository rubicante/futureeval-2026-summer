# Architecture — design rationale

Why scuttlebot is built the way it is: the constraints and decisions a
reader of only the code would get wrong or accidentally undo. The code
is the source of truth for *how*.

Rules for this file: statements of fact, not history. No tunable
values, model names, sample counts, or current settings — nothing that
goes stale on an ordinary code change. Update only when a constraint or
decision changes.

---

## Tournament constraints

- Spot peer score at question close, zero-sum against other bots.
- Questions open for a few hours; missing the window forfeits the
  question. The game is single-shot quality plus never missing a
  window. Re-forecasting and crowd-tracking are dead strategies.
- The crowd is invisible: the community prediction is hidden while
  questions are open, and the API serves no aggregation data to
  external consumers even after close.
- Scoring noise exceeds realistic effect sizes; in-season A/Bs decide
  nothing. Ship on priors, capture provenance for retrospective
  attribution. The platform's single-model baseline bots are the
  standing control.

## Platform facts

- MC forecasts are keyed by server-owned option strings, validated at
  submit time. Options must be re-keyed canonically against the current
  question, never trusted from LLM output.
- The API dual-identifies content as post and question, with per-type
  submission formats. The adapter is the single seam that absorbs this;
  everything else operates on a platform-agnostic question type.

## Invariants

- Partial results beat no results. Failures are isolated per search,
  per ensemble sample, and per question; only total failure aborts.
- Research latency is bounded: one follow-up round, hard token cap on
  synthesis. Every retrieved token is multiplied by ensemble size, so
  the cap is the cost gate.
- The ensemble decorrelates model families; no family holds a majority
  of samples. Allocation is the primary dial, weights second-order.
- Every behavior change ships behind a config knob; every run's knobs,
  models, code version, and costs land in the capture record. Never
  make a change that can't be attributed later.
- Plain append-only files, no database. Logs for humans, captures for
  machines; capture rows are immutable and schema-versioned.

## Seams

- Gateway: pipeline code requests a logical purpose, never a provider
  or model. Provider differences (retries, prompt caching) are absorbed
  behind it.
- Context dict: research hands forecasting an untyped dict so new
  prompt variables need no type changes.
- Two-phase parsing: free-text reasoning first, structured extraction
  second — schema-constrained generation degrades the reasoning. The
  rationale is preserved for audit.
- Primitives: each question type bundles prompt, parse, aggregate, and
  trim, keeping the pipeline generic. Trim returns keep-indices, not
  values, to avoid ambiguity on duplicates.
- Protocols, not base classes; live wiring config is separate from
  serialized provenance config.

## Specialized paths

- Quant: LLM-elicited CDFs on recurring financial series are
  near-uniform slabs; empirical-volatility distributions beat them. An
  allowlist routes matched series; the sanity gate may only widen quant
  output, never sharpen it.
- Tail humility: one overconfident CDF loses more than many good
  forecasts win; non-quant continuous forecasts get a wide mixture
  component. The quant path carries its own.
- Market anchoring: prediction markets are the only visible crowd.
  Matched prices enter the research context as evidence behind a
  congruence gate; they are not blended into the forecast until
  blending earns evidence on resolutions.
