# How Reliable Are LLM Judges?

A statistical audit of LLM-as-judge reliability — how much should you trust an LLM used to grade or score
other LLM outputs?

## Motivation

"LLM-as-judge" has become a default way to evaluate model outputs at scale (RLHF reward signals, eval
pipelines, model-vs-model comparisons), but judges are themselves unreliable, biased instruments whose
error properties are rarely quantified rigorously. This project treats an LLM judge the way a
psychometrician would treat a human rater: something whose reliability, bias, and calibration need to be
*measured*, not assumed.

## Methodology — three statistical lenses

1. **Classical inference** — inter-rater reliability between each judge and human ground-truth labels
   (Cohen's/Krippendorff's kappa), confusion matrices, calibration curves, and McNemar's test for
   judge-vs-judge disagreement.
2. **Causal inference / experiment design** — judge-prompt variant (e.g. "score directly" vs. "reason
   then score") is treated as a randomized treatment applied across the same item pool, with blocking on
   item and confidence intervals on the estimated effect on agreement-with-ground-truth — not just a
   raw score comparison.
3. **Bayesian modeling / uncertainty quantification** — a hierarchical (Rasch/IRT-style) model,
   implemented in PyMC, that jointly estimates each judge's leniency/bias and each item's true difficulty,
   with full posteriors and credible intervals rather than a single point-estimate accuracy number.

See [`TODO.md`](./TODO.md) for the build roadmap and [`CASE_STUDY.md`](./CASE_STUDY.md) for the
write-up of results.

## Project structure

```
data/raw/            ground-truth labeled items
data/processed/       judge outputs / scored data
notebooks/            exploratory analysis
dashboard/             Artifact dashboard source
src/llm_judge_audit/   library code (judges, classical_stats, causal_experiment, bayesian_model, viz)
tests/                 test suite
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m ensurepip --upgrade
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```

Run tests with `.venv/bin/pytest`.

## Status

Pipeline complete and tested (74 tests passing). A real 150-item run against the live Anthropic API
(`claude-haiku-4-5-20251001`) is done, analyzed, and written up — see `CASE_STUDY.md`. Results
dashboard: https://claude.ai/code/artifact/667fd490-ed70-473d-bc5b-12dbf49c3f47. See `TODO.md` for
full progress and remaining items.
