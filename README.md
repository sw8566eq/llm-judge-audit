# How Reliable Are LLM Judges?

A statistical audit of LLM-as-judge reliability: how much should you trust an LLM used to grade
or score other LLM outputs?

Three angles on the question — classical inter-rater statistics, a paired causal experiment on
judge-prompt design, and a hierarchical Bayesian (Rasch/IRT) model — run against Claude Haiku 4.5
scoring 150 real head-to-head battles from MT-Bench Human Judgments.

**Write-up:** [CASE_STUDY.md](./CASE_STUDY.md)
**Dashboard:** https://claude.ai/code/artifact/667fd490-ed70-473d-bc5b-12dbf49c3f47

## Structure

```
src/llm_judge_audit/   judges, classical_stats, causal_experiment, bayesian_model, viz
scripts/               run_pipeline.py (CLI), generate_fake_results.py
tests/                 test suite
dashboard/             dashboard source
data/                  raw items / judge outputs (gitignored)
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY
.venv/bin/pytest
```

Run the pipeline: `.venv/bin/python scripts/run_pipeline.py --n-items 150`
