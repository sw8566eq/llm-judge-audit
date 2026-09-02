# How Reliable Are LLM Judges?

A statistical audit of LLM-as-judge, using classical inter-rater statistics, a paired causal
experiment, and a hierarchical Bayesian model, run against Claude Haiku 4.5 scoring 150 real
model-vs-model battles.

**Dashboard:** https://claude.ai/code/artifact/667fd490-ed70-473d-bc5b-12dbf49c3f47
**Code:** `src/llm_judge_audit/` + `scripts/run_pipeline.py`, 74 tests

## Question

Eval pipelines that use an LLM to judge model outputs often add a "reason step by step, then give
a verdict" prompt on the assumption that it makes the judge more reliable. Does it?

## Setup

150 battles from [MT-Bench Human Judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments),
each with a human-majority winner label. Each battle was scored by Claude Haiku 4.5 under two
prompts — direct (verdict only) and cot (reasoning, then verdict) — and under both answer
orderings, to catch position bias. A verdict that flips between orderings is scored
`tie (inconsistent)` rather than credited either way. 600 API calls total.

Three methods, cross-checked against each other:
- Classical: Cohen's κ, confusion matrices, McNemar's paired test.
- Causal: direct vs. cot as a within-item treatment (every battle judged both ways), effect and CI
  from an item-level bootstrap.
- Bayesian: a Rasch/IRT model (`agree ~ Bernoulli(sigmoid(ability_judge − difficulty_item))`) in
  PyMC, estimating judge reliability and item difficulty jointly with full posteriors.

## Results

|  | direct | cot |
|---|---|---|
| accuracy vs. human majority | 77.3% | 74.7% |
| Cohen's κ | 0.625 | 0.599 |
| position-bias rate (verdict flips on swap) | 7.3% | 13.3% |

**Accuracy: no effect.** The bootstrap estimates the direct→cot effect at −2.7 points, 95% CI
[−8.0, +2.7]; McNemar's test gives p = 0.48. The Bayesian model puts direct and cot's ability
posteriors at +0.088 and −0.088 with heavily overlapping 94% intervals. Both methods agree: no
detectable accuracy difference.

**Position bias roughly doubles.** 7.3% → 13.3% verdict flips on order swap. CoT costs consistency
without buying accuracy. Looking at where the flips land: for direct, 6 of 11 flips are on battles
humans scored a true tie — the close calls, as expected. For cot, 10 of 20 flips are on battles
humans scored a clean win — a shift toward flipping on battles that had a real answer. That's from
n=11 and n=20, so it's a pattern worth noting, not a claim to lean on.

**Accuracy tracks human agreement.** On the 129 battles with full human consensus, both configs
score ~81%. The next bucket down (n=15, humans split roughly 2:1) drops to 60% direct / 40% cot —
same direction as the position-bias result, but too small a sample to treat as confirmation.

## A bug that mattered

The first full run showed cot accuracy significantly worse than direct (p = 0.003) — the opposite
of what's usually reported, which is why I didn't trust it. Cause: the cot config had a 16%
unparseable-verdict rate, and every unparseable case had `stop_reason="max_tokens"` — 512 tokens
wasn't enough for the model to reason and still reach its verdict line, so about 1 in 6 cot calls
got cut off and counted as wrong instead of truncated.

Fix: per-variant `max_tokens` (1536 for cot), a tighter reasoning-length instruction, and
`stop_reason`/raw-response logging on every call so this is diagnosable without more live calls
next time. Added a regression test. A follow-up `/code-review high` pass then caught six more real
bugs, including a swap-order API failure that silently discarded an already-successful
first-order verdict, and a verdict parser that took the first regex match instead of the last
(which a cot response could spoof by mentioning "Verdict: X" while reasoning). All fixed, all
tested, before the run above.

## Limitations

- One judge model run live (Haiku 4.5). A Haiku-vs-Sonnet comparison exists only as synthetic demo
  data used to build the dashboard — not a real result.
- One dataset, which skews toward clear wins (129/150 battles had full human consensus); the
  contested cases are underpowered here.
- n=150. The accuracy CI doesn't rule out a small real effect either way, and the n=15/n=4
  difficulty buckets are too small to stand alone.
- Only position bias was measured; other known judge biases (verbosity, self-preference) weren't
  tested.

## Takeaway

If a CoT judge prompt is in your pipeline on the assumption that it improves reliability, check
that against your own data — here it didn't move accuracy and roughly doubled the rate at which
the judge disagreed with itself on order alone. Swap answer order and check consistency before
trusting a judge's accuracy number, regardless of prompt.

## Reproducing this

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY
.venv/bin/pytest
.venv/bin/python scripts/run_pipeline.py --n-items 150 --max-workers 8
```

See [`README.md`](./README.md) for structure, [`TODO.md`](./TODO.md) for the full build log.
