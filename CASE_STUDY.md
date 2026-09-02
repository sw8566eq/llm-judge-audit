# How Reliable Are LLM Judges?

A statistical audit of LLM-as-judge, using classical inter-rater statistics, a paired causal
experiment, and a hierarchical Bayesian model, run against Claude Haiku 4.5 scoring 150 real
model-vs-model battles.

Dashboard: https://claude.ai/code/artifact/667fd490-ed70-473d-bc5b-12dbf49c3f47
Code: `src/llm_judge_audit/` + `scripts/run_pipeline.py`, 74 tests

## Question

A lot of eval pipelines add a "reason step by step, then give a verdict" prompt to their LLM
judge, on the assumption that it makes the judge more reliable. I wanted to check whether that's
actually true.

## Setup

150 battles from [MT-Bench Human Judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments),
each with a human-majority winner label. I scored every battle with Claude Haiku 4.5 under two
prompts — direct (verdict only) and cot (reasoning, then verdict) — and under both answer
orderings, to catch position bias. A verdict that flips between orderings gets scored
`tie (inconsistent)` instead of credited either way. 600 API calls total.

I checked the result three ways:
- Classical stats: Cohen's κ, confusion matrices, McNemar's paired test.
- A causal experiment: direct vs. cot as a within-item treatment (every battle judged both ways),
  effect and CI from an item-level bootstrap.
- A Bayesian model: a Rasch/IRT model (`agree ~ Bernoulli(sigmoid(ability_judge − difficulty_item))`)
  in PyMC, estimating judge reliability and item difficulty jointly.

## Results

|  | direct | cot |
|---|---|---|
| accuracy vs. human majority | 77.3% | 74.7% |
| Cohen's κ | 0.625 | 0.599 |
| position-bias rate (verdict flips on swap) | 7.3% | 13.3% |

Accuracy doesn't move. The bootstrap puts the direct→cot effect at −2.7 points, 95% CI
[−8.0, +2.7], and McNemar's test gives p = 0.48. The Bayesian model lands in the same place from a
different angle: direct and cot's ability posteriors come out to +0.088 and −0.088, with 94%
intervals that overlap almost entirely.

Position bias is the real story. It nearly doubles, from 7.3% to 13.3%, so cot costs you
consistency without buying any accuracy back. Where the flips land shifts too: for direct, most
flips (6 of 11) happen on battles humans themselves called a tie, the genuinely close calls. For
cot, most flips (10 of 20) happen on battles humans scored as a clean win. That's a small sample —
11 and 20 flips — so I'm not leaning on it hard, but it points toward cot flipping on battles that
had a real answer, not just the ambiguous ones.

Accuracy also tracks how much humans agreed with each other. On the 129 battles with full
consensus, both configs score around 81%. Drop to the 15 battles where humans split roughly 2:1,
and accuracy falls to 60% (direct) and 40% (cot) — same direction as the position-bias result,
though at n=15 I wouldn't call that confirmed.

## Limitations

- Only one judge model was run live, Haiku 4.5. A Haiku-vs-Sonnet comparison exists in the repo,
  but it's synthetic data used to build the dashboard, not a real result.
- One dataset, and it skews toward clear wins (129 of 150 battles had full human consensus), so
  the contested cases are underpowered here.
- n=150 total. The accuracy CI doesn't rule out a small real effect either way, and the smaller
  difficulty buckets (n=15, n=4) are too thin to stand on their own.
- I only measured position bias. Other known judge biases, like verbosity or self-preference,
  weren't in scope.

## Takeaway

If a CoT judge prompt is in your pipeline because it seems like it should be more reliable, it's
worth checking that against your own data. Here it didn't move accuracy and roughly doubled how
often the judge disagreed with itself depending on answer order. Swapping answer order and
checking consistency is a cheap thing to do before trusting any judge's accuracy number.

## Reproducing this

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY
.venv/bin/pytest
.venv/bin/python scripts/run_pipeline.py --n-items 150 --max-workers 8
```

See [`README.md`](./README.md) for structure and setup.
