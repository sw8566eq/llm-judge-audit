# Roadmap

- [x] Scaffold repo structure and environment
- [x] Source a ground-truth item set — using MT-Bench Human Judgments (`lmsys/mt_bench_human_judgments`
      on Hugging Face); `load_mt_bench_items()` in `judges.py` loads and prepares it
- [x] Implement LLM judge orchestration (`src/llm_judge_audit/judges.py::run_judges`) across prompt variants
- [x] Run a real (paid) judge pass against the Anthropic API — small `--n-items 5` smoke test
      (20 calls) completed cleanly first: 0 errors, position-bias consolidation verified correct on
      real responses.
- [x] Full `--n-items 150` run (600 calls, claude-haiku-4-5-20251001): direct accuracy=0.77 (κ=0.62),
      **cot accuracy=0.66 (κ=0.50) — significantly worse** (effect=-0.11, 95% CI [-0.19,-0.05],
      McNemar p=0.003).
- **Bug found and fixed, not yet re-run:** that "cot is worse" result was suspicious (opposite of
  the synthetic demo assumption, and opposite of the LLM-judge literature's usual claim), so before
  trusting it: `claude-haiku-4-5-20251001-cot` had a **16% unparseable rate** (vs. 0% for direct) —
  and an unparseable verdict counts as "wrong," which can manufacture exactly this kind of gap. A
  targeted diagnostic (4 more live calls, cot prompt, same items) showed **every unparseable case
  had `stop_reason="max_tokens"`**: the old `max_tokens=512` default wasn't enough for the model to
  finish reasoning *and* reach the "Verdict: X" line, so ~1 in 6 cot calls got cut off mid-thought
  and were then wrongly counted as a bad judgment rather than a truncated one.
  - Fixed in `judges.py`: `JudgeConfig.max_tokens` now defaults per-variant (direct=256, cot=1536,
    via `_DEFAULT_MAX_TOKENS`) instead of a flat 512; the cot prompt now asks for reasoning "in 3-5
    sentences" as a second line of defense; `run_judges` now also records `stop_reason_*` and
    `response_*` (raw text) columns so this class of bug is diagnosable from saved results next
    time, without needing more live calls to investigate.
  - Added a regression test (`test_run_judges_flags_truncated_response_via_stop_reason`) so a
    silent truncation-as-wrong-answer bug like this can't reappear unnoticed.
  - **The 150-item run above is now known-unreliable for the cot config and needs to be redone**
    with the fix before the "does chain-of-thought help or hurt" question has a trustworthy answer.
- [x] Implement classical reliability/calibration stats (`src/llm_judge_audit/classical_stats.py`)
- [x] Design & run the causal experiment on judge-prompt variants (`src/llm_judge_audit/causal_experiment.py`)
- [x] Implement hierarchical Bayesian model (`src/llm_judge_audit/bayesian_model.py`)
- [x] Implement local plotting + dashboard-JSON prep (`src/llm_judge_audit/viz.py`) — calibration
      curve, confusion matrix, and posterior-interval plots, plus `to_dashboard_json` combining
      `evaluate_judge` / `estimate_*_effect` / `bayesian_model.summarize` outputs into one payload
- [x] Build the results dashboard (Artifact, via the `dataviz` skill) — first built against
      synthetic data, then **rebuilt with the real 150-item run's data**. Source:
      `dashboard/index.html`. Published at
      https://claude.ai/code/artifact/667fd490-ed70-473d-bc5b-12dbf49c3f47
  - Rebuilding onto real (single-model) data caught one more real bug: the accuracy chart
    hardcoded `order = ["haiku", "sonnet"]` — the real run only used one model, so that line would
    throw (`byModel["sonnet"]` undefined) and, since it's the first of several sequential
    `<script>` statements, silently kill every chart after it too. Fixed to derive the model list
    from whatever's actually in the data (`Object.keys(byModel).sort()`), so it now scales to 1,
    2, or more models without editing the dashboard again.
  - Banner/KPI tiles/footer rewritten from the synthetic-data framing to the real finding: 77%
    direct-prompt accuracy (κ=0.62); CoT's accuracy effect is not significant (−3 pts, 95% CI
    [−8,+3], McNemar p=0.48); but CoT nearly doubles the position-bias rate (7.3% → 13.3%) despite
    the flat accuracy — the actual headline result of this run.
  - Verified with the same headless jsdom render check as the first build: 0 JS errors, 0
    NaN/undefined, chart element counts matching the real 2-config data shape.
- [x] Write the case-study / blog post — `CASE_STUDY.md`, grounded in the real 150-item run's
      numbers (`data/processed/dashboard.json`). Covers all three statistical lenses, the
      position-bias finding, an honest small-n caveat on the difficulty-breakdown subgroups, the
      max_tokens-truncation bug as a process/rigor story, and a limitations section (single model
      run live, single dataset, n=150). README updated to link it and drop the stale
      "not yet implemented" status line.
- [ ] `git init`, first commit, push to GitHub (user-driven)

## Notes / open questions

- **High-effort code review (`/code-review high`) run across `src/llm_judge_audit/` and
  `scripts/`** after the max_tokens bug, specifically to catch this class of mistake. 6 of 7
  findings fixed immediately (all confirmed real, not false positives):
  - `judges.py::run_judges` — **data-loss bug**: the original-order and swap-order API calls
    shared one try/except, so a failure in the *second* (already-paid-for) call discarded the
    first call's successful verdict too. Fixed: separate try/excepts; a swap-call failure now
    keeps `verdict_original_order`/`response_original_order`/`stop_reason_original_order` instead
    of wiping them to `None`. Regression test added.
  - `judges.py::_parse_verdict` — used the *first* regex match, not the last, even though the cot
    prompt asks for the verdict "on the final line." A cot response reasoning out loud could
    mention "Verdict: X"-shaped phrasing before its real answer and get the wrong one parsed out.
    Fixed: now takes the last match. Regression test added.
  - `bayesian_model.py::fit` — caught bare `Exception` and *always* blamed it on missing C
    compiler, silently re-running the full (expensive) sampling a second time even for an
    unrelated real bug. Fixed: narrowed to `pytensor.link.c.exceptions.CompileError` specifically.
  - `pipeline.py::analyze_results` — a leftover `except ValueError` around
    `accuracy_by_human_agreement` was dead code from the pre-fix (qcut-based) version, which
    silently masked any *future* unrelated ValueError. Removed.
  - `classical_stats.py::mcnemars_test` — docstring described the returned `table`'s rows/columns
    transposed from what the code actually returns (statistic/p-value themselves were unaffected —
    McNemar's is symmetric in the off-diagonal cells — but a caller reading `table` directly by the
    documented convention would get it backwards). Fixed docstring; added a test locking in the
    actual orientation.
  - `run_pipeline.py` / `generate_fake_results.py` — near-identical summary-printing logic
    duplicated between the two scripts (drift risk). Factored into
    `pipeline.print_analysis_summary`, used by both.
  - **Also now fixed (was flagged as a separate tradeoff, then addressed on request):**
    `run_judges` was fully sequential. Refactored the per-(item, config) scoring into `_score_one`
    and run those units concurrently via `ThreadPoolExecutor` (`max_workers`, default 8; one
    shared `anthropic.Anthropic` client across threads, which is the SDK's documented-safe
    pattern). `max_workers=1` gives fully sequential/deterministic execution, used by all
    fake-client tests. Added a timing-based test (`test_run_judges_runs_jobs_concurrently`) that
    verifies an actual ~6x wall-clock speedup, not just that the parameter is accepted — stable
    across repeated runs. `scripts/run_pipeline.py` got a `--max-workers` flag (default 8).
- **All 7 code-review findings now addressed** (6 fixed outright, concurrency added on request).
  Full suite: 74/74 passing. Ready for the real 150-item re-run.

- `pipeline.py` added: `analyze_results(results, judge_configs, items=None, seed=42)` factors the
  "results -> classical/causal/Bayesian summaries -> dashboard JSON" orchestration out of
  `scripts/run_pipeline.py` so it's shared with `scripts/generate_fake_results.py` (and any future
  results source) rather than duplicated.
- `scripts/generate_fake_results.py`: simulates 4 judge configs (2 models x direct/cot) scoring the
  real 120-item MT-Bench sample, with fabricated-but-plausible accuracy rates (reasoning helps both
  models; sonnet > haiku). Writes `data/processed/fake_*` — clearly prefixed so it can't be confused
  with a real run. Useful any time the dashboard/analysis code needs exercising without spending on
  the live API.
- **Found and fixed while dogfooding:** `accuracy_by_human_agreement` originally used
  `pd.qcut` quantile binning, which collapses to a single bin whenever a majority of items share one
  `human_agreement` value — common here, since most MT-Bench battles have full human agreement.
  Switched to grouping by the exact value (`human_agreement` is near-discrete: only a handful of
  possible values for a small vote count), which is both more robust and more interpretable. Caught
  this by actually running the pipeline on real ground-truth items, not just synthetic unit-test data.
  API changed: `n_bins` param removed; output columns are now `human_agreement`/`n`/`judge_accuracy`.

- **Dataset:** MT-Bench Human Judgments — 80 questions, pairwise "battles" between model answers,
  independent human votes per battle (up to 5; 761 battles have >1 vote, letting us measure human-human
  inter-rater agreement as a baseline), plus a GPT-4-as-judge verdict on the same battles as a reference
  point. `load_mt_bench_items()` aggregates multi-vote battles into a majority `human_winner` +
  `human_agreement` fraction, drops exact-tie/no-consensus battles, and samples up to `n_items`
  (default 200), preferring multi-vote battles first.
- Judge prompt variants implemented: "direct" (verdict only) vs. "cot" (reason then verdict). The task
  itself is always pairwise (model_a vs. model_b vs. tie), matching the dataset's `winner` field, so
  "pairwise" isn't a separate variant.
- Position-bias handling: `JudgeConfig.swap_order` (default True) runs each battle in both answer
  orderings and reports `"tie (inconsistent)"` when they disagree — mirrors the convention already used
  by the dataset's own `gpt4_pair` judgments, so our judges' inconsistency rate is directly comparable
  to GPT-4's.
- `bayesian_model.py` design note: a one-parameter logistic (Rasch/IRT) model —
  `agree_ij ~ Bernoulli(sigmoid(ability_j - difficulty_i))` — over a generic (rater, item, agree)
  table, with `ability` given a `ZeroSumNormal` prior for identifiability (resolves the classic
  Rasch additive indeterminacy). Two data-prep paths feed it: `human_agreement_observations` (real
  MT-Bench human raters vs. the human-majority label — a genuine multi-rater validation on real
  data, addressing the earlier idea of sanity-checking the model before ever calling an LLM judge)
  and `judge_agreement_observations` (the actual project goal, from `run_judges` output). Also
  added `human_raters` to `load_mt_bench_items`'s output (parallel to `human_votes`) since the
  model needs per-vote rater identity, which wasn't being kept before.
- **Environment gotcha, fixed:** this sandbox has no C compiler / Python dev headers
  (`python3.13-dev` not installed, and no passwordless sudo to install it), so PyTensor's default
  JIT-compiled backend fails outright. `bayesian_model.fit()` now catches that failure and retries
  once with `pytensor.config.cxx = ""` (pure-Python execution — slower, but no system packages
  needed). Confirmed working: both real MCMC recovery tests pass in ~10s combined. If you later
  install `python3.13-dev` (`sudo apt install python3.13-dev`), the fast compiled path will be used
  automatically and this fallback won't trigger.
- `causal_experiment.py` design note: chose a **within-item (paired/crossed) design** over
  between-item randomization — every item is judged under every variant, since a stateless LLM
  judge call has no carryover/learning effect for randomization to guard against, and pairing on
  item removes item-difficulty variance from the comparison (more power at the same sample size).
  Reported honestly as a controlled paired comparison, not a between-subjects RCT. Effect size +
  CI come from an item-level (block) bootstrap; significance reuses `classical_stats.mcnemars_test`
  (correct existing tool for paired binary outcomes) rather than reimplementing it.
- `classical_stats.py` design note: our judges emit a discrete verdict (model_a/model_b/tie), not a
  numeric confidence, so the originally-planned `calibration_curve(judge_scores, human_labels)` doesn't
  have real inputs to run on yet — it's implemented as a generic, well-tested reliability-diagram
  primitive (confidence-in-[0,1] vs. observed accuracy) for reuse if a future judge variant elicits a
  confidence score. In its place, `accuracy_by_human_agreement` uses `human_agreement` (from
  `load_mt_bench_items`) as an item-difficulty proxy and checks whether judge accuracy tracks it — the
  categorical analogue that actually runs on current data. `evaluate_judge` ties kappa/confusion/accuracy
  together for one judge config's results.
- Bayesian model: PyMC is the default; installed cleanly (v6.3.1) on Python 3.13 in this environment, so
  no NumPyro fallback needed.
- Verified `load_mt_bench_items(n_items=200)`: all 200 sampled battles have >=2 human votes (761
  multi-vote battles exist, more than enough — no need to fall back to single-vote battles at this
  sample size). `human_agreement` ranges 0.5-1.0 as expected (a 2/1/1 four-vote split is the lowest
  non-tied plurality possible). Only ~50-56% of sampled battles have a `gpt4_pair_winner` — the two
  splits don't fully overlap, so any GPT-4-judge-vs-human comparison should filter to non-null rows.
