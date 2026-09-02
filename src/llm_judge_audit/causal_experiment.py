"""Causal experiment on judge-prompt design.

Estimates the causal effect of judge-prompt variant (e.g. "direct" vs. "cot") on how often a judge
agrees with human ground truth, using a **within-item (paired/crossed) design**: every item is
judged under every variant being compared, rather than randomly splitting items between arms.

Why paired, not between-item randomization: applying a second variant to an item a judge has
already scored doesn't contaminate the first result — each `judges.run_judges` call is a stateless
API request with no memory of other calls, so there's no carryover/learning effect for
randomization to guard against. Pairing on item instead removes item-to-item difficulty variance
(some battles are just harder to call than others) from the comparison entirely, which gives a
much more precise effect estimate at the same sample size than a between-item split would. The
tradeoff is honesty about what this design supports: it's a controlled paired comparison, not a
randomized-controlled trial in the classic between-subjects sense — reported as such rather than
overclaiming.

The outcome is binary (was the judge's verdict correct against `human_winner`?) and paired by
construction, so:
    - the point estimate + bootstrap confidence interval come from an item-level (block) bootstrap
      — resampling *items*, not individual judge calls, so the resampling respects the pairing;
    - `classical_stats.mcnemars_test` (already implemented, and exactly the right classical test
      for paired binary outcomes) is reused for the significance test, rather than re-implementing
      it here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .classical_stats import mcnemars_test

DEFAULT_ITEM_KEY = ("question_id", "model_a", "model_b", "judge_model")


def add_correctness(
    results: pd.DataFrame, judge_col: str = "winner", human_col: str = "human_winner"
) -> pd.DataFrame:
    """Add a boolean `correct` column: whether the judge's verdict matches human ground truth."""
    out = results.copy()
    out["correct"] = out[judge_col] == out[human_col]
    return out


def paired_design(
    results: pd.DataFrame,
    treatment_col: str,
    outcome_col: str = "correct",
    item_key: tuple[str, ...] = DEFAULT_ITEM_KEY,
) -> pd.DataFrame:
    """Pivot long-format judge results (one row per item x treatment level) into a paired wide
    table: one row per item (`item_key`), one column per level of `treatment_col`.

    `results` must have exactly one row per (item, treatment level) — `pivot` raises a ValueError
    if there are duplicates, since silently aggregating them would hide a data problem rather than
    a modeling choice.

    The default `item_key` includes `judge_model`, so comparing two variants of the *same* model is
    correctly paired even if `results` also contains other models: if you're comparing configs that
    differ in more than `treatment_col` (e.g. different models *and* different prompt variants),
    narrow `item_key`/pre-filter `results` accordingly, or the effect will conflate both differences.
    """
    return results.pivot(index=list(item_key), columns=treatment_col, values=outcome_col)


def estimate_paired_effect(
    wide: pd.DataFrame,
    baseline: str,
    treatment: str,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    """Estimate the causal effect of `treatment` vs. `baseline` on a paired binary outcome.

    Args:
        wide: paired outcome table, e.g. from `paired_design` — one row per item, with `baseline`
            and `treatment` as (boolean-valued) columns.
        baseline: column name of the baseline/control arm.
        treatment: column name of the treatment arm.
        n_bootstrap: number of item-level bootstrap resamples for the confidence interval.
        seed: random seed, for reproducibility.
        ci: confidence level for the interval (e.g. 0.95 for a 95% CI).

    Returns:
        dict with n_items, baseline_accuracy, treatment_accuracy, effect (treatment - baseline),
        ci_lower, ci_upper, ci_level, and the paired McNemar's `mcnemar_statistic`/`mcnemar_pvalue`
        for the same comparison.
    """
    paired = wide[[baseline, treatment]].dropna()
    n = len(paired)
    if n == 0:
        raise ValueError("no items have outcomes for both `baseline` and `treatment`")

    baseline_vals = paired[baseline].to_numpy(dtype=bool)
    treatment_vals = paired[treatment].to_numpy(dtype=bool)

    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_diff = treatment_vals[boot_idx].mean(axis=1) - baseline_vals[boot_idx].mean(axis=1)

    alpha = 1 - ci
    ci_lower, ci_upper = np.quantile(boot_diff, [alpha / 2, 1 - alpha / 2])

    mcnemar = mcnemars_test(baseline_vals, treatment_vals)

    return {
        "n_items": n,
        "baseline_accuracy": float(baseline_vals.mean()),
        "treatment_accuracy": float(treatment_vals.mean()),
        "effect": float(treatment_vals.mean() - baseline_vals.mean()),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "ci_level": ci,
        "mcnemar_statistic": mcnemar["statistic"],
        "mcnemar_pvalue": mcnemar["pvalue"],
    }


def estimate_variant_effect(
    results: pd.DataFrame,
    baseline: str,
    treatment: str,
    treatment_col: str = "variant",
    judge_col: str = "winner",
    human_col: str = "human_winner",
    item_key: tuple[str, ...] = DEFAULT_ITEM_KEY,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    """End-to-end: from raw `judges.run_judges` long-format output, estimate the causal effect of
    one judge-prompt variant vs. another on accuracy against human ground truth, blocked on item.

    Equivalent to `estimate_paired_effect(paired_design(add_correctness(results, ...), ...), ...)`.
    """
    with_correctness = add_correctness(results, judge_col=judge_col, human_col=human_col)
    wide = paired_design(with_correctness, treatment_col=treatment_col, item_key=item_key)
    return estimate_paired_effect(
        wide, baseline=baseline, treatment=treatment, n_bootstrap=n_bootstrap, seed=seed, ci=ci
    )
