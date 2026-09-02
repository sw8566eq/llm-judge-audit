"""Hierarchical Bayesian model of rater reliability and item difficulty.

A one-parameter logistic (Rasch/IRT-style) model, implemented in PyMC, over a generic
(rater, item, agree) observation table:

    agree_ij ~ Bernoulli(sigmoid(ability_j - difficulty_i))
    ability_j ~ ZeroSumNormal(sigma_ability)      # per-rater reliability, mean-zero for identifiability
    difficulty_i ~ Normal(0, sigma_difficulty)     # per-item difficulty
    sigma_ability, sigma_difficulty ~ HalfNormal(1)

`agree_ij` = 1 if rater j's individual verdict on item i matches the item's reference label (the
human-consensus `human_winner`), 0 otherwise. `ability_j` is then how reliably rater j tracks that
reference, and `difficulty_i` is how hard item i is to call correctly — full posteriors and
credible intervals for both, rather than a single point-estimate accuracy number.

This is deliberately generic (it doesn't care whether "rater" is a human annotator or an LLM
judge config), so it's used two ways:
    - `human_agreement_observations` fits it on MT-Bench's *human* raters against the
      human-majority label — a genuine multi-rater validation on real data, checkable before ever
      calling an LLM judge (are known-careful annotators estimated as more reliable? do
      known-ambiguous battles come out as more difficult?).
    - `judge_agreement_observations` fits it on `judges.run_judges` output — the actual project
      goal: per-judge-config reliability and per-battle difficulty, estimated jointly.

Identifiability: a Rasch model has an additive indeterminacy (shifting every ability and every
difficulty by the same constant leaves `ability_j - difficulty_i` unchanged). `ability` is given a
`ZeroSumNormal` prior (mean pinned exactly to 0 across raters) to anchor it; `difficulty` is then
identified relative to that anchor.

Environment note: PyTensor (PyMC's backend) normally JIT-compiles a small C extension for speed.
On a machine without a C compiler / Python dev headers (e.g. `python3-dev` not installed), that
compilation fails outright instead of falling back gracefully. `fit()` catches that failure and
retries once in PyTensor's pure-Python execution mode (slower, but requires no system packages).
"""

from __future__ import annotations

import warnings

import arviz as az
import pandas as pd
import pymc as pm
import pytensor
from pytensor.link.c.exceptions import CompileError


def human_agreement_observations(items: pd.DataFrame, min_votes: int = 2) -> pd.DataFrame:
    """Expand `judges.load_mt_bench_items`'s per-item (human_raters, human_votes) lists into one
    row per (individual human vote, item), for `build_model`.

    `agree` is whether that individual vote matches the item's already-computed majority
    `human_winner` label. Only items with >= `min_votes` human votes are included: with exactly
    one vote, that vote *is* the majority by construction and would trivially always "agree",
    which would just dilute the signal rather than tell us anything about rater reliability.

    Returns a DataFrame with columns: item, rater, agree.
    """
    multi = items[items["human_n_votes"] >= min_votes]
    rows = [
        {
            "item": f"{row.question_id}:{row.model_a}:{row.model_b}",
            "rater": rater,
            "agree": int(vote == row.human_winner),
        }
        for row in multi.itertuples()
        for rater, vote in zip(row.human_raters, row.human_votes)
    ]
    return pd.DataFrame(rows, columns=["item", "rater", "agree"])


def judge_agreement_observations(
    results: pd.DataFrame,
    judge_col: str = "winner",
    human_col: str = "human_winner",
    rater_col: str = "judge_name",
) -> pd.DataFrame:
    """Turn `judges.run_judges` output into a (rater, item, agree) table for `build_model`.

    `rater` is the judge config name (`rater_col`, default "judge_name") — so ability is estimated
    per judge configuration (i.e. per model x prompt-variant combination under test), and `agree`
    is whether that config's verdict matched `human_col` (ground truth) on that item.
    """
    out = pd.DataFrame(
        {
            "item": results["question_id"].astype(str)
            + ":"
            + results["model_a"]
            + ":"
            + results["model_b"],
            "rater": results[rater_col],
            "agree": (results[judge_col] == results[human_col]).astype(int),
        }
    )
    return out


def build_model(
    observations: pd.DataFrame,
    rater_col: str = "rater",
    item_col: str = "item",
    outcome_col: str = "agree",
) -> pm.Model:
    """Construct the hierarchical rater/item model from a tidy (rater, item, agree) table.

    Args:
        observations: one row per (rater, item) observation, e.g. from
            `human_agreement_observations` or `judge_agreement_observations`.
        rater_col, item_col, outcome_col: column names in `observations`.

    Returns:
        An unfit `pm.Model` with coords "rater"/"item" (so the fitted trace's `ability`/
        `difficulty` variables are indexed by the actual rater/item labels, not bare integers).
    """
    raters = observations[rater_col].astype("category")
    items = observations[item_col].astype("category")
    if len(raters.cat.categories) < 2:
        raise ValueError("need at least 2 distinct raters to fit ability (got fewer)")
    if len(items.cat.categories) < 2:
        raise ValueError("need at least 2 distinct items to fit difficulty (got fewer)")

    coords = {"rater": raters.cat.categories.tolist(), "item": items.cat.categories.tolist()}
    with pm.Model(coords=coords) as model:
        sigma_ability = pm.HalfNormal("sigma_ability", sigma=1.0)
        sigma_difficulty = pm.HalfNormal("sigma_difficulty", sigma=1.0)

        ability = pm.ZeroSumNormal("ability", sigma=sigma_ability, dims="rater")
        difficulty = pm.Normal("difficulty", mu=0.0, sigma=sigma_difficulty, dims="item")

        theta = ability[raters.cat.codes.to_numpy()] - difficulty[items.cat.codes.to_numpy()]
        pm.Bernoulli("agree_obs", logit_p=theta, observed=observations[outcome_col].to_numpy())

    return model


def fit(
    model: pm.Model,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    seed: int = 42,
) -> az.InferenceData:
    """Run NUTS sampling and return the trace.

    See the module docstring's environment note: if PyTensor's C compilation fails, this retries
    once with `pytensor.config.cxx = ""` (pure-Python execution) rather than aborting the run.
    Only `CompileError` triggers the fallback — a genuine sampling problem (bad priors, a
    degenerate model) must surface as itself, not get misdiagnosed as a missing compiler and
    silently re-sampled from scratch a second time.
    """
    sample_kwargs = dict(
        draws=draws, tune=tune, chains=chains, target_accept=target_accept,
        random_seed=seed, progressbar=False,
    )
    with model:
        try:
            return pm.sample(**sample_kwargs)
        except CompileError as exc:
            if pytensor.config.cxx == "":
                raise  # already in the fallback mode; this failure is something else
            warnings.warn(
                "PyTensor C compilation failed (no working C compiler / Python dev headers?); "
                f"retrying in pure-Python mode, which is slower. Original error: {exc}",
                stacklevel=2,
            )
            pytensor.config.cxx = ""
            return pm.sample(**sample_kwargs)


def summarize(
    idata: az.InferenceData, var_names: tuple[str, ...] = ("ability", "difficulty"), ci_prob: float = 0.94
) -> pd.DataFrame:
    """Posterior summaries (mean, sd, credible interval, R-hat/ESS diagnostics) per rater and item."""
    return az.summary(idata, var_names=list(var_names), ci_prob=ci_prob)
