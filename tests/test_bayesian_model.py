"""Tests for `llm_judge_audit.bayesian_model`.

Data-prep functions and `build_model`'s validation are pure/fast. The two `fit()`-based tests
actually run NUTS sampling (small draw counts, kept deliberately fast) and check that the model
*recovers the right ordering* of well-separated true rater-ability / item-difficulty levels —
that's a more robust correctness check than a tight numeric tolerance would be against noisy MCMC
output, while still being a real end-to-end run, not a mock.
"""

import numpy as np
import pandas as pd
import pytest

from llm_judge_audit.bayesian_model import (
    build_model,
    fit,
    human_agreement_observations,
    judge_agreement_observations,
    summarize,
)


# --- data prep -------------------------------------------------------------------------------


def test_human_agreement_observations_filters_and_computes_agree():
    items = pd.DataFrame(
        {
            "question_id": [1, 2],
            "model_a": ["m1", "m1"],
            "model_b": ["m2", "m2"],
            "human_raters": [["e1", "e2"], ["e3"]],
            "human_votes": [["model_a", "model_b"], ["model_a"]],
            "human_n_votes": [2, 1],
            "human_winner": ["model_a", "model_a"],
        }
    )

    obs = human_agreement_observations(items, min_votes=2)

    assert len(obs) == 2  # question_id=2 has only 1 vote, filtered out by min_votes
    assert set(obs["rater"]) == {"e1", "e2"}
    e1_row = obs[obs["rater"] == "e1"].iloc[0]
    e2_row = obs[obs["rater"] == "e2"].iloc[0]
    assert e1_row["agree"] == 1  # e1 voted model_a, matching human_winner
    assert e2_row["agree"] == 0  # e2 voted model_b, not matching


def test_judge_agreement_observations():
    results = pd.DataFrame(
        {
            "question_id": [1, 2],
            "model_a": ["m1", "m1"],
            "model_b": ["m2", "m2"],
            "judge_name": ["claude-direct", "claude-direct"],
            "winner": ["model_a", "model_b"],
            "human_winner": ["model_a", "model_a"],
        }
    )

    obs = judge_agreement_observations(results)

    assert obs["item"].tolist() == ["1:m1:m2", "2:m1:m2"]
    assert obs["rater"].tolist() == ["claude-direct", "claude-direct"]
    assert obs["agree"].tolist() == [1, 0]


# --- build_model validation / structure (fast, no sampling) ----------------------------------


def test_build_model_raises_with_too_few_raters():
    obs = pd.DataFrame({"rater": ["r0", "r0"], "item": ["i0", "i1"], "agree": [1, 0]})
    with pytest.raises(ValueError):
        build_model(obs)


def test_build_model_raises_with_too_few_items():
    obs = pd.DataFrame({"rater": ["r0", "r1"], "item": ["i0", "i0"], "agree": [1, 0]})
    with pytest.raises(ValueError):
        build_model(obs)


def test_build_model_has_expected_coords_and_variables():
    obs = pd.DataFrame(
        {
            "rater": ["r0", "r0", "r1", "r1"],
            "item": ["i0", "i1", "i0", "i1"],
            "agree": [1, 0, 1, 1],
        }
    )
    model = build_model(obs)

    assert set(model.coords["rater"]) == {"r0", "r1"}
    assert set(model.coords["item"]) == {"i0", "i1"}
    free_rv_names = {rv.name for rv in model.free_RVs}
    assert {"sigma_ability", "sigma_difficulty", "ability", "difficulty"}.issubset(free_rv_names)


# --- fit / summarize: real (small) MCMC runs, checked by recovered ordering ------------------


def test_model_recovers_rater_ability_ordering():
    rng = np.random.default_rng(0)
    true_p = {"reliable": 0.95, "middling": 0.65, "unreliable": 0.15}
    n_items = 20

    rows = [
        {"rater": rater, "item": f"item{i}", "agree": rng.binomial(1, p)}
        for rater, p in true_p.items()
        for i in range(n_items)
    ]
    obs = pd.DataFrame(rows)

    model = build_model(obs)
    idata = fit(model, draws=400, tune=800, chains=2, target_accept=0.95, seed=0)
    summary = summarize(idata)

    ability = {rater: summary.loc[f"ability[{rater}]", "mean"] for rater in true_p}
    assert ability["reliable"] > ability["middling"] > ability["unreliable"]


def test_model_recovers_item_difficulty_ordering():
    rng = np.random.default_rng(1)
    true_ability = {"rater_a": 1.0, "rater_b": -1.0}
    true_difficulty = {"easy": -1.5, "hard": 1.5}
    n_reps = 15

    rows = []
    for rater, a in true_ability.items():
        for level, d in true_difficulty.items():
            p = 1 / (1 + np.exp(-(a - d)))
            for k in range(n_reps):
                rows.append({"rater": rater, "item": f"{level}_{k}", "agree": rng.binomial(1, p)})
    obs = pd.DataFrame(rows)

    model = build_model(obs)
    idata = fit(model, draws=400, tune=800, chains=2, target_accept=0.95, seed=1)
    summary = summarize(idata)

    easy_mean = summary.loc[[f"difficulty[easy_{k}]" for k in range(n_reps)], "mean"].mean()
    hard_mean = summary.loc[[f"difficulty[hard_{k}]" for k in range(n_reps)], "mean"].mean()
    assert hard_mean > easy_mean
