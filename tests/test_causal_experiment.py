"""Tests for `llm_judge_audit.causal_experiment`."""

import numpy as np
import pandas as pd
import pytest

from llm_judge_audit.causal_experiment import (
    add_correctness,
    estimate_paired_effect,
    estimate_variant_effect,
    paired_design,
)


def test_add_correctness():
    df = pd.DataFrame({"winner": ["model_a", "model_b"], "human_winner": ["model_a", "model_a"]})
    out = add_correctness(df)
    assert out["correct"].tolist() == [True, False]


def test_paired_design_raises_on_duplicate_item_treatment_rows():
    df = pd.DataFrame(
        {
            "question_id": [1, 1],
            "model_a": ["m1", "m1"],
            "model_b": ["m2", "m2"],
            "judge_model": ["claude-x", "claude-x"],
            "variant": ["direct", "direct"],  # two rows for the same (item, treatment level)
            "correct": [True, False],
        }
    )
    with pytest.raises(ValueError):
        paired_design(df, treatment_col="variant")


def test_estimate_paired_effect_detects_positive_effect():
    # baseline correct on the first 20/40 items; treatment correct on the first 32/40 — so the
    # only discordant direction is "treatment right, baseline wrong" (12 items), never the reverse.
    baseline = np.array([True] * 20 + [False] * 20)
    treatment = np.array([True] * 32 + [False] * 8)
    wide = pd.DataFrame({"direct": baseline, "cot": treatment})

    result = estimate_paired_effect(wide, baseline="direct", treatment="cot", n_bootstrap=2000, seed=0)

    assert result["n_items"] == 40
    assert result["baseline_accuracy"] == pytest.approx(0.5)
    assert result["treatment_accuracy"] == pytest.approx(0.8)
    assert result["effect"] == pytest.approx(0.3)
    assert result["ci_lower"] > 0  # effect is unambiguously positive; CI shouldn't cross zero
    assert result["mcnemar_pvalue"] < 0.05


def test_estimate_paired_effect_no_difference():
    same = np.array([True] * 15 + [False] * 5)
    wide = pd.DataFrame({"direct": same, "cot": same})

    result = estimate_paired_effect(wide, baseline="direct", treatment="cot", n_bootstrap=2000, seed=0)

    assert result["effect"] == pytest.approx(0.0)
    assert result["ci_lower"] <= 0.0 <= result["ci_upper"]
    assert result["mcnemar_pvalue"] == pytest.approx(1.0)


def test_estimate_paired_effect_drops_items_missing_either_arm():
    wide = pd.DataFrame(
        {
            "direct": [True, False, None, True],
            "cot": [True, True, True, None],
        }
    )
    result = estimate_paired_effect(wide, baseline="direct", treatment="cot", n_bootstrap=500, seed=0)
    assert result["n_items"] == 2  # only the first two rows have both arms present


def test_estimate_paired_effect_raises_when_no_overlap():
    wide = pd.DataFrame({"direct": [True, None], "cot": [None, True]})
    with pytest.raises(ValueError):
        estimate_paired_effect(wide, baseline="direct", treatment="cot")


def test_estimate_variant_effect_end_to_end():
    rows = []
    for qid in range(1, 11):
        human = "model_a" if qid % 2 == 0 else "model_b"
        flipped = "model_b" if human == "model_a" else "model_a"
        direct_winner = human if qid <= 5 else flipped  # 5/10 correct
        cot_winner = human if qid != 10 else flipped  # 9/10 correct
        for variant, winner in [("direct", direct_winner), ("cot", cot_winner)]:
            rows.append(
                {
                    "question_id": qid,
                    "model_a": "m1",
                    "model_b": "m2",
                    "judge_model": "claude-x",
                    "variant": variant,
                    "winner": winner,
                    "human_winner": human,
                }
            )
    results = pd.DataFrame(rows)

    effect = estimate_variant_effect(results, baseline="direct", treatment="cot", n_bootstrap=1000, seed=1)

    assert effect["n_items"] == 10
    assert effect["baseline_accuracy"] == pytest.approx(0.5)
    assert effect["treatment_accuracy"] == pytest.approx(0.9)
    assert effect["effect"] == pytest.approx(0.4)
