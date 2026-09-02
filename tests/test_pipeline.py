"""Tests for `llm_judge_audit.pipeline.analyze_results`."""

import pandas as pd

from llm_judge_audit.judges import JudgeConfig
from llm_judge_audit.pipeline import analyze_results, print_analysis_summary


def _synthetic_results() -> pd.DataFrame:
    # 2 configs of the same model (direct/cot) x 8 items, so both the causal-effect path and the
    # Bayesian model (>=2 raters, >=2 items) are exercised.
    rows = []
    for qid in range(1, 9):
        human = "model_a" if qid % 2 == 0 else "model_b"
        flipped = "model_b" if human == "model_a" else "model_a"
        direct_winner = human if qid <= 5 else flipped  # 5/8 correct
        cot_winner = human if qid != 8 else flipped  # 7/8 correct
        for variant, winner in [("direct", direct_winner), ("cot", cot_winner)]:
            rows.append(
                {
                    "question_id": qid,
                    "model_a": "m1",
                    "model_b": "m2",
                    "judge_name": f"claude-x-{variant}",
                    "judge_model": "claude-x",
                    "variant": variant,
                    "winner": winner,
                    "human_winner": human,
                }
            )
    return pd.DataFrame(rows)


def _synthetic_items() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "question_id": range(1, 9),
            "model_a": ["m1"] * 8,
            "model_b": ["m2"] * 8,
            "human_agreement": [1.0, 1.0, 0.67, 0.67, 0.5, 0.5, 1.0, 1.0],
        }
    )


def test_analyze_results_end_to_end():
    results = _synthetic_results()
    items = _synthetic_items()
    judge_configs = [
        JudgeConfig(name="claude-x-direct", model="claude-x", variant="direct"),
        JudgeConfig(name="claude-x-cot", model="claude-x", variant="cot"),
    ]

    analysis = analyze_results(results, judge_configs, items=items, seed=0)

    assert set(analysis["judge_summaries"]) == {"claude-x-direct", "claude-x-cot"}
    assert analysis["judge_summaries"]["claude-x-direct"]["accuracy"] == 5 / 8
    assert analysis["judge_summaries"]["claude-x-cot"]["accuracy"] == 7 / 8

    assert "claude-x: cot_vs_direct" in analysis["causal_effects"]
    effect = analysis["causal_effects"]["claude-x: cot_vs_direct"]
    assert effect["effect"] > 0  # cot outperforms direct in this synthetic data

    assert analysis["bayesian_summary"] is not None
    assert "ability[claude-x-direct]" in analysis["bayesian_summary"].index

    assert set(analysis["difficulty_breakdown"]) == {"claude-x-direct", "claude-x-cot"}

    # dashboard payload must be assembled from all four pieces and be JSON-safe
    import json

    json.dumps(analysis["dashboard_payload"])
    assert set(analysis["dashboard_payload"]) == {
        "judges",
        "causal_effects",
        "bayesian_summary",
        "difficulty_breakdown",
    }


def test_analyze_results_skips_causal_and_bayesian_when_underpowered():
    # a single judge config: no direct/cot pair for a causal effect, and only 1 rater for the
    # Bayesian model, which needs >= 2.
    results = _synthetic_results()
    results = results[results["variant"] == "direct"]
    judge_configs = [JudgeConfig(name="claude-x-direct", model="claude-x", variant="direct")]

    analysis = analyze_results(results, judge_configs, seed=0)

    assert analysis["causal_effects"] == {}
    assert analysis["bayesian_summary"] is None
    assert analysis["difficulty_breakdown"] is None  # no `items` passed


def test_print_analysis_summary_prints_accuracy_and_effect(capsys):
    results = _synthetic_results()
    judge_configs = [
        JudgeConfig(name="claude-x-direct", model="claude-x", variant="direct"),
        JudgeConfig(name="claude-x-cot", model="claude-x", variant="cot"),
    ]
    analysis = analyze_results(results, judge_configs, seed=0)

    print_analysis_summary(analysis)

    out = capsys.readouterr().out
    assert "claude-x-direct" in out
    assert "accuracy=" in out
    assert "McNemar p=" in out
