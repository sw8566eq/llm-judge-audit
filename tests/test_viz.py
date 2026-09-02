"""Tests for `llm_judge_audit.viz`.

Plotting functions are checked for "runs without error and returns a Figure" rather than pixel
content — that's the meaningful contract for helpers whose job is to not crash on real data shapes
from the other modules. Figures are closed after each check to avoid leaking matplotlib state.
"""

import json

import matplotlib.figure
import pandas as pd
import pytest

from llm_judge_audit.viz import (
    _find_ci_columns,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_posterior_intervals,
    to_dashboard_json,
)


def test_plot_calibration_curve_returns_figure():
    curve = pd.DataFrame(
        {
            "bin_lower": [0.0, 0.5],
            "bin_upper": [0.5, 1.0],
            "n": [10, 20],
            "mean_confidence": [0.3, 0.8],
            "observed_accuracy": [0.4, 0.75],
            "gap": [-0.1, 0.05],
        }
    )
    fig = plot_calibration_curve(curve)
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_plot_confusion_matrix_returns_figure_and_uses_accuracy():
    matrix = pd.DataFrame(
        [[8, 2], [1, 9]], index=["model_a", "model_b"], columns=["model_a", "model_b"]
    )
    matrix.attrs["accuracy"] = 0.85

    fig = plot_confusion_matrix(matrix)
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
        ax = fig.axes[0]
        assert "0.85" in ax.get_title()
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_find_ci_columns_locates_suffix_pair():
    summary = pd.DataFrame({"mean": [0.1], "eti94_lb": [-0.2], "eti94_ub": [0.4]})
    assert _find_ci_columns(summary) == ("eti94_lb", "eti94_ub")


def test_find_ci_columns_raises_when_absent():
    summary = pd.DataFrame({"mean": [0.1], "sd": [0.2]})
    with pytest.raises(ValueError):
        _find_ci_columns(summary)


def test_plot_posterior_intervals_returns_figure():
    summary = pd.DataFrame(
        {
            "mean": [0.5, -0.3, 1.1],
            "eti94_lb": [0.1, -0.9, 0.4],
            "eti94_ub": [0.9, 0.3, 1.8],
        },
        index=["ability[a]", "ability[b]", "ability[c]"],
    )
    fig = plot_posterior_intervals(summary)
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_to_dashboard_json_is_json_serializable_and_shaped_correctly():
    confusion = pd.DataFrame(
        [[8, 2], [1, 9]], index=["model_a", "model_b"], columns=["model_a", "model_b"]
    )
    confusion.attrs["accuracy"] = 0.85

    judge_summaries = {
        "claude-direct": {
            "n_items": 20,
            "kappa": 0.62,
            "confusion": confusion,
            "accuracy": 0.85,
            "inconsistency_rate": 0.05,
            "unparseable_rate": 0.0,
            "error_rate": 0.0,
        }
    }
    causal_effects = {
        "cot_vs_direct": {
            "n_items": 20,
            "baseline_accuracy": 0.7,
            "treatment_accuracy": 0.85,
            "effect": 0.15,
            "ci_lower": 0.02,
            "ci_upper": 0.28,
            "ci_level": 0.95,
            "mcnemar_statistic": 3.0,
            "mcnemar_pvalue": 0.04,
        }
    }
    bayesian_summary = pd.DataFrame(
        {"mean": [0.5], "eti94_lb": [0.1], "eti94_ub": [0.9]}, index=["ability[claude-direct]"]
    )

    payload = to_dashboard_json(
        judge_summaries=judge_summaries,
        causal_effects=causal_effects,
        bayesian_summary=bayesian_summary,
    )

    # must not raise — every value has to be a native JSON-safe type, not numpy/pandas objects
    serialized = json.dumps(payload)
    assert isinstance(serialized, str)

    assert payload["judges"]["claude-direct"]["n_items"] == 20
    assert payload["judges"]["claude-direct"]["confusion"]["data"] == [[8, 2], [1, 9]]
    assert payload["causal_effects"]["cot_vs_direct"]["effect"] == pytest.approx(0.15)
    assert payload["bayesian_summary"][0]["parameter"] == "ability[claude-direct]"


def test_to_dashboard_json_handles_missing_sections():
    assert to_dashboard_json() == {}


def test_to_dashboard_json_serializes_difficulty_breakdown():
    breakdown = pd.DataFrame(
        {"human_agreement": [0.6, 1.0], "n": [10, 15], "judge_accuracy": [0.5, 0.8]}
    )
    payload = to_dashboard_json(difficulty_breakdown={"claude-direct": breakdown})

    serialized = json.dumps(payload)
    assert isinstance(serialized, str)
    assert payload["difficulty_breakdown"]["claude-direct"][0]["n"] == 10
    assert payload["difficulty_breakdown"]["claude-direct"][1]["judge_accuracy"] == pytest.approx(0.8)
