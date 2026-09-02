"""Plotting and dashboard data prep.

Turns the outputs of `classical_stats`, `causal_experiment`, and `bayesian_model` into figures for
local exploration (matplotlib) and into a JSON-safe payload for the published dashboard (see
`docs/index.html`, published via GitHub Pages). Built directly against those modules' real return
shapes:
    - `plot_calibration_curve` <- `classical_stats.calibration_curve`'s DataFrame
    - `plot_confusion_matrix` <- `classical_stats.confusion_matrix_report`'s DataFrame
    - `plot_posterior_intervals` <- `bayesian_model.summarize`'s DataFrame
    - `to_dashboard_json` <- `classical_stats.evaluate_judge`, `causal_experiment.estimate_*_effect`,
      and `bayesian_model.summarize` results, combined into one payload.

Note: none of these are called from a production code path (pipeline.py / run_pipeline.py /
generate_fake_results.py) — the dashboard's charts are built directly from to_dashboard_json's
payload. Kept here, tested, and ready for local/ad-hoc plotting if needed.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # headless-safe: no display assumed (dev sandbox, CI, etc.)

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_calibration_curve(
    curve_data: pd.DataFrame, ax: matplotlib.axes.Axes | None = None
) -> matplotlib.figure.Figure:
    """Reliability diagram from `classical_stats.calibration_curve`'s output: observed accuracy
    vs. mean predicted confidence per bin, marker size proportional to bin count, with the
    perfect-calibration diagonal for reference."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(5, 5))

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    sizes = 200 * curve_data["n"] / curve_data["n"].max()
    ax.scatter(curve_data["mean_confidence"], curve_data["observed_accuracy"], s=sizes, alpha=0.7)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Calibration")
    ax.legend()
    return fig


def plot_confusion_matrix(
    matrix: pd.DataFrame, ax: matplotlib.axes.Axes | None = None
) -> matplotlib.figure.Figure:
    """Heatmap of a `classical_stats.confusion_matrix_report` confusion table (rows = judge label,
    columns = human label), annotated with counts and, if present, `matrix.attrs["accuracy"]`."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(5, 5))

    im = ax.imshow(matrix.to_numpy(), cmap="Blues")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Human label")
    ax.set_ylabel("Judge label")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix.iat[i, j]), ha="center", va="center")

    title = "Confusion matrix"
    accuracy = matrix.attrs.get("accuracy")
    if accuracy is not None:
        title += f" (accuracy={accuracy:.2f})"
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    return fig


def _find_ci_columns(summary: pd.DataFrame) -> tuple[str, str]:
    """Locate the credible-interval lower/upper bound columns in an `az.summary` DataFrame.

    Their name depends on the `ci_prob` passed to `bayesian_model.summarize` (e.g. "eti94_lb" /
    "eti94_ub" for ci_prob=0.94), so this looks them up by suffix rather than hardcoding a name.
    """
    lb_cols = [c for c in summary.columns if c.endswith("_lb")]
    ub_cols = [c for c in summary.columns if c.endswith("_ub")]
    if len(lb_cols) != 1 or len(ub_cols) != 1:
        raise ValueError(
            f"expected exactly one lower/upper credible-interval column, found {lb_cols} / {ub_cols}"
        )
    return lb_cols[0], ub_cols[0]


def plot_posterior_intervals(
    summary: pd.DataFrame, ax: matplotlib.axes.Axes | None = None
) -> matplotlib.figure.Figure:
    """Forest plot of posterior mean +/- credible interval for each row of a
    `bayesian_model.summarize` result (pass the ability rows, the difficulty rows, or both)."""
    lb_col, ub_col = _find_ci_columns(summary)
    ordered = summary.sort_values("mean")

    fig, ax = (
        (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6, max(2, 0.3 * len(ordered))))
    )

    y = np.arange(len(ordered))
    lower_err = ordered["mean"] - ordered[lb_col]
    upper_err = ordered[ub_col] - ordered["mean"]
    ax.errorbar(ordered["mean"], y, xerr=[lower_err, upper_err], fmt="o", capsize=3)
    ax.axvline(0, linestyle="--", color="gray", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered.index)
    ax.set_xlabel("Posterior mean (credible interval)")
    ax.set_title("Posterior intervals")
    return fig


def _df_to_jsonable(df: pd.DataFrame, orient: str = "records"):
    """Round-trip a DataFrame through pandas' own JSON encoder to guarantee JSON-safe native
    Python types — numpy scalar dtypes (e.g. int64) aren't all directly JSON-serializable."""
    return json.loads(df.to_json(orient=orient))


def to_dashboard_json(
    judge_summaries: dict[str, dict] | None = None,
    causal_effects: dict[str, dict] | None = None,
    bayesian_summary: pd.DataFrame | None = None,
    difficulty_breakdown: dict[str, pd.DataFrame] | None = None,
) -> dict:
    """Assemble a JSON-serializable payload for the Artifact results dashboard.

    Args:
        judge_summaries: judge config name -> `classical_stats.evaluate_judge` result.
        causal_effects: comparison label (e.g. "cot_vs_direct") ->
            `causal_experiment.estimate_paired_effect` / `estimate_variant_effect` result.
        bayesian_summary: `bayesian_model.summarize` result.
        difficulty_breakdown: judge config name -> `classical_stats.accuracy_by_human_agreement`
            result.

    Returns:
        A plain dict of JSON-safe primitives, ready for `json.dumps`.
    """
    payload: dict = {}

    if judge_summaries:
        payload["judges"] = {
            name: {
                "n_items": int(s["n_items"]),
                "kappa": float(s["kappa"]),
                "accuracy": float(s["accuracy"]),
                "inconsistency_rate": float(s["inconsistency_rate"]),
                "unparseable_rate": float(s["unparseable_rate"]),
                "error_rate": float(s["error_rate"]),
                "confusion": {
                    "index": list(s["confusion"].index),
                    "columns": list(s["confusion"].columns),
                    "data": s["confusion"].to_numpy().tolist(),
                },
            }
            for name, s in judge_summaries.items()
        }

    if causal_effects:
        payload["causal_effects"] = {
            label: {k: (int(v) if k == "n_items" else float(v)) for k, v in effect.items()}
            for label, effect in causal_effects.items()
        }

    if bayesian_summary is not None:
        payload["bayesian_summary"] = _df_to_jsonable(
            bayesian_summary.reset_index(names="parameter")
        )

    if difficulty_breakdown:
        payload["difficulty_breakdown"] = {
            name: _df_to_jsonable(df) for name, df in difficulty_breakdown.items()
        }

    return payload
