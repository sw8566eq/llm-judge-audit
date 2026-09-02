"""Classical inter-rater reliability and calibration statistics.

Compares judge outputs against human ground-truth labels (and against each other) using
well-established classical statistics — the "can you communicate uncertainty clearly" layer of
the audit. Two layers of functions:

Generic, reusable primitives (tested against synthetic/textbook values, independent of this
project's specific data shape):
    - `cohens_kappa` — agreement between two raters' categorical labels.
    - `confusion_matrix_report` — confusion matrix + accuracy between two raters' labels.
    - `calibration_curve` — classic reliability-diagram binning of a numeric confidence/probability
      against observed correctness.
    - `mcnemars_test` — paired test for whether two raters/judges disagree systematically.

Project-specific glue, built directly on `judges.load_mt_bench_items` / `judges.run_judges` output:
    - `evaluate_judge` — kappa/confusion/accuracy of one judge config against `human_winner`.
    - `accuracy_by_human_agreement` — the categorical analogue of a calibration curve for this
      project: our judges emit a discrete verdict, not a numeric confidence, so there's nothing to
      feed `calibration_curve` directly. Instead, `human_agreement` (how unanimous the human raters
      were on an item, from `load_mt_bench_items`) is used as a proxy for item difficulty, and we
      check whether judge accuracy tracks it — i.e. is the judge in fact less reliable on the
      battles humans themselves found ambiguous?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar as _sm_mcnemar
from statsmodels.stats.inter_rater import cohens_kappa as _sm_cohens_kappa


def _confusion_table(labels_a, labels_b, categories: list | None = None) -> pd.DataFrame:
    """Square confusion table between two raters' labels, aligned to the same category order.

    `cohens_kappa` requires a square table with matching row/column categories, which
    `pd.crosstab` alone doesn't guarantee if one rater never uses a category the other does.
    """
    labels_a = pd.Series(labels_a).reset_index(drop=True)
    labels_b = pd.Series(labels_b).reset_index(drop=True)
    if categories is None:
        categories = sorted(set(labels_a) | set(labels_b))
    table = pd.crosstab(labels_a, labels_b)
    return table.reindex(index=categories, columns=categories, fill_value=0)


def cohens_kappa(judge_labels, human_labels, categories: list | None = None) -> float:
    """Cohen's kappa agreement between a judge's labels and human ground-truth labels.

    Delegates to statsmodels' `cohens_kappa` (simple/unweighted kappa) on the confusion table.
    """
    table = _confusion_table(judge_labels, human_labels, categories=categories)
    result = _sm_cohens_kappa(table.values, return_results=True)
    return float(result.kappa)


def confusion_matrix_report(judge_labels, human_labels, categories: list | None = None) -> pd.DataFrame:
    """Confusion matrix (rows = judge label, columns = human label) between judge and human labels.

    The raw accuracy (fraction of items where judge_labels == human_labels) is attached as
    `.attrs["accuracy"]` on the returned DataFrame.
    """
    table = _confusion_table(judge_labels, human_labels, categories=categories)
    judge_labels = pd.Series(judge_labels).reset_index(drop=True)
    human_labels = pd.Series(human_labels).reset_index(drop=True)
    table.attrs["accuracy"] = float((judge_labels == human_labels).mean())
    return table


def calibration_curve(confidences, correct, n_bins: int = 10) -> pd.DataFrame:
    """Classic reliability-diagram data: binned predicted confidence vs. observed accuracy.

    Args:
        confidences: predicted confidence/probability in [0, 1] for each item.
        correct: whether the prediction was actually correct (bool/0-1) for each item, same length.
        n_bins: number of equal-width bins spanning [0, 1].

    Returns:
        One row per non-empty bin: bin_lower, bin_upper, n, mean_confidence, observed_accuracy,
        gap (mean_confidence - observed_accuracy; positive means overconfident).
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")
    if np.any((confidences < 0) | (confidences > 1)):
        raise ValueError("confidences must be in [0, 1]")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(confidences, edges[1:-1], right=True), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        mean_conf = float(confidences[mask].mean())
        obs_acc = float(correct[mask].mean())
        rows.append(
            {
                "bin_lower": edges[b],
                "bin_upper": edges[b + 1],
                "n": n,
                "mean_confidence": mean_conf,
                "observed_accuracy": obs_acc,
                "gap": mean_conf - obs_acc,
            }
        )
    return pd.DataFrame(rows)


def mcnemars_test(a_correct, b_correct, exact: bool = True) -> dict:
    """Paired test for whether two judges disagree systematically on the same items.

    `a_correct`/`b_correct` are same-length, item-aligned boolean arrays (was judge A/B correct on
    item i?). Tests whether the discordant pairs (A right & B wrong, vs. A wrong & B right) are
    lopsided — i.e. whether one judge is significantly more accurate than the other on these items,
    not just different in aggregate accuracy.

    Returns a dict with `statistic`, `pvalue`, and the 2x2 discordant-pairs `table` (rows/cols:
    a_correct=[True, False], b_correct=[True, False] — e.g. `table[0][1]` is the "a_only" count:
    a correct, b wrong).
    """
    a_correct = np.asarray(a_correct, dtype=bool)
    b_correct = np.asarray(b_correct, dtype=bool)
    if len(a_correct) != len(b_correct):
        raise ValueError("a_correct and b_correct must be the same length")

    both = int(np.sum(a_correct & b_correct))
    a_only = int(np.sum(a_correct & ~b_correct))
    b_only = int(np.sum(~a_correct & b_correct))
    neither = int(np.sum(~a_correct & ~b_correct))
    table = np.array([[both, a_only], [b_only, neither]])

    result = _sm_mcnemar(table, exact=exact)
    return {"statistic": float(result.statistic), "pvalue": float(result.pvalue), "table": table}


def evaluate_judge(
    results: pd.DataFrame, judge_col: str = "winner", human_col: str = "human_winner"
) -> dict:
    """Summarize one judge config's reliability against human ground truth.

    `results` should look like a single judge config's slice of `judges.run_judges`' output: one
    row per item, with a `judge_col` prediction and a `human_col` ground-truth label. Non-substantive
    outcomes ("tie (inconsistent)", "unparseable", "error") are kept as their own categories in the
    confusion matrix/kappa (they're never "correct"), and reported separately as rates.

    Returns a dict with: n_items, kappa, confusion (DataFrame), accuracy, inconsistency_rate,
    unparseable_rate, error_rate.
    """
    judge_labels = results[judge_col]
    human_labels = results[human_col]
    categories = sorted(set(judge_labels) | set(human_labels))

    confusion = confusion_matrix_report(judge_labels, human_labels, categories=categories)
    return {
        "n_items": len(results),
        "kappa": cohens_kappa(judge_labels, human_labels, categories=categories),
        "confusion": confusion,
        "accuracy": confusion.attrs["accuracy"],
        "inconsistency_rate": float((judge_labels == "tie (inconsistent)").mean()),
        "unparseable_rate": float((judge_labels == "unparseable").mean()),
        "error_rate": float((judge_labels == "error").mean()),
    }


def accuracy_by_human_agreement(
    results: pd.DataFrame,
    items: pd.DataFrame,
    judge_col: str = "winner",
    human_col: str = "human_winner",
) -> pd.DataFrame:
    """Judge accuracy broken down by item difficulty (how unanimous humans were on that item).

    Joins `results` to `items`'s `human_agreement` column on (question_id, model_a, model_b) and
    groups by the *exact* `human_agreement` value, rather than quantile-binning it. `human_agreement`
    is `top_vote_count / n_votes` for a small number of human votes (see `load_mt_bench_items`), so
    it only takes a handful of distinct values (0.5, 0.6, 0.67, 0.75, 0.8, 1.0, ...) — quantile
    binning (an earlier version of this function used `pd.qcut`) degenerates to a single bin
    whenever a majority of items land on the same value, which in practice is common (most MT-Bench
    battles have full human agreement). Grouping by the exact value sidesteps that failure mode and
    is more interpretable besides. See the module docstring for why this stands in for a
    calibration curve here.

    Returns one row per distinct `human_agreement` value present: human_agreement, n, judge_accuracy.
    """
    merged = results.merge(
        items[["question_id", "model_a", "model_b", "human_agreement"]],
        on=["question_id", "model_a", "model_b"],
        how="left",
    )
    merged["correct"] = (merged[judge_col] == merged[human_col]).astype(float)

    return (
        merged.groupby("human_agreement", observed=True)
        .agg(n=("correct", "size"), judge_accuracy=("correct", "mean"))
        .reset_index()
        .sort_values("human_agreement")
        .reset_index(drop=True)
    )
