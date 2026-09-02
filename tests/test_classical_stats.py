"""Tests for `llm_judge_audit.classical_stats`."""

import numpy as np
import pandas as pd
import pytest

from llm_judge_audit.classical_stats import (
    accuracy_by_human_agreement,
    calibration_curve,
    cohens_kappa,
    confusion_matrix_report,
    evaluate_judge,
    mcnemars_test,
)


# --- cohens_kappa --------------------------------------------------------------------------------


def test_cohens_kappa_perfect_agreement():
    labels = ["model_a", "model_b", "tie", "model_a", "model_b"]
    assert cohens_kappa(labels, labels) == pytest.approx(1.0)


def test_cohens_kappa_textbook_example():
    # Classic 2x2 example: 100 items, agreement table
    #                human=yes  human=no
    # judge=yes           45        15
    # judge=no             10        30
    # p_o = 0.75, p_e = 0.51 -> kappa = 0.24 / 0.49 ~= 0.4898
    judge = ["yes"] * 60 + ["no"] * 40
    human = ["yes"] * 45 + ["no"] * 15 + ["yes"] * 10 + ["no"] * 30
    assert cohens_kappa(judge, human) == pytest.approx(0.4898, abs=1e-3)


# --- confusion_matrix_report -----------------------------------------------------------------


def test_confusion_matrix_report_counts_and_accuracy():
    judge = ["model_a", "model_a", "model_b", "tie"]
    human = ["model_a", "model_b", "model_b", "tie"]

    table = confusion_matrix_report(judge, human)

    assert table.loc["model_a", "model_a"] == 1
    assert table.loc["model_a", "model_b"] == 1
    assert table.loc["model_b", "model_b"] == 1
    assert table.loc["tie", "tie"] == 1
    assert table.attrs["accuracy"] == pytest.approx(0.75)


def test_confusion_matrix_report_includes_extra_categories():
    # "error" never matches any human label, but should still appear as a row if passed explicitly.
    judge = ["model_a", "error"]
    human = ["model_a", "model_b"]
    table = confusion_matrix_report(judge, human, categories=["model_a", "model_b", "tie", "error"])
    assert list(table.index) == ["model_a", "model_b", "tie", "error"]
    assert table.loc["error", "model_b"] == 1
    assert table.attrs["accuracy"] == pytest.approx(0.5)


# --- calibration_curve ----------------------------------------------------------------------


def test_calibration_curve_bins_and_accuracy():
    # 2 bins: [0, 0.5) and [0.5, 1.0]. Low-confidence items are wrong more often.
    confidences = [0.1, 0.2, 0.9, 0.8, 1.0]
    correct = [0, 1, 1, 1, 1]

    curve = calibration_curve(confidences, correct, n_bins=2)

    assert len(curve) == 2
    low_bin = curve.iloc[0]
    high_bin = curve.iloc[1]
    assert low_bin["n"] == 2
    assert low_bin["observed_accuracy"] == pytest.approx(0.5)
    assert high_bin["n"] == 3
    assert high_bin["observed_accuracy"] == pytest.approx(1.0)


def test_calibration_curve_rejects_out_of_range_confidence():
    with pytest.raises(ValueError):
        calibration_curve([1.5], [1])


def test_calibration_curve_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        calibration_curve([0.1, 0.2], [1])


# --- mcnemars_test ---------------------------------------------------------------------------


def test_mcnemars_test_balanced_discordance_gives_pvalue_one():
    a_correct = np.array([True] * 50 + [True] * 7 + [False] * 7 + [False] * 20)
    b_correct = np.array([True] * 50 + [False] * 7 + [True] * 7 + [False] * 20)

    result = mcnemars_test(a_correct, b_correct)

    assert result["statistic"] == pytest.approx(7.0)
    assert result["pvalue"] == pytest.approx(1.0)


def test_mcnemars_test_lopsided_discordance_is_significant():
    # A correct-only on 30 items, B correct-only on 2 items -> clearly lopsided.
    a_correct = np.array([True] * 50 + [True] * 30 + [False] * 2 + [False] * 18)
    b_correct = np.array([True] * 50 + [False] * 30 + [True] * 2 + [False] * 18)

    result = mcnemars_test(a_correct, b_correct)

    assert result["statistic"] == pytest.approx(2.0)
    assert result["pvalue"] < 0.01


def test_mcnemars_test_table_orientation_matches_docstring():
    # rows = a_correct=[True, False], columns = b_correct=[True, False] — locks in the documented
    # convention so the returned table can't silently drift out of sync with its docstring again.
    a_correct = np.array([True, True, False, False])
    b_correct = np.array([True, False, True, False])

    table = mcnemars_test(a_correct, b_correct)["table"]

    assert table[0, 0] == 1  # a=T, b=T ("both")
    assert table[0, 1] == 1  # a=T, b=F ("a_only")
    assert table[1, 0] == 1  # a=F, b=T ("b_only")
    assert table[1, 1] == 1  # a=F, b=F ("neither")


# --- evaluate_judge / accuracy_by_human_agreement (project-shaped data) ----------------------


def _results_and_items():
    results = pd.DataFrame(
        {
            "question_id": [1, 2, 3, 4],
            "model_a": ["m1", "m1", "m1", "m1"],
            "model_b": ["m2", "m2", "m2", "m2"],
            "winner": ["model_a", "model_b", "tie (inconsistent)", "model_a"],
            "human_winner": ["model_a", "model_b", "model_a", "model_b"],
        }
    )
    items = pd.DataFrame(
        {
            "question_id": [1, 2, 3, 4],
            "model_a": ["m1", "m1", "m1", "m1"],
            "model_b": ["m2", "m2", "m2", "m2"],
            "human_agreement": [1.0, 1.0, 0.5, 0.67],
        }
    )
    return results, items


def test_evaluate_judge_reports_accuracy_and_inconsistency():
    results, _ = _results_and_items()
    summary = evaluate_judge(results)

    assert summary["n_items"] == 4
    assert summary["accuracy"] == pytest.approx(0.5)  # items 1 and 2 correct, 3 and 4 wrong
    assert summary["inconsistency_rate"] == pytest.approx(0.25)
    assert summary["error_rate"] == 0.0


def test_accuracy_by_human_agreement_groups_by_exact_value():
    results, items = _results_and_items()
    # items' human_agreement values: [1.0, 1.0, 0.5, 0.67] -> 3 distinct values
    report = accuracy_by_human_agreement(results, items)

    assert report["n"].sum() == 4
    assert set(report["human_agreement"]) == {1.0, 0.5, 0.67}
    # the two human_agreement=1.0 items (question_id 1 and 2) were both judged correctly
    row = report[report["human_agreement"] == 1.0].iloc[0]
    assert row["n"] == 2
    assert row["judge_accuracy"] == pytest.approx(1.0)
    assert report["judge_accuracy"].between(0, 1).all()


def test_accuracy_by_human_agreement_does_not_collapse_when_skewed():
    # a majority of items sharing one human_agreement value used to collapse pd.qcut to 1 bin;
    # grouping by exact value must still separate out the minority values.
    results = pd.DataFrame(
        {
            "question_id": range(1, 11),
            "model_a": ["m1"] * 10,
            "model_b": ["m2"] * 10,
            "winner": ["model_a"] * 8 + ["model_b"] * 2,
            "human_winner": ["model_a"] * 10,
        }
    )
    items = pd.DataFrame(
        {
            "question_id": range(1, 11),
            "model_a": ["m1"] * 10,
            "model_b": ["m2"] * 10,
            "human_agreement": [1.0] * 8 + [0.5] * 2,
        }
    )

    report = accuracy_by_human_agreement(results, items)

    assert len(report) == 2
    assert set(report["human_agreement"]) == {1.0, 0.5}
