"""End-to-end analysis orchestration.

Turns `judges.run_judges` output into the classical/causal/Bayesian summaries and the dashboard
payload, in one place — so the live pipeline (`scripts/run_pipeline.py`) and any other results
source (e.g. a synthetic/dev dataset) share exactly the same analysis logic rather than each
re-implementing it.
"""

from __future__ import annotations

import pandas as pd

from . import bayesian_model, causal_experiment, classical_stats, viz
from .judges import JudgeConfig


def analyze_results(
    results: pd.DataFrame,
    judge_configs: list[JudgeConfig],
    items: pd.DataFrame | None = None,
    seed: int = 42,
) -> dict:
    """Compute every downstream analysis for a set of judge results.

    - Classical stats (`classical_stats.evaluate_judge`) for each judge config individually.
    - A causal effect (`causal_experiment.estimate_variant_effect`) for each model that has both
      a "direct" and a "cot" config in `judge_configs` — comparing cot vs. direct for that model.
    - The Bayesian reliability model (`bayesian_model`) fit jointly across *all* judge configs
      (each config is one "rater"), if there are enough distinct raters/items to fit it.
    - If `items` (e.g. `load_mt_bench_items`'s output) is given, each judge config's accuracy
      broken down by item difficulty (`classical_stats.accuracy_by_human_agreement`).

    Returns:
        dict with judge_summaries, causal_effects, bayesian_summary, difficulty_breakdown,
        dashboard_payload.
    """
    judge_summaries = {
        config.name: classical_stats.evaluate_judge(results[results["judge_name"] == config.name])
        for config in judge_configs
    }

    difficulty_breakdown = {}
    if items is not None:
        for config in judge_configs:
            subset = results[results["judge_name"] == config.name]
            difficulty_breakdown[config.name] = classical_stats.accuracy_by_human_agreement(
                subset, items
            )

    causal_effects = {}
    configs_by_model: dict[str, dict[str, JudgeConfig]] = {}
    for config in judge_configs:
        configs_by_model.setdefault(config.model, {})[config.variant] = config
    for model, variants in configs_by_model.items():
        if "direct" not in variants or "cot" not in variants:
            continue
        model_results = results[results["judge_model"] == model]
        effect = causal_experiment.estimate_variant_effect(
            model_results, baseline="direct", treatment="cot", seed=seed
        )
        causal_effects[f"{model}: cot_vs_direct"] = effect

    bayesian_summary = None
    observations = bayesian_model.judge_agreement_observations(results)
    if observations["rater"].nunique() >= 2 and observations["item"].nunique() >= 2:
        model_obj = bayesian_model.build_model(observations)
        idata = bayesian_model.fit(model_obj, seed=seed)
        bayesian_summary = bayesian_model.summarize(idata)

    dashboard_payload = viz.to_dashboard_json(
        judge_summaries=judge_summaries,
        causal_effects=causal_effects,
        bayesian_summary=bayesian_summary,
        difficulty_breakdown=difficulty_breakdown or None,
    )

    return {
        "judge_summaries": judge_summaries,
        "causal_effects": causal_effects,
        "bayesian_summary": bayesian_summary,
        "difficulty_breakdown": difficulty_breakdown or None,
        "dashboard_payload": dashboard_payload,
    }


def print_analysis_summary(analysis: dict) -> None:
    """Print a short human-readable summary of an `analyze_results` result to stdout.

    Shared by `scripts/run_pipeline.py` and `scripts/generate_fake_results.py` so the two don't
    drift out of sync with each other as the summary format evolves.
    """
    for name, summary in analysis["judge_summaries"].items():
        print(f"  {name}: accuracy={summary['accuracy']:.2f}, kappa={summary['kappa']:.2f}")
    for label, effect in analysis["causal_effects"].items():
        print(
            f"  {label}: effect={effect['effect']:+.2f} "
            f"95% CI [{effect['ci_lower']:+.2f}, {effect['ci_upper']:+.2f}] "
            f"McNemar p={effect['mcnemar_pvalue']:.3f}"
        )
