#!/usr/bin/env python3
"""Generate SYNTHETIC judge results for dashboard development — NOT real judge output.

Real MT-Bench ground-truth items (free, no API calls) are scored by 4 *simulated* judge configs
(2 models x 2 prompt variants) with plausible-but-fabricated accuracy/inconsistency rates, so the
analysis pipeline and dashboard can be built and visually checked before spending real API money.
Every output file is prefixed `fake_` so it can't be mistaken for real results later.

Usage:
    .venv/bin/python scripts/generate_fake_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from llm_judge_audit.judges import JudgeConfig, load_mt_bench_items  # noqa: E402
from llm_judge_audit.pipeline import analyze_results, print_analysis_summary  # noqa: E402

OUT_DIR = _REPO_ROOT / "data" / "processed"

# (model, variant) -> (base_accuracy, inconsistency_rate, error_rate). Fabricated but plausible:
# reasoning (cot) helps both models; the larger model (sonnet) is more reliable than haiku.
FAKE_JUDGE_PROFILES = {
    ("claude-haiku-4-5-20251001", "direct"): (0.72, 0.08, 0.01),
    ("claude-haiku-4-5-20251001", "cot"): (0.83, 0.04, 0.01),
    ("claude-sonnet-5", "direct"): (0.80, 0.05, 0.01),
    ("claude-sonnet-5", "cot"): (0.90, 0.02, 0.01),
}

_OTHER_LABELS = {
    "model_a": ["model_b", "tie"],
    "model_b": ["model_a", "tie"],
    "tie": ["model_a", "model_b"],
}


def _simulate_winner(rng, human_winner, human_agreement, base_accuracy, inconsistency_rate, error_rate):
    """One fabricated verdict. Accuracy is nudged down on items with lower `human_agreement`
    (harder/more ambiguous battles), so the fake data shows a plausible difficulty effect too."""
    if rng.random() < error_rate:
        return "error"
    if rng.random() < inconsistency_rate:
        return "tie (inconsistent)"
    p_correct = float(np.clip(base_accuracy + 0.3 * (human_agreement - 0.8), 0.05, 0.98))
    if rng.random() < p_correct:
        return human_winner
    return rng.choice(_OTHER_LABELS[human_winner])


def generate(n_items: int = 120, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, list[JudgeConfig]]:
    items = load_mt_bench_items(n_items=n_items, seed=seed)
    rng = np.random.default_rng(seed)

    judge_configs = [
        JudgeConfig(name=f"{model}-{variant}", model=model, variant=variant)
        for model, variant in FAKE_JUDGE_PROFILES
    ]

    rows = []
    for (model, variant), (base_acc, inconsistency, error_rate) in FAKE_JUDGE_PROFILES.items():
        for row in items.itertuples():
            winner = _simulate_winner(
                rng, row.human_winner, row.human_agreement, base_acc, inconsistency, error_rate
            )
            rows.append(
                {
                    "question_id": row.question_id,
                    "model_a": row.model_a,
                    "model_b": row.model_b,
                    "judge_name": f"{model}-{variant}",
                    "judge_model": model,
                    "variant": variant,
                    "winner": winner,
                    "human_winner": row.human_winner,
                    "error": "synthetic" if winner == "error" else None,
                }
            )
    results = pd.DataFrame(rows)
    return items, results, judge_configs


def main() -> int:
    items, results, judge_configs = generate()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    items.drop(columns=["human_votes", "human_raters"]).to_json(OUT_DIR / "fake_items.json", orient="records")
    results.to_json(OUT_DIR / "fake_judge_results.json", orient="records")
    print(f"Generated {len(results)} synthetic verdicts across {len(judge_configs)} fake judge configs.")

    print("Analyzing (classical stats, causal effects, Bayesian model)...")
    analysis = analyze_results(results, judge_configs, items=items)
    print_analysis_summary(analysis)

    if analysis["bayesian_summary"] is not None:
        analysis["bayesian_summary"].to_csv(OUT_DIR / "fake_bayesian_summary.csv")

    (OUT_DIR / "fake_dashboard.json").write_text(json.dumps(analysis["dashboard_payload"], indent=2))
    print(f"Wrote fake_items.json, fake_judge_results.json, fake_dashboard.json to {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
