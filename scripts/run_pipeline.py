#!/usr/bin/env python3
"""End-to-end pipeline: load ground-truth items, run LLM judges against the real Anthropic API,
and compute all three statistical analyses (classical, causal, Bayesian), saving results to
`data/processed/`.

This makes real, billed Anthropic API calls — it prints how many before doing anything, and
requires `--yes` (or an interactive "y" confirmation) to proceed.

Usage:
    .venv/bin/python scripts/run_pipeline.py --n-items 20 --variants direct cot --yes

Requires ANTHROPIC_API_KEY (in `.env` or the environment) — see judges.run_judges.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))  # run standalone without installing the package

from llm_judge_audit.judges import JudgeConfig, load_mt_bench_items, run_judges  # noqa: E402
from llm_judge_audit.pipeline import analyze_results, print_analysis_summary  # noqa: E402

DEFAULT_OUT_DIR = _REPO_ROOT / "data" / "processed"
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-items", type=int, default=20, help="number of battles to judge (default: 20)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Anthropic model id (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["direct", "cot"],
        choices=["direct", "cot"],
        help="judge-prompt variant(s) to run; pass both for the causal comparison (default: both)",
    )
    parser.add_argument(
        "--no-swap", action="store_true", help="disable position-bias swap-order calls (halves API calls)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="concurrent judge API calls in flight (default: 8; higher is faster but more likely to hit rate limits)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--yes", action="store_true", help="skip the cost confirmation prompt")
    return parser.parse_args(argv)


def _confirm(args: argparse.Namespace) -> bool:
    swap_order = not args.no_swap
    calls_per_item = (2 if swap_order else 1) * len(args.variants)
    total_calls = args.n_items * calls_per_item
    print(
        f"This will make {total_calls} Anthropic API calls "
        f"({args.n_items} items x {len(args.variants)} variant(s) x "
        f"{'2 orderings' if swap_order else '1 ordering'}), model={args.model!r}."
    )
    if args.yes:
        return True
    return input("Proceed? [y/N] ").strip().lower() == "y"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not _confirm(args):
        print("Aborted.")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    swap_order = not args.no_swap

    print(f"Loading {args.n_items} ground-truth items from MT-Bench Human Judgments...")
    items = load_mt_bench_items(n_items=args.n_items, seed=args.seed)
    items.drop(columns=["human_votes", "human_raters"]).to_json(
        args.out_dir / "items.json", orient="records"
    )

    judge_configs = [
        JudgeConfig(name=f"{args.model}-{variant}", model=args.model, variant=variant, swap_order=swap_order)
        for variant in args.variants
    ]
    print(f"Running {len(judge_configs)} judge config(s) on {len(items)} items...")
    results = run_judges(items, judge_configs, max_workers=args.max_workers)
    results.to_json(args.out_dir / "judge_results.json", orient="records")
    n_errors = int((results["winner"] == "error").sum())
    if n_errors:
        print(f"  Warning: {n_errors}/{len(results)} judge calls errored — see the 'error' column.")

    print("Analyzing results (classical stats, causal effect, Bayesian model)...")
    analysis = analyze_results(results, judge_configs, items=items, seed=args.seed)
    print_analysis_summary(analysis)

    if analysis["bayesian_summary"] is not None:
        analysis["bayesian_summary"].to_csv(args.out_dir / "bayesian_summary.csv")
    else:
        print("  Bayesian model skipped: need >=2 judge configs and >=2 items.")

    (args.out_dir / "dashboard.json").write_text(json.dumps(analysis["dashboard_payload"], indent=2))

    print(f"Done. Results written to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
