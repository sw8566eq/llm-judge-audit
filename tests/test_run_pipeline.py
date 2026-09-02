"""Tests for `scripts/run_pipeline.py`'s argument parsing.

`main()` makes real, billed Anthropic API calls, so it isn't exercised here — only the pure
`parse_args`/`_confirm` logic, which needs no network access or API key.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import run_pipeline  # noqa: E402


def test_parse_args_defaults():
    args = run_pipeline.parse_args([])
    assert args.n_items == 20
    assert args.variants == ["direct", "cot"]
    assert args.model == run_pipeline.DEFAULT_MODEL
    assert args.no_swap is False
    assert args.yes is False
    assert args.max_workers == 8


def test_parse_args_accepts_custom_max_workers():
    args = run_pipeline.parse_args(["--max-workers", "1"])
    assert args.max_workers == 1


def test_parse_args_rejects_unknown_variant():
    with pytest.raises(SystemExit):
        run_pipeline.parse_args(["--variants", "bogus"])


def test_parse_args_accepts_custom_values():
    args = run_pipeline.parse_args(
        ["--n-items", "5", "--variants", "direct", "--no-swap", "--yes", "--model", "claude-sonnet-5"]
    )
    assert args.n_items == 5
    assert args.variants == ["direct"]
    assert args.no_swap is True
    assert args.yes is True
    assert args.model == "claude-sonnet-5"


def test_confirm_skips_prompt_when_yes_flag_set(capsys):
    args = run_pipeline.parse_args(["--n-items", "3", "--yes"])
    assert run_pipeline._confirm(args) is True
    out = capsys.readouterr().out
    assert "12 Anthropic API calls" in out  # 3 items x 2 variants x 2 orderings


def test_confirm_reports_call_count_without_swap(capsys):
    args = run_pipeline.parse_args(["--n-items", "3", "--variants", "direct", "--no-swap", "--yes"])
    run_pipeline._confirm(args)
    out = capsys.readouterr().out
    assert "3 Anthropic API calls" in out  # 3 items x 1 variant x 1 ordering
