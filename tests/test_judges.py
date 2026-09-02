"""Tests for `llm_judge_audit.judges`.

Pure helpers (`_majority_vote`, `_extract_qa`, `_build_prompt`, `_parse_verdict`, `_consolidate`)
are tested directly. `load_mt_bench_items` hits the Hugging Face Hub (downloads/caches the
dataset) — it's exercised here with a small `n_items` to keep the test fast. `run_judges` is
tested against a fake Anthropic client (`_FakeClient` below) so the suite never makes real,
billed API calls.
"""

import time
from types import SimpleNamespace

import anthropic
import httpx2
import pandas as pd
import pytest

from llm_judge_audit.judges import (
    JudgeConfig,
    _build_prompt,
    _consolidate,
    _extract_qa,
    _majority_vote,
    _parse_verdict,
    load_mt_bench_items,
    run_judges,
)


def test_majority_vote_unanimous():
    assert _majority_vote(["model_a", "model_a", "model_a"]) == ("model_a", 1.0)


def test_majority_vote_majority_with_dissent():
    label, agreement = _majority_vote(["model_a", "model_a", "model_b"])
    assert label == "model_a"
    assert agreement == 2 / 3


def test_majority_vote_exact_tie_is_no_consensus():
    label, agreement = _majority_vote(["model_a", "model_b"])
    assert label == "no_consensus"
    assert agreement == 0.5


def test_extract_qa_turn_1():
    conversation = [
        {"role": "user", "content": "question one"},
        {"role": "assistant", "content": "answer one"},
        {"role": "user", "content": "question two"},
        {"role": "assistant", "content": "answer two"},
    ]
    assert _extract_qa(conversation, turn=1) == ("question one", "answer one")


def test_extract_qa_turn_2():
    conversation = [
        {"role": "user", "content": "question one"},
        {"role": "assistant", "content": "answer one"},
        {"role": "user", "content": "question two"},
        {"role": "assistant", "content": "answer two"},
    ]
    assert _extract_qa(conversation, turn=2) == ("question two", "answer two")


def test_load_mt_bench_items_smoke():
    items = load_mt_bench_items(n_items=5, seed=0)

    assert len(items) == 5
    expected_columns = {
        "question_id",
        "turn",
        "model_a",
        "model_b",
        "question",
        "answer_a",
        "answer_b",
        "human_raters",
        "human_votes",
        "human_n_votes",
        "human_winner",
        "human_agreement",
        "gpt4_pair_winner",
    }
    assert expected_columns.issubset(items.columns)
    assert (items["turn"] == 1).all()
    assert items["human_winner"].isin(["model_a", "model_b", "tie"]).all()
    # no_consensus (exact-tie) battles are dropped, so every remaining battle has a unique
    # plurality winner — the lowest that can produce is 0.5 (e.g. a 2/1/1 split of 4 votes).
    assert (items["human_agreement"] >= 0.5).all()
    assert all(
        len(raters) == n for raters, n in zip(items["human_raters"], items["human_n_votes"])
    )


# --- JudgeConfig -------------------------------------------------------------------------------


def test_judge_config_rejects_invalid_variant():
    with pytest.raises(ValueError):
        JudgeConfig(name="x", model="claude-haiku-4-5-20251001", variant="bogus")


def test_judge_config_max_tokens_defaults_are_variant_aware():
    # cot needs real headroom to fit reasoning *and* the verdict line — see judges.py's
    # _DEFAULT_MAX_TOKENS comment for the live-run bug this default is guarding against.
    direct = JudgeConfig(name="d", model="claude-haiku-4-5-20251001", variant="direct")
    cot = JudgeConfig(name="c", model="claude-haiku-4-5-20251001", variant="cot")
    assert direct.max_tokens < cot.max_tokens


def test_judge_config_max_tokens_explicit_override_respected():
    config = JudgeConfig(
        name="c", model="claude-haiku-4-5-20251001", variant="cot", max_tokens=99
    )
    assert config.max_tokens == 99


# --- prompt building / parsing ------------------------------------------------------------------


def test_build_prompt_includes_question_and_answers():
    prompt = _build_prompt("Q?", "A-answer-text", "B-answer-text", "direct")
    assert "Q?" in prompt
    assert "A-answer-text" in prompt
    assert "B-answer-text" in prompt


def test_build_prompt_cot_asks_for_reasoning_direct_does_not():
    direct = _build_prompt("Q?", "a", "b", "direct")
    cot = _build_prompt("Q?", "a", "b", "cot")
    assert "step by step" not in direct.lower()
    assert "step by step" in cot.lower()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Verdict: A", "A"),
        ("verdict: b", "B"),
        ("Some reasoning here.\nVerdict: TIE", "TIE"),
        ("Verdict: [A]", "A"),
        ("no verdict here", None),
        # a cot response can mention "Verdict:"-shaped phrasing while reasoning out loud, before
        # its actual final answer — the LAST occurrence must win, not the first.
        ("Leaning towards Verdict: A at first, but reconsidering... Verdict: B", "B"),
    ],
)
def test_parse_verdict(text, expected):
    assert _parse_verdict(text) == expected


# --- position-bias consolidation -----------------------------------------------------------------
# verdict_ab is the verdict when model_a is shown in slot A; verdict_ba is the verdict when
# model_b is shown in slot A (i.e. model_a is in slot B). "Agreement" means the same underlying
# model wins regardless of which slot it was shown in.


def test_consolidate_agrees_on_model_a():
    assert _consolidate("A", "B") == "model_a"


def test_consolidate_agrees_on_model_b():
    assert _consolidate("B", "A") == "model_b"


def test_consolidate_agrees_on_tie():
    assert _consolidate("TIE", "TIE") == "tie"


def test_consolidate_disagreement_is_inconsistent():
    # "A" both times means: model_a wins when shown first, but model_b wins when shown first —
    # i.e. whichever model is in slot A wins. That's position bias, not a real preference.
    assert _consolidate("A", "A") == "tie (inconsistent)"


def test_consolidate_unparseable_verdict():
    assert _consolidate(None, "A") == "unparseable"
    assert _consolidate("A", None) == "unparseable"


# --- run_judges, against a fake Anthropic client (no real API calls) ----------------------------


class _FakeMessages:
    """Stand-in for `client.messages`. `responses` is consumed in call order; an item that is an
    exception instance is raised instead of returned. A plain string implies stop_reason="end_turn";
    pass a (text, stop_reason) tuple to simulate e.g. truncation ("max_tokens")."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        text, stop_reason = item if isinstance(item, tuple) else (item, "end_turn")
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)], stop_reason=stop_reason
        )


class _FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


class _SlowFakeMessages:
    """Like `_FakeMessages`, but every call sleeps `delay` seconds — used to verify `run_judges`
    actually executes jobs concurrently when `max_workers > 1`, not just accepts the parameter."""

    def __init__(self, delay: float):
        self.delay = delay
        self.call_count = 0

    def create(self, **kwargs):
        time.sleep(self.delay)
        self.call_count += 1
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Verdict: TIE")], stop_reason="end_turn"
        )


class _SlowFakeClient:
    def __init__(self, delay: float):
        self.messages = _SlowFakeMessages(delay)


def _make_items(question_id=1):
    return pd.DataFrame(
        [
            {
                "question_id": question_id,
                "model_a": "modelA",
                "model_b": "modelB",
                "question": "Q?",
                "answer_a": "answer A text",
                "answer_b": "answer B text",
                "human_winner": "model_a",
            }
        ]
    )


def test_run_judges_consistent_verdict_both_orders():
    items = _make_items()
    config = JudgeConfig(name="test-judge", model="claude-haiku-4-5-20251001", variant="direct")
    # A-first call says "A" (model_a); B-first call says "B" (also model_a, in the other slot).
    client = _FakeClient(["Verdict: A", "Verdict: B"])

    result = run_judges(items, [config], client=client, max_workers=1)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["winner"] == "model_a"
    assert row["verdict_original_order"] == "A"
    assert row["verdict_swapped_order"] == "B"
    assert row["error"] is None
    assert row["response_original_order"] == "Verdict: A"
    assert row["response_swapped_order"] == "Verdict: B"
    assert row["stop_reason_original_order"] == "end_turn"
    assert row["stop_reason_swapped_order"] == "end_turn"
    assert len(client.messages.calls) == 2


def test_run_judges_flags_truncated_response_via_stop_reason():
    # Regression test for a real bug: a response cut off by max_tokens before it reaches the
    # verdict line parses as "unparseable" just like a genuinely ambiguous judgment would — the
    # only way to tell them apart is stop_reason, so it must survive into the results table.
    items = _make_items()
    config = JudgeConfig(
        name="test-judge", model="claude-haiku-4-5-20251001", variant="cot", swap_order=False
    )
    client = _FakeClient([("some reasoning that never reaches a verdict...", "max_tokens")])

    result = run_judges(items, [config], client=client, max_workers=1)

    row = result.iloc[0]
    assert row["winner"] == "unparseable"
    assert row["stop_reason_original_order"] == "max_tokens"


def test_run_judges_no_swap_makes_a_single_call():
    items = _make_items()
    config = JudgeConfig(
        name="test-judge", model="claude-haiku-4-5-20251001", variant="direct", swap_order=False
    )
    client = _FakeClient(["Verdict: TIE"])

    result = run_judges(items, [config], client=client, max_workers=1)

    assert len(client.messages.calls) == 1
    assert result.iloc[0]["winner"] == "tie"
    assert result.iloc[0]["verdict_swapped_order"] is None


def test_run_judges_flags_position_bias_as_inconsistent():
    items = _make_items()
    config = JudgeConfig(name="test-judge", model="claude-haiku-4-5-20251001", variant="direct")
    client = _FakeClient(["Verdict: A", "Verdict: A"])

    result = run_judges(items, [config], client=client, max_workers=1)

    assert result.iloc[0]["winner"] == "tie (inconsistent)"


def test_run_judges_records_api_error_and_keeps_going():
    items = pd.concat([_make_items(1), _make_items(2)], ignore_index=True)
    config = JudgeConfig(
        name="test-judge", model="claude-haiku-4-5-20251001", variant="direct", swap_order=False
    )
    dummy_request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.APIConnectionError(message="boom", request=dummy_request)
    client = _FakeClient([error, "Verdict: A"])

    result = run_judges(items, [config], client=client, max_workers=1)

    assert len(result) == 2
    assert result.iloc[0]["winner"] == "error"
    assert "boom" in result.iloc[0]["error"]
    assert result.iloc[1]["winner"] == "model_a"


def test_run_judges_preserves_original_order_verdict_when_swap_call_fails():
    # Regression test: the original-order and swap-order calls are two independent, already-paid
    # API calls. A failure in the second must not discard the successful first one.
    items = _make_items()
    config = JudgeConfig(name="test-judge", model="claude-haiku-4-5-20251001", variant="direct")
    dummy_request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.APIConnectionError(message="swap call boom", request=dummy_request)
    client = _FakeClient(["Verdict: A", error])

    result = run_judges(items, [config], client=client, max_workers=1)

    row = result.iloc[0]
    assert row["winner"] == "error"
    assert "swap call boom" in row["error"]
    # the successful original-order call's data must survive, not be wiped to None
    assert row["verdict_original_order"] == "A"
    assert row["response_original_order"] == "Verdict: A"
    assert row["stop_reason_original_order"] == "end_turn"
    assert row["verdict_swapped_order"] is None


def test_run_judges_requires_api_key_when_no_client_given(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    items = _make_items()
    config = JudgeConfig(name="test-judge", model="claude-haiku-4-5-20251001", variant="direct")

    with pytest.raises(RuntimeError):
        run_judges(items, [config])


def test_run_judges_runs_jobs_concurrently():
    # 6 independent single-call jobs; with real concurrency this should take well under the
    # fully-sequential time. Not just "did max_workers get accepted" — actually timed.
    items = pd.concat([_make_items(i) for i in range(6)], ignore_index=True)
    config = JudgeConfig(
        name="t", model="claude-haiku-4-5-20251001", variant="direct", swap_order=False
    )
    delay = 0.05

    sequential_client = _SlowFakeClient(delay)
    start = time.monotonic()
    run_judges(items, [config], client=sequential_client, max_workers=1)
    sequential_elapsed = time.monotonic() - start

    concurrent_client = _SlowFakeClient(delay)
    start = time.monotonic()
    run_judges(items, [config], client=concurrent_client, max_workers=6)
    concurrent_elapsed = time.monotonic() - start

    assert sequential_client.messages.call_count == 6
    assert concurrent_client.messages.call_count == 6
    # generous margin for CI/sandbox timing jitter — this is checking for a real, large speedup
    # (~6x expected), not a marginal one
    assert concurrent_elapsed < sequential_elapsed * 0.6
