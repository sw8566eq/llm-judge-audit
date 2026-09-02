"""LLM judge orchestration.

Runs one or more "judge" configurations (model + prompt variant) over a set of pairwise battles
(one question, two models' answers) and collects verdicts alongside the existing human
ground-truth labels.

A judge's *task* is always pairwise here — "which of these two responses is better, A, B, or a
tie?" — because that's the structure of the ground-truth data (see `load_mt_bench_items`). What
varies between judge configurations is the *prompting strategy*:
    - "direct": ask for a verdict with no reasoning.
    - "cot": ask the judge to reason step by step before giving a verdict.

Every judge call is run in **both answer orderings** (A-then-B and B-then-A) by default
(`JudgeConfig.swap_order`). This mirrors the methodology already baked into the MT-Bench dataset
itself — its `gpt4_pair` split includes a `"tie (inconsistent)"` label for battles where GPT-4's
verdict flipped depending on which answer was shown first, i.e. a position-bias detector. Doing
the same for our own judges lets us directly compare their position-bias rate to GPT-4's.

Ground-truth items come from the MT-Bench Human Judgments dataset (`lmsys/mt_bench_human_judgments`
on Hugging Face) — the dataset from Zheng et al.'s "Judging LLM-as-a-Judge" paper, which is the paper
that established the LLM-as-judge practice this project audits. It contains pairwise "battles"
(two models' answers to the same MT-Bench question), each with:
    - one or more independent human votes ("human" split) — used here as ground truth, and as a
      human-human inter-rater reliability baseline;
    - a GPT-4-as-judge verdict on the same battle ("gpt4_pair" split) — kept as a reference point,
      not treated as ground truth.
"""

from __future__ import annotations

import concurrent.futures
import os
import re
from collections import Counter
from dataclasses import dataclass

import anthropic
import pandas as pd
from datasets import load_dataset

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

_HF_DATASET = "lmsys/mt_bench_human_judgments"

VALID_VARIANTS = ("direct", "cot")

# Verdicts come back as "Verdict: A" / "Verdict: B" / "Verdict: TIE", optionally bracketed.
_VERDICT_RE = re.compile(r"verdict\s*:\s*\[?(A|B|TIE)\]?", re.IGNORECASE)


def _extract_qa(conversation: list[dict], turn: int) -> tuple[str, str]:
    """Pull the (question, answer) text for a 1-indexed `turn` out of an MT-Bench conversation.

    Conversations alternate user/assistant messages: [user_0, assistant_0, user_1, assistant_1, ...].
    Turn 1 is (user_0, assistant_0), turn 2 is (user_1, assistant_1), etc.
    """
    idx = (turn - 1) * 2
    question = conversation[idx]["content"]
    answer = conversation[idx + 1]["content"]
    return question, answer


def _majority_vote(winners: list[str]) -> tuple[str, float]:
    """Majority label and agreement fraction among a list of human "winner" votes for one battle.

    Returns ("no_consensus", top_count / n) when the top categories are exactly tied — e.g. one
    vote for model_a and one for model_b, with no third vote to break it.
    """
    counts = Counter(winners)
    top_label, top_count = counts.most_common(1)[0]
    tied_leaders = [label for label, c in counts.items() if c == top_count]
    if len(tied_leaders) > 1:
        return "no_consensus", top_count / len(winners)
    return top_label, top_count / len(winners)


def load_mt_bench_items(n_items: int = 200, turn: int = 1, seed: int = 42) -> pd.DataFrame:
    """Load a ground-truth item set from MT-Bench Human Judgments.

    Each row is one "battle": a question plus two models' answers to it, a human-consensus
    winner (`human_winner`) with an agreement fraction (`human_agreement`) computed across all
    independent human votes on that battle, and — where available — the GPT-4-as-judge verdict on
    the same battle as a reference baseline.

    Battles with multiple independent human votes are preferred (they let us measure human-human
    reliability as a baseline for judge reliability); single-vote battles fill out the sample up
    to `n_items`. Battles with no human consensus (an exact tie among vote categories) are dropped.

    Args:
        n_items: target number of battles to return.
        turn: which MT-Bench conversation turn to use (1 = first question only, no history).
        seed: random seed for sampling, for reproducibility.

    Returns:
        A DataFrame with columns: question_id, turn, model_a, model_b, question, answer_a,
        answer_b, human_raters, human_votes, human_n_votes, human_winner, human_agreement,
        gpt4_pair_winner. `human_raters[k]` is the annotator id (e.g. "expert_24") who cast the
        vote in `human_votes[k]` — needed to fit `bayesian_model`'s per-rater hierarchical model.
    """
    ds = load_dataset(_HF_DATASET)
    human = ds["human"].to_pandas()
    gpt4 = ds["gpt4_pair"].to_pandas()

    human = human[human["turn"] == turn]
    gpt4 = gpt4[gpt4["turn"] == turn]

    group_cols = ["question_id", "model_a", "model_b", "turn"]
    battles = []
    for (question_id, model_a, model_b, turn_), group in human.groupby(group_cols):
        votes = group["winner"].tolist()
        raters = group["judge"].tolist()
        winner, agreement = _majority_vote(votes)
        question, answer_a = _extract_qa(group.iloc[0]["conversation_a"], turn_)
        _, answer_b = _extract_qa(group.iloc[0]["conversation_b"], turn_)
        battles.append(
            {
                "question_id": question_id,
                "turn": turn_,
                "model_a": model_a,
                "model_b": model_b,
                "question": question,
                "answer_a": answer_a,
                "answer_b": answer_b,
                "human_raters": raters,
                "human_votes": votes,
                "human_n_votes": len(votes),
                "human_winner": winner,
                "human_agreement": agreement,
            }
        )
    battles_df = pd.DataFrame(battles)
    battles_df = battles_df[battles_df["human_winner"] != "no_consensus"].reset_index(drop=True)

    gpt4_lookup = gpt4.set_index(["question_id", "model_a", "model_b", "turn"])["winner"].to_dict()
    battles_df["gpt4_pair_winner"] = [
        gpt4_lookup.get((row.question_id, row.model_a, row.model_b, row.turn))
        for row in battles_df.itertuples()
    ]

    multi_vote = battles_df[battles_df["human_n_votes"] > 1]
    single_vote = battles_df[battles_df["human_n_votes"] == 1]

    sample = multi_vote.sample(frac=1, random_state=seed) if len(multi_vote) else multi_vote
    if len(sample) < n_items and len(single_vote):
        need = min(n_items - len(sample), len(single_vote))
        sample = pd.concat(
            [sample, single_vote.sample(n=need, random_state=seed)], ignore_index=True
        )

    return sample.head(n_items).reset_index(drop=True)


# Default max_tokens per variant. "cot" needs real headroom: a truncated response never reaches
# its "Verdict: X" line, which silently turns into an "unparseable" verdict — indistinguishable
# from a genuine judgment call unless you go looking. Found via a live run where 16% of cot
# verdicts came back unparseable and every single one had stop_reason="max_tokens" at the old
# default of 512.
_DEFAULT_MAX_TOKENS = {"direct": 256, "cot": 1536}


@dataclass(frozen=True)
class JudgeConfig:
    """One judge configuration under test: a model + a prompting strategy.

    Attributes:
        name: short identifier for this config, used in results (e.g. "claude-haiku-direct").
        model: the Anthropic model id to call (e.g. "claude-haiku-4-5-20251001").
        variant: prompting strategy, one of VALID_VARIANTS ("direct" | "cot").
        swap_order: if True (default), run both answer orderings per item to detect and neutralize
            position bias; a battle where the two orderings disagree is reported as
            "tie (inconsistent)", matching the convention already used by the dataset's own
            gpt4_pair judgments.
        max_tokens: max_tokens passed to the Anthropic API call. Defaults to a variant-appropriate
            value (`_DEFAULT_MAX_TOKENS`) if not given — "cot" gets substantially more headroom
            than "direct", since it has to fit reasoning *and* the verdict line.
    """

    name: str
    model: str
    variant: str
    swap_order: bool = True
    max_tokens: int | None = None

    def __post_init__(self):
        if self.variant not in VALID_VARIANTS:
            raise ValueError(f"variant must be one of {VALID_VARIANTS}, got {self.variant!r}")
        if self.max_tokens is None:
            object.__setattr__(self, "max_tokens", _DEFAULT_MAX_TOKENS[self.variant])


def _build_prompt(question: str, answer_a: str, answer_b: str, variant: str) -> str:
    """Build the judge prompt for one ordering of (answer_a, answer_b)."""
    context = (
        "You are comparing two AI assistant responses to the same user question. Judge strictly "
        "on the quality of the response to the question — not response length, and not the order "
        "in which the responses are presented.\n\n"
        f"[Question]\n{question}\n\n"
        f"[Assistant A's response]\n{answer_a}\n\n"
        f"[Assistant B's response]\n{answer_b}\n\n"
    )
    if variant == "direct":
        instructions = (
            "Which response is better? Reply with exactly one line and no other text:\n"
            "Verdict: A\nor\nVerdict: B\nor\nVerdict: TIE"
        )
    else:  # "cot"
        instructions = (
            "First, reason step by step about the strengths and weaknesses of each response "
            "relative to the other, in 3-5 sentences — do not write more than that. Then, on the "
            "final line, give your verdict in exactly this format:\n"
            "Verdict: A\nor\nVerdict: B\nor\nVerdict: TIE"
        )
    return context + instructions


def _parse_verdict(response_text: str) -> str | None:
    """Extract 'A' / 'B' / 'TIE' from a judge response, or None if unparseable.

    Uses the LAST match in the text, not the first: the prompt asks for the verdict on the final
    line, but a "cot" response's reasoning can legitimately mention "Verdict: X"-shaped phrasing
    earlier while thinking out loud (e.g. "leaning towards Verdict: A initially, but on
    reflection..."). The last occurrence is far more likely to be the model's actual final answer
    than the first.
    """
    matches = list(_VERDICT_RE.finditer(response_text))
    return matches[-1].group(1).upper() if matches else None


def _consolidate(verdict_ab: str | None, verdict_ba: str | None) -> str:
    """Combine the two order verdicts into a single winner label.

    `verdict_ab` is the verdict when A is shown first (slot A = model_a, slot B = model_b);
    `verdict_ba` is the verdict when B is shown first (slot A = model_b, slot B = model_a). Both
    are normalized to model_a/model_b/tie before comparing, so a swapped-order re-run isn't
    mistaken for disagreement just because the slot labels flipped.
    """

    def normalize(verdict: str | None, first_is_model_a: bool) -> str | None:
        if verdict is None:
            return None
        if verdict == "TIE":
            return "tie"
        return "model_a" if (verdict == "A") == first_is_model_a else "model_b"

    winner_ab = normalize(verdict_ab, first_is_model_a=True)
    winner_ba = normalize(verdict_ba, first_is_model_a=False)

    if winner_ab is None or winner_ba is None:
        return "unparseable"
    return winner_ab if winner_ab == winner_ba else "tie (inconsistent)"


def _call_judge(client: anthropic.Anthropic, config: JudgeConfig, prompt: str) -> tuple[str, str]:
    """Make one Anthropic API call; return (response_text, stop_reason).

    `stop_reason` is what makes a truncated response ("max_tokens") distinguishable from a
    response the model just declined to give a clean verdict on ("end_turn") — the difference
    between an engineering bug (too little `max_tokens` headroom) and an actual judge behavior.

    Retries for transient errors (rate limits, timeouts, server errors) are handled by the
    Anthropic client itself (`max_retries` on the client passed into `run_judges`); this function
    lets non-transient errors (auth, bad request, etc.) propagate immediately.
    """
    response = client.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return text, response.stop_reason


def _score_one(client: anthropic.Anthropic, config: JudgeConfig, row) -> dict:
    """Score a single (item, judge_config) pair. One unit of `run_judges`' concurrent work."""
    record = {
        "question_id": row.question_id,
        "model_a": row.model_a,
        "model_b": row.model_b,
        "judge_name": config.name,
        "judge_model": config.model,
        "variant": config.variant,
        "human_winner": row.human_winner,
    }
    # The original-order and swap-order calls are wrapped in *separate* try/excepts on purpose:
    # they're two independent, already-paid-for API calls, so a failure in the second one must
    # not discard a verdict the first one already returned successfully.
    try:
        prompt_ab = _build_prompt(row.question, row.answer_a, row.answer_b, config.variant)
        response_ab, stop_reason_ab = _call_judge(client, config, prompt_ab)
    except anthropic.APIError as exc:
        record.update(
            verdict_original_order=None,
            verdict_swapped_order=None,
            winner="error",
            error=str(exc),
            response_original_order=None,
            response_swapped_order=None,
            stop_reason_original_order=None,
            stop_reason_swapped_order=None,
        )
        return record

    verdict_ab = _parse_verdict(response_ab)
    verdict_ba = response_ba = stop_reason_ba = None
    error = None

    if config.swap_order:
        try:
            prompt_ba = _build_prompt(row.question, row.answer_b, row.answer_a, config.variant)
            response_ba, stop_reason_ba = _call_judge(client, config, prompt_ba)
            verdict_ba = _parse_verdict(response_ba)
            winner = _consolidate(verdict_ab, verdict_ba)
        except anthropic.APIError as exc:
            # The original-order verdict is still real data — keep it (below) rather than
            # discarding it just because its paired swap-order call failed.
            winner = "error"
            error = str(exc)
    else:
        winner = {"A": "model_a", "B": "model_b", "TIE": "tie"}.get(verdict_ab, "unparseable")

    record.update(
        verdict_original_order=verdict_ab,
        verdict_swapped_order=verdict_ba,
        winner=winner,
        error=error,
        response_original_order=response_ab,
        response_swapped_order=response_ba,
        stop_reason_original_order=stop_reason_ab,
        stop_reason_swapped_order=stop_reason_ba,
    )
    return record


def run_judges(
    items: pd.DataFrame,
    judge_configs: list[JudgeConfig],
    client: anthropic.Anthropic | None = None,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Score `items` (e.g. from `load_mt_bench_items`) with each configuration in `judge_configs`.

    Requires `ANTHROPIC_API_KEY` to be set (directly, or via a `.env` file) unless a pre-built
    `client` is passed in. A failed API call for one (item, config) pair is recorded as a row with
    `winner="error"` rather than aborting the whole run.

    Each (item, judge_config) pair is scored by `_score_one`, and those units of work are run
    concurrently across a thread pool (`max_workers`) — one Anthropic client shared across threads
    is the SDK's documented-safe pattern for concurrent requests (it's built on a connection-pooled
    HTTP client). Set `max_workers=1` for fully sequential, deterministic-order execution (e.g. in
    tests against a fake client whose canned responses are consumed in call order). Results come
    back in the same (config, item) order regardless of `max_workers`, since completion order and
    output order aren't the same thing here.

    Args:
        items: the battles to be judged — a DataFrame like `load_mt_bench_items`'s output. Must
            have columns question, answer_a, answer_b, question_id, model_a, model_b, human_winner.
        judge_configs: the judge model/prompt-variant configurations to run.
        client: an existing `anthropic.Anthropic` client (e.g. one built with a custom
            `max_retries`/`timeout`, or a test double); defaults to `anthropic.Anthropic()`.
        max_workers: number of concurrent judge calls in flight. Higher is faster wall-clock but
            more likely to trigger 429s (which the client already retries with backoff).

    Returns:
        A tidy table with one row per (item, judge_config): question_id, model_a, model_b,
        judge_name, judge_model, variant, verdict_original_order, verdict_swapped_order, winner
        (model_a/model_b/tie/"tie (inconsistent)"/unparseable/error), human_winner, error,
        response_original_order, response_swapped_order (raw response text), stop_reason_original_order,
        stop_reason_swapped_order (e.g. "end_turn" vs. "max_tokens" — the latter on an unparseable
        verdict means `max_tokens` cut the response off before the verdict line, not that the judge
        gave a genuinely unparseable answer).
    """
    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in, or pass "
                "an explicit `client=`."
            )
        client = anthropic.Anthropic(max_retries=5)

    jobs = [(config, row) for config in judge_configs for row in items.itertuples()]

    if max_workers == 1:
        rows = [_score_one(client, config, row) for config, row in jobs]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # executor.map preserves input order in its output, regardless of completion order.
            rows = list(
                executor.map(lambda job: _score_one(client, job[0], job[1]), jobs)
            )

    return pd.DataFrame(rows)
