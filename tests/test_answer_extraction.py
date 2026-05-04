"""Tests for the answer-extraction parser.

Cover the cases that have actually appeared in the data so far: boxed
answers, "Final answer: X" labels, naked numerical answers, and Qwen3
Thinking model <think>...</think> blocks. When a regression happens here,
accuracy in the run summaries silently moves and is hard to debug.
"""

from __future__ import annotations


def test_boxed_answer_takes_precedence(runner_module):
    text = "She earns 9 * 2 = $18 every day. \\boxed{18}"
    pred, boxed = runner_module.extract_predicted_number(text)
    assert pred == "18"
    assert boxed == "18"


def test_boxed_handles_dollar_sign_and_commas(runner_module):
    text = "The profit is \\boxed{$70,000}."
    pred, boxed = runner_module.extract_predicted_number(text)
    assert pred == "70000"
    assert boxed == "$70,000"


def test_label_pattern_beats_last_number_fallback(runner_module):
    text = (
        "Step 1: 80000 + 50000 = 130000.\n"
        "Step 2: 130000 * 2.5 = 325000 new value.\n"
        "Final answer: 195000"
    )
    pred, boxed = runner_module.extract_predicted_number(text)
    assert pred == "195000"
    assert boxed is None


def test_thinking_block_is_skipped(runner_module):
    text = (
        "<think>Let me work this out. 16 - 3 - 4 = 9 eggs left to sell. "
        "9 * 2 = 18 dollars. Wait, double-check: 9 * 2 is 18.</think>\n"
        "She makes 18 dollars."
    )
    pred, _ = runner_module.extract_predicted_number(text)
    assert pred == "18"


def test_thinking_block_internal_numbers_not_picked_when_label_present(runner_module):
    text = "<think>Try 100, 200, 300...</think>\nAnswer: 540"
    pred, _ = runner_module.extract_predicted_number(text)
    assert pred == "540"


def test_returns_none_when_no_number(runner_module):
    text = "I cannot answer that."
    pred, boxed = runner_module.extract_predicted_number(text)
    assert pred is None
    assert boxed is None


def test_normalize_handles_decimals_and_commas(runner_module):
    assert runner_module.normalize_number_str("1,234") == "1234"
    assert runner_module.normalize_number_str("$8.33") == "8.33"
    assert runner_module.normalize_number_str("18.0") == "18"
    assert runner_module.normalize_number_str("  ") is None


def test_parse_gsm8k_answer_extracts_after_hash_marker(runner_module):
    answer = (
        "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 eggs.\n"
        "She makes 9 * 2 = $<<9*2=18>>18 every day.\n"
        "#### 18"
    )
    assert runner_module.parse_gsm8k_answer(answer) == "18"
