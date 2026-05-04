"""Tests for the per-baseline entropy summary helper.

`summarize_entropy_over_generated` turns per-token rows into the two scalar
columns Task 6 cares about (mean predictive-distribution entropy across
generated tokens, plus a linear slope vs token position). A regression here
silently shifts the entropy column in the structured table.
"""

from __future__ import annotations


def _make_token_row(idx: int, *, is_generated: bool, is_special: bool, entropy):
    return {
        "token_index": idx,
        "token_id": idx,
        "token_str": f"t{idx}",
        "char_start_full": idx,
        "char_end_full": idx + 1,
        "char_start_generated": (idx if is_generated else None),
        "char_end_generated": (idx + 1 if is_generated else None),
        "is_special": is_special,
        "is_generated": is_generated,
        "is_reasoning": is_generated,
        "is_answer": False,
        "token_logprob": -1.0 if entropy is not None else None,
        "token_entropy": entropy,
    }


def test_returns_zero_count_when_no_generated_tokens(runner_module):
    rows = [_make_token_row(0, is_generated=False, is_special=False, entropy=0.5)]
    result = runner_module.summarize_entropy_over_generated(rows)
    assert result["mean_entropy_generated"] is None
    assert result["entropy_slope_generated"] is None
    assert result["num_entropy_tokens"] == 0


def test_special_and_prompt_tokens_are_excluded(runner_module):
    rows = [
        _make_token_row(0, is_generated=False, is_special=False, entropy=10.0),
        _make_token_row(1, is_generated=True, is_special=True, entropy=99.0),
        _make_token_row(2, is_generated=True, is_special=False, entropy=1.0),
        _make_token_row(3, is_generated=True, is_special=False, entropy=2.0),
    ]
    result = runner_module.summarize_entropy_over_generated(rows)
    assert result["num_entropy_tokens"] == 2
    assert result["mean_entropy_generated"] == 1.5


def test_slope_is_positive_when_entropy_rises_with_index(runner_module):
    rows = [
        _make_token_row(idx, is_generated=True, is_special=False, entropy=float(idx))
        for idx in range(10)
    ]
    result = runner_module.summarize_entropy_over_generated(rows)
    assert result["entropy_slope_generated"] > 0
    assert result["mean_entropy_generated"] == 4.5


def test_constant_entropy_slope_is_near_zero(runner_module):
    rows = [
        _make_token_row(idx, is_generated=True, is_special=False, entropy=0.7)
        for idx in range(5)
    ]
    result = runner_module.summarize_entropy_over_generated(rows)
    # polyfit on a constant signal returns a slope on the order of 1e-17
    # rather than exactly zero. Anything within a few epsilon is fine for
    # downstream aggregation.
    assert abs(result["entropy_slope_generated"]) < 1e-12
    assert result["mean_entropy_generated"] == 0.7


def test_none_entropy_rows_are_dropped(runner_module):
    rows = [
        _make_token_row(0, is_generated=True, is_special=False, entropy=None),
        _make_token_row(1, is_generated=True, is_special=False, entropy=2.0),
    ]
    result = runner_module.summarize_entropy_over_generated(rows)
    assert result["num_entropy_tokens"] == 1
    assert result["mean_entropy_generated"] == 2.0
