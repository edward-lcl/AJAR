"""Tests for the Task 6 aggregator's join logic and per-question metric math.

The aggregator joins four heterogeneous JSONL/CSV sources by
(model_key, prompt_name, original_sample_idx). When a join silently drops
a row the affected column ends up blank in the table, which is hard to
distinguish from "this metric was not collected for this sample". These
tests pin down the math and the audit warnings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _baseline_row(sample_idx: int, correct: bool, prediction: str, gen_secs: float = 1.0):
    return {
        "sample_idx": sample_idx,
        "model_key": "instruct",
        "model_name": "Qwen3-Instruct",
        "prompt_name": "neutral_strict",
        "question": f"q{sample_idx}",
        "gold_numeric": "18",
        "prediction_numeric": prediction,
        "correct": correct,
        "generated_len_tokens": 50,
        "generation_seconds": gen_secs,
        "num_steps": 5,
        "finish_reason": "stop",
    }


def test_perturbation_delta_is_signed_per_question(tmp_path: Path, aggregator_module):
    agg = aggregator_module

    baseline_dir = tmp_path / "baseline"
    perturb_dir = tmp_path / "perturb"
    perturb_index = tmp_path / "perturb_index.csv"
    _write_jsonl(
        baseline_dir / "baselines.jsonl",
        [
            _baseline_row(0, correct=True, prediction="18"),
            _baseline_row(1, correct=False, prediction="9"),
        ],
    )
    # Perturbation fixture row 0 is original-passthrough, row 1 is the perturbed
    # variant. The runner ran both as flat sample_idx 0 and 1.
    _write_jsonl(
        perturb_dir / "baselines.jsonl",
        [
            _baseline_row(0, correct=True, prediction="18"),
            _baseline_row(1, correct=False, prediction="11"),  # broke under distractor
            _baseline_row(2, correct=False, prediction="9"),
            _baseline_row(3, correct=False, prediction="7"),
        ],
    )
    _write_csv(
        perturb_index,
        [
            {"row_idx": 0, "original_sample_idx": 0, "variant_kind": "original",
             "variant_index": 0, "perturbation_op": "", "gold_numeric": "18"},
            {"row_idx": 1, "original_sample_idx": 0, "variant_kind": "perturbation",
             "variant_index": 1, "perturbation_op": "distractor_irrelevant",
             "gold_numeric": "18"},
            {"row_idx": 2, "original_sample_idx": 1, "variant_kind": "original",
             "variant_index": 0, "perturbation_op": "", "gold_numeric": "9"},
            {"row_idx": 3, "original_sample_idx": 1, "variant_kind": "perturbation",
             "variant_index": 1, "perturbation_op": "distractor_irrelevant",
             "gold_numeric": "9"},
        ],
    )

    rows = agg.build_table(
        baseline_dir=baseline_dir,
        paraphrase_dir=None,
        paraphrase_index=None,
        perturbation_dir=perturb_dir,
        perturbation_index=perturb_index,
        mi_dir=None,
        dataset_label="GSM8K",
        trial_id=1,
    )
    by_idx = {row["sample_idx"]: row for row in rows}
    # Sample 0 was correct, perturbed was wrong: Δ = 1 - 0 = +1
    assert by_idx[0]["perturbation_delta_accuracy"] == 1.0
    # Sample 1 was wrong, perturbed also wrong: Δ = 0 - 0 = 0
    assert by_idx[1]["perturbation_delta_accuracy"] == 0.0


def test_paraphrase_consistency_uses_original_prediction(tmp_path: Path, aggregator_module):
    agg = aggregator_module

    baseline_dir = tmp_path / "baseline"
    para_dir = tmp_path / "para"
    para_index = tmp_path / "para_index.csv"
    _write_jsonl(
        baseline_dir / "baselines.jsonl",
        [
            _baseline_row(0, correct=False, prediction="42"),  # wrong gold but consistent
        ],
    )
    _write_jsonl(
        para_dir / "baselines.jsonl",
        [
            # Row 0 = original passthrough (we ignore it)
            _baseline_row(0, correct=False, prediction="42"),
            # Row 1 = paraphrase 1, agrees with original prediction
            _baseline_row(1, correct=False, prediction="42"),
            # Row 2 = paraphrase 2, disagrees with original (different number)
            _baseline_row(2, correct=False, prediction="11"),
        ],
    )
    _write_csv(
        para_index,
        [
            {"row_idx": 0, "original_sample_idx": 0, "variant_kind": "original",
             "variant_index": 0, "paraphrase_style": "", "gold_numeric": "18",
             "numbers_preserved": "true"},
            {"row_idx": 1, "original_sample_idx": 0, "variant_kind": "paraphrase",
             "variant_index": 1, "paraphrase_style": "narrative", "gold_numeric": "18",
             "numbers_preserved": "true"},
            {"row_idx": 2, "original_sample_idx": 0, "variant_kind": "paraphrase",
             "variant_index": 2, "paraphrase_style": "concise", "gold_numeric": "18",
             "numbers_preserved": "true"},
        ],
    )

    rows = agg.build_table(
        baseline_dir=baseline_dir,
        paraphrase_dir=para_dir,
        paraphrase_index=para_index,
        perturbation_dir=None,
        perturbation_index=None,
        mi_dir=None,
        dataset_label="GSM8K",
        trial_id=1,
    )
    assert len(rows) == 1
    # Two paraphrases, one agrees with original prediction (42), one disagrees.
    assert rows[0]["paraphrase_consistency"] == 0.5
    # Both paraphrases are wrong vs gold 18.
    assert rows[0]["paraphrase_accuracy"] == 0.0
    assert rows[0]["paraphrase_count"] == 2


def test_mechanistic_delta_is_anchor_minus_control_drop(tmp_path: Path, aggregator_module):
    agg = aggregator_module

    baseline_dir = tmp_path / "baseline"
    mi_dir = tmp_path / "mi"
    _write_jsonl(
        baseline_dir / "baselines.jsonl",
        [_baseline_row(0, correct=True, prediction="18")],
    )
    _write_jsonl(mi_dir / "baselines.jsonl",
                 [_baseline_row(0, correct=True, prediction="18")])
    intervention_rows = [
        # Two anchor interventions; both broke the answer.
        {"sample_idx": 0, "model_key": "instruct", "prompt_name": "neutral_strict",
         "step_kind": "anchor", "step_index": 5, "intervention_mode": "residual_zero",
         "correct": False, "prediction_numeric": "0"},
        {"sample_idx": 0, "model_key": "instruct", "prompt_name": "neutral_strict",
         "step_kind": "anchor", "step_index": 7, "intervention_mode": "residual_zero",
         "correct": False, "prediction_numeric": "0"},
        # Two control interventions; one broke, one survived.
        {"sample_idx": 0, "model_key": "instruct", "prompt_name": "neutral_strict",
         "step_kind": "control", "step_index": 2, "intervention_mode": "residual_zero",
         "correct": True, "prediction_numeric": "18"},
        {"sample_idx": 0, "model_key": "instruct", "prompt_name": "neutral_strict",
         "step_kind": "control", "step_index": 3, "intervention_mode": "residual_zero",
         "correct": False, "prediction_numeric": "5"},
    ]
    _write_jsonl(mi_dir / "interventions.jsonl", intervention_rows)

    rows = agg.build_table(
        baseline_dir=baseline_dir,
        paraphrase_dir=None,
        paraphrase_index=None,
        perturbation_dir=None,
        perturbation_index=None,
        mi_dir=mi_dir,
        dataset_label="GSM8K",
        trial_id=1,
    )
    row = rows[0]
    # baseline_correct = 1; anchor_acc = 0; control_acc = 0.5
    # anchor_drop = 1; control_drop = 0.5 -> delta = 0.5
    assert row["mechanistic_anchor_accuracy"] == 0.0
    assert row["mechanistic_control_accuracy"] == 0.5
    assert row["mechanistic_intervention_delta_accuracy"] == 0.5
    assert row["mechanistic_available"] == 1


def test_audit_warns_on_orphan_index_rows(
    tmp_path: Path,
    aggregator_module,
    capsys: pytest.CaptureFixture[str],
):
    agg = aggregator_module

    baseline_dir = tmp_path / "baseline"
    para_dir = tmp_path / "para"
    para_index = tmp_path / "para_index.csv"
    _write_jsonl(baseline_dir / "baselines.jsonl",
                 [_baseline_row(0, correct=True, prediction="18")])
    # Index references rows 0..2 but the run only produced rows 0 and 1.
    _write_jsonl(para_dir / "baselines.jsonl", [
        _baseline_row(0, correct=False, prediction="42"),
        _baseline_row(1, correct=False, prediction="42"),
    ])
    _write_csv(
        para_index,
        [
            {"row_idx": 0, "original_sample_idx": 0, "variant_kind": "original",
             "variant_index": 0, "paraphrase_style": "", "gold_numeric": "18",
             "numbers_preserved": "true"},
            {"row_idx": 1, "original_sample_idx": 0, "variant_kind": "paraphrase",
             "variant_index": 1, "paraphrase_style": "narrative", "gold_numeric": "18",
             "numbers_preserved": "true"},
            {"row_idx": 2, "original_sample_idx": 0, "variant_kind": "paraphrase",
             "variant_index": 2, "paraphrase_style": "concise", "gold_numeric": "18",
             "numbers_preserved": "true"},
        ],
    )

    agg.build_table(
        baseline_dir=baseline_dir,
        paraphrase_dir=para_dir,
        paraphrase_index=para_index,
        perturbation_dir=None,
        perturbation_index=None,
        mi_dir=None,
        dataset_label="GSM8K",
        trial_id=1,
    )
    err = capsys.readouterr().err
    assert "paraphrase index row" in err
