#!/usr/bin/env python
"""Aggregate Task 10 anchor sensitivity per (model, prompt) from a torch MI run.

For each (model, prompt) condition, the script computes the per-question
accuracy drop under anchor-step interventions and under control-step
interventions, then reports the difference (anchor − control). A larger
positive value means suppressing reasoning at high-anchor steps damages the
final answer more than suppressing at randomly-chosen control steps, which is
the proposal's mechanistic signature of "this step mattered."

Inputs:
- <mi_dir>/baselines.jsonl     to recover baseline correctness per question
- <mi_dir>/interventions.jsonl to read per-step intervention outcomes

Output (CSV):
  model_key, prompt_name, n_questions,
  baseline_accuracy,
  anchor_mean_accuracy, control_mean_accuracy,
  anchor_drop, control_drop, anchor_minus_control_drop

Usage:
    python3 scripts/aggregate_anchor_sensitivity.py \\
        --mi-dir outputs/2026-05-05_xxxx_gsm8k50_qwen3-4b_torch_mech-slice \\
        --out results/task10_anchor_sensitivity.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def aggregate(mi_dir: Path) -> List[Dict[str, Any]]:
    baselines = read_jsonl(mi_dir / "baselines.jsonl")
    interventions_path = mi_dir / "interventions.jsonl"
    interventions: List[Dict[str, Any]] = (
        read_jsonl(interventions_path) if interventions_path.exists() else []
    )

    base_correct: Dict[Tuple[str, str, int], int] = {}
    for row in baselines:
        key = (str(row["model_key"]), str(row["prompt_name"]), int(row["sample_idx"]))
        base_correct[key] = int(bool(row.get("correct")))

    # Per (model, prompt, sample_idx, kind) collect the list of correctness
    # outcomes across intervention modes.
    per_question: Dict[Tuple[str, str, int], Dict[str, List[int]]] = defaultdict(
        lambda: {"anchor": [], "control": []}
    )
    for row in interventions:
        key = (
            str(row["model_key"]),
            str(row["prompt_name"]),
            int(row["sample_idx"]),
        )
        kind = str(row.get("step_kind", ""))
        if kind not in ("anchor", "control"):
            continue
        per_question[key][kind].append(int(bool(row.get("correct"))))

    by_condition: Dict[Tuple[str, str], List[Dict[str, float]]] = defaultdict(list)
    for key, buckets in per_question.items():
        model_key, prompt_name, sample_idx = key
        if not buckets["anchor"] and not buckets["control"]:
            continue
        anchor_acc = mean(buckets["anchor"]) if buckets["anchor"] else None
        control_acc = mean(buckets["control"]) if buckets["control"] else None
        baseline_corr = float(base_correct.get(key, 0))
        anchor_drop = (baseline_corr - anchor_acc) if anchor_acc is not None else None
        control_drop = (baseline_corr - control_acc) if control_acc is not None else None
        by_condition[(model_key, prompt_name)].append(
            {
                "sample_idx": sample_idx,
                "baseline_correct": baseline_corr,
                "anchor_mean_accuracy": anchor_acc if anchor_acc is not None else float("nan"),
                "control_mean_accuracy": control_acc if control_acc is not None else float("nan"),
                "anchor_drop": anchor_drop if anchor_drop is not None else float("nan"),
                "control_drop": control_drop if control_drop is not None else float("nan"),
            }
        )

    out_rows: List[Dict[str, Any]] = []
    for (model_key, prompt_name), per_q in sorted(by_condition.items()):
        n = len(per_q)
        baseline_acc = mean(row["baseline_correct"] for row in per_q) if n else None
        anchor_vals = [row["anchor_drop"] for row in per_q if row["anchor_drop"] == row["anchor_drop"]]
        control_vals = [row["control_drop"] for row in per_q if row["control_drop"] == row["control_drop"]]
        anchor_acc_vals = [
            row["anchor_mean_accuracy"]
            for row in per_q
            if row["anchor_mean_accuracy"] == row["anchor_mean_accuracy"]
        ]
        control_acc_vals = [
            row["control_mean_accuracy"]
            for row in per_q
            if row["control_mean_accuracy"] == row["control_mean_accuracy"]
        ]
        anchor_drop = mean(anchor_vals) if anchor_vals else None
        control_drop = mean(control_vals) if control_vals else None
        delta = (
            (anchor_drop - control_drop)
            if anchor_drop is not None and control_drop is not None
            else None
        )
        out_rows.append(
            {
                "model_key": model_key,
                "prompt_name": prompt_name,
                "n_questions": n,
                "baseline_accuracy": baseline_acc,
                "anchor_mean_accuracy": mean(anchor_acc_vals) if anchor_acc_vals else None,
                "control_mean_accuracy": mean(control_acc_vals) if control_acc_vals else None,
                "anchor_drop": anchor_drop,
                "control_drop": control_drop,
                "anchor_minus_control_drop": delta,
            }
        )
    return out_rows


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_key",
        "prompt_name",
        "n_questions",
        "baseline_accuracy",
        "anchor_mean_accuracy",
        "control_mean_accuracy",
        "anchor_drop",
        "control_drop",
        "anchor_minus_control_drop",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mi-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = aggregate(args.mi_dir)
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} condition row(s) to {args.out}.")


if __name__ == "__main__":
    main()
