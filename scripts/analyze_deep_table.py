#!/usr/bin/env python
"""Print a per-condition summary of the Task 6 deep table.

Reports mean values for every column the runner populates, plus a few
derived observations (HCDS-style feature distances, anchor sensitivity).
Designed to be the first analysis pass after a deep run completes.

Usage:
    python3 scripts/analyze_deep_table.py \\
        --task6-csv results/runs/<run_id>/task6_table.csv \\
        --task10-csv results/runs/<run_id>/task10_anchor_sensitivity.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def to_float(v):
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def safe_mean(vs):
    return statistics.mean(vs) if vs else None


def safe_stdev(vs):
    if len(vs) < 2:
        return None
    return statistics.stdev(vs)


def fmt(x, prec=4, width=8):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return f"{'-':>{width}}"
    return f"{x:>{width}.{prec}f}"


def per_condition_summary(rows):
    agg = defaultdict(list)
    for r in rows:
        agg[(r["model"], r["prompt_condition"])].append(r)

    print("Per-condition means (n = 50 questions per condition):")
    print()
    header = (
        f"{'condition':<35} {'acc':>5} {'lat/tok':>8} {'tokens':>6} "
        f"{'H_mean':>7} {'H_slope':>9} {'para_C':>7} {'para_acc':>8} "
        f"{'pert_dA':>8} {'mech_dA':>8}"
    )
    print(header)
    print("-" * len(header))

    summary = {}
    for key, lst in sorted(agg.items()):
        cols = {
            "accuracy": [to_float(r["accuracy"]) for r in lst],
            "latency_per_output_token": [
                to_float(r["latency_per_output_token"]) for r in lst
            ],
            "output_tokens": [to_float(r["output_tokens"]) for r in lst],
            "token_entropy_mean": [
                to_float(r["token_entropy_mean"]) for r in lst
            ],
            "token_entropy_slope": [
                to_float(r["token_entropy_slope"]) for r in lst
            ],
            "paraphrase_consistency": [
                to_float(r["paraphrase_consistency"]) for r in lst
            ],
            "paraphrase_accuracy": [
                to_float(r["paraphrase_accuracy"]) for r in lst
            ],
            "perturbation_delta_accuracy": [
                to_float(r["perturbation_delta_accuracy"]) for r in lst
            ],
            "mechanistic_intervention_delta_accuracy": [
                to_float(r["mechanistic_intervention_delta_accuracy"]) for r in lst
            ],
        }
        cleaned = {k: [v for v in vs if v is not None] for k, vs in cols.items()}
        means = {k: safe_mean(vs) for k, vs in cleaned.items()}
        summary[key] = means

        cond = f"{key[0]}/{key[1]}"
        print(
            f"{cond:<35} "
            f"{fmt(means['accuracy'], 3, 5)} "
            f"{fmt(means['latency_per_output_token'], 4, 8)} "
            f"{fmt(means['output_tokens'], 0, 6)} "
            f"{fmt(means['token_entropy_mean'], 4, 7)} "
            f"{fmt(means['token_entropy_slope'], 5, 9)} "
            f"{fmt(means['paraphrase_consistency'], 3, 7)} "
            f"{fmt(means['paraphrase_accuracy'], 3, 8)} "
            f"{fmt(means['perturbation_delta_accuracy'], 3, 8)} "
            f"{fmt(means['mechanistic_intervention_delta_accuracy'], 3, 8)}"
        )
    return summary


HCDS_FEATURES = [
    "latency_per_output_token",
    "token_entropy_mean",
    "token_entropy_slope",
    "paraphrase_consistency",
    "perturbation_delta_accuracy",
    "mechanistic_intervention_delta_accuracy",
]


def hcds(summary):
    """Compute a simple HCDS-style score per model.

    Per the proposal:
      HCDS = D(Neutral, NoCoT) - D(Neutral, CoT)
      D = Euclidean distance in z-scored feature space.

    Higher HCDS => Neutral behaves more like CoT than like NoCoT, which is
    the proposal's signature of latent CoT use under neutral prompting.

    We z-score per-feature across the three conditions of a single model
    so each feature is on a comparable scale before computing distances.
    """
    print()
    print("=" * 80)
    print("HCDS-style summary per model:")
    print()
    print(f"{'model':<12} {'D(N,NoCoT)':>12} {'D(N,CoT)':>12} {'HCDS':>10} {'verdict':<28}")
    print("-" * 80)

    by_model = defaultdict(dict)
    for (model, prompt), means in summary.items():
        by_model[model][prompt] = means

    rows = []
    for model, prompts in sorted(by_model.items()):
        if not all(p in prompts for p in ("explicit_cot", "explicit_no_cot", "neutral_strict")):
            print(f"{model:<12} skipped (missing prompts: have {list(prompts)})")
            continue
        # Build feature matrix: rows=conditions, cols=features
        condition_keys = ["explicit_cot", "explicit_no_cot", "neutral_strict"]
        feature_vectors = {}
        for cond in condition_keys:
            vec = [prompts[cond].get(f) for f in HCDS_FEATURES]
            feature_vectors[cond] = vec

        # Z-score per feature across the three conditions.
        n_features = len(HCDS_FEATURES)
        z = {cond: [None] * n_features for cond in condition_keys}
        for fi in range(n_features):
            vals = [feature_vectors[c][fi] for c in condition_keys]
            non_null = [v for v in vals if v is not None]
            if len(non_null) < 2:
                continue
            mean_v = statistics.mean(non_null)
            std_v = statistics.stdev(non_null)
            if std_v == 0:
                continue
            for c in condition_keys:
                if feature_vectors[c][fi] is not None:
                    z[c][fi] = (feature_vectors[c][fi] - mean_v) / std_v

        def euclid(a, b):
            sq = []
            for ai, bi in zip(a, b):
                if ai is None or bi is None:
                    continue
                sq.append((ai - bi) ** 2)
            return math.sqrt(sum(sq)) if sq else None

        d_n_nocot = euclid(z["neutral_strict"], z["explicit_no_cot"])
        d_n_cot = euclid(z["neutral_strict"], z["explicit_cot"])
        if d_n_nocot is None or d_n_cot is None:
            print(f"{model:<12} could not compute (insufficient feature coverage)")
            continue
        hcds_value = d_n_nocot - d_n_cot
        if hcds_value > 0.3:
            verdict = "neutral aligns with CoT"
        elif hcds_value < -0.3:
            verdict = "neutral aligns with NoCoT"
        else:
            verdict = "ambiguous (|HCDS| < 0.3)"
        print(
            f"{model:<12} {fmt(d_n_nocot, 3, 12)} {fmt(d_n_cot, 3, 12)} "
            f"{fmt(hcds_value, 3, 10)}  {verdict}"
        )
        rows.append({"model": model, "d_n_nocot": d_n_nocot, "d_n_cot": d_n_cot, "hcds": hcds_value})

    print()
    print("Note: HCDS is descriptive only at this scale (n=50). Bootstrap CIs and")
    print("paired tests (Task 8) needed before declaring significance.")
    return rows


def per_question_anchor_signal(rows):
    """Quick distribution check: how often does anchor_drop > control_drop?"""
    print()
    print("=" * 80)
    print("Per-question anchor-vs-control sign distribution:")
    print()
    by_cond = defaultdict(list)
    for r in rows:
        m = to_float(r.get("mechanistic_intervention_delta_accuracy"))
        if m is None:
            continue
        by_cond[(r["model"], r["prompt_condition"])].append(m)
    print(f"{'condition':<35} {'n':>4} {'> 0':>6} {'== 0':>6} {'< 0':>6} {'mean':>8}")
    print("-" * 70)
    for key in sorted(by_cond.keys()):
        vals = by_cond[key]
        pos = sum(1 for v in vals if v > 0)
        zero = sum(1 for v in vals if v == 0)
        neg = sum(1 for v in vals if v < 0)
        print(
            f"{key[0]}/{key[1]:<28} {len(vals):>4} {pos:>6} {zero:>6} {neg:>6} "
            f"{fmt(safe_mean(vals), 3, 8)}"
        )


def task10_summary(path):
    if not Path(path).exists():
        print(f"\n(skip) {path} does not exist")
        return
    print()
    print("=" * 80)
    print("Task 10 (anchor sensitivity) raw:")
    print()
    with open(path) as f:
        for line in f:
            print(line.rstrip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--task6-csv",
        type=Path,
        default=Path("results/runs/2026-05-04_2232_gsm8k50_qwen3-4b_omlx_baseline-rerun-nocot-1024/task6_starter_table.csv"),
    )
    ap.add_argument("--task10-csv", type=Path, default=None)
    args = ap.parse_args()

    with args.task6_csv.open() as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} rows from {args.task6_csv}")
    print()
    summary = per_condition_summary(rows)
    hcds(summary)
    per_question_anchor_signal(rows)
    if args.task10_csv:
        task10_summary(args.task10_csv)


if __name__ == "__main__":
    main()
