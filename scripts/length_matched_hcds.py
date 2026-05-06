#!/usr/bin/env python
"""Length-matched HCDS: does the headline signal survive when we
stratify questions by output length?

The reviewer concern: HCDS shows neutral_strict resembles explicit_cot
more than explicit_no_cot, but neutral_strict outputs are also much
longer than explicit_no_cot outputs (300+ tokens vs 16 tokens for
Instruct). Maybe what we're measuring is "neutral_strict is verbose
like CoT" rather than "neutral_strict is reasoning like CoT". A
length-matched analysis checks whether the signal holds within
output-length tiers, where verbosity is held roughly constant.

Approach:
  1. For each model, take the per-question output_tokens under each
     prompt condition.
  2. Bin questions by the explicit_cot output length (the longest
     condition by construction). Three equal-frequency tiers:
     short / medium / long.
  3. Within each tier, run the same per-question HCDS computation
     used for the headline result, and report the mean / 95% bootstrap
     CI / paired t-test against zero.
  4. If HCDS is positive AND significant in each tier, the signal is
     not driven by output length differences.

Outputs:
  <out-dir>/hcds_length_matched.csv  per-tier-per-model summary
  <out-dir>/hcds_length_matched.md   human-readable report

Usage:
  python3 scripts/length_matched_hcds.py \\
      --task6-csv results/runs/<run_id>/task6_table.csv \\
      --out-dir   results/runs/<run_id>
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# Reuse the helpers from compute_hcds without duplicating logic.
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("compute_hcds", HERE / "compute_hcds.py")
compute_hcds = importlib.util.module_from_spec(spec)
sys.modules["compute_hcds"] = compute_hcds
spec.loader.exec_module(compute_hcds)


def to_float(v):
    return compute_hcds.to_float(v)


def per_question_output_tokens(rows):
    """Return {(model, sample_idx): {prompt: output_tokens}} mapping."""
    out = defaultdict(dict)
    for r in rows:
        toks = to_float(r.get("output_tokens"))
        if toks is None:
            continue
        out[(r["model"], int(r["sample_idx"]))][r["prompt_condition"]] = toks
    return out


def length_tier_for_question(
    cot_tokens: float, tier_thresholds: Tuple[float, float]
) -> str:
    """Map an output-tokens count under explicit_cot to short/med/long."""
    lo, hi = tier_thresholds
    if cot_tokens <= lo:
        return "short"
    if cot_tokens <= hi:
        return "medium"
    return "long"


def compute_tiered_hcds(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict]:
    """Per-(model, tier) HCDS with bootstrap CI and paired t-test."""
    z_by_key = compute_hcds.zscore_per_model(rows)
    pq_all = compute_hcds.per_question_hcds(rows, z_by_key)

    # For each question, look up its explicit_cot output_tokens to assign a tier.
    tokens_lookup = per_question_output_tokens(rows)

    # Compute per-model tier thresholds from tertiles of explicit_cot tokens.
    by_model_cot: Dict[str, List[float]] = defaultdict(list)
    for (model, _idx), prompt_tokens in tokens_lookup.items():
        cot = prompt_tokens.get("explicit_cot")
        if cot is not None:
            by_model_cot[model].append(cot)
    thresholds: Dict[str, Tuple[float, float]] = {}
    for model, cot_lengths in by_model_cot.items():
        arr = np.asarray(cot_lengths)
        if arr.size < 3:
            thresholds[model] = (float("inf"), float("inf"))
            continue
        thresholds[model] = (
            float(np.percentile(arr, 33.3)),
            float(np.percentile(arr, 66.6)),
        )

    # Bucket per-question HCDS rows into tiers and aggregate.
    by_bucket: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    bucket_token_means: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in pq_all:
        model = str(row["model"])
        sample_idx = int(row["sample_idx"])
        cot_tokens = tokens_lookup.get((model, sample_idx), {}).get("explicit_cot")
        if cot_tokens is None:
            continue
        tier = length_tier_for_question(cot_tokens, thresholds[model])
        by_bucket[(model, tier)].append(float(row["hcds_q"]))
        bucket_token_means[(model, tier)].append(cot_tokens)

    results: Dict[Tuple[str, str], Dict] = {}
    for key, hcds_values in by_bucket.items():
        n = len(hcds_values)
        if n < 2:
            continue
        mean_hcds = statistics.mean(hcds_values)
        ci_low, ci_high = compute_hcds.bootstrap_ci(hcds_values, n_boot=1000)
        t_stat, p_value = stats.ttest_1samp(hcds_values, popmean=0.0)
        mean_cot_tokens = statistics.mean(bucket_token_means[key])
        verdict = (
            "neutral aligns with CoT (p < 0.05)"
            if mean_hcds > 0 and p_value < 0.05
            else "neutral aligns with NoCoT (p < 0.05)"
            if mean_hcds < 0 and p_value < 0.05
            else "ambiguous"
        )
        results[key] = {
            "model": key[0],
            "length_tier": key[1],
            "n": n,
            "mean_cot_tokens": mean_cot_tokens,
            "mean_hcds": mean_hcds,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "verdict": verdict,
        }
    return results, thresholds


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


TIER_ORDER = ["short", "medium", "long"]


def write_report(
    path: Path,
    results: Dict[Tuple[str, str], Dict],
    thresholds: Dict[str, Tuple[float, float]],
) -> None:
    lines: List[str] = []
    lines.append("# HCDS length-matched analysis\n")
    lines.append(
        "Robustness check on the headline HCDS result. We bin questions "
        "into three equal-frequency tiers by their explicit_cot output "
        "length, then re-run the per-question HCDS computation within "
        "each tier. If HCDS is positive AND significant in each tier, "
        "the signal cannot be a pure length-driven artefact.\n"
    )

    for model, (lo, hi) in sorted(thresholds.items()):
        lines.append(f"## {model}\n")
        lines.append(
            f"Length tier thresholds (explicit_cot output tokens): "
            f"short ≤ {lo:.0f}, medium ≤ {hi:.0f}, long > {hi:.0f}.\n"
        )
        lines.append("| tier | n | mean cot tokens | HCDS | 95% CI | t | p | verdict |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for tier in TIER_ORDER:
            row = results.get((model, tier))
            if row is None:
                lines.append(f"| {tier} | — | — | — | — | — | — | (no rows) |")
                continue
            ci = f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"
            lines.append(
                f"| {tier} | {row['n']} | {row['mean_cot_tokens']:.0f} | "
                f"{row['mean_hcds']:+.3f} | {ci} | "
                f"{row['t_stat']:+.2f} | {row['p_value']:.3e} | "
                f"{row['verdict']} |"
            )
        lines.append("")

    lines.append("## Interpretation\n")
    lines.append(
        "If every tier under a given model shows positive significant "
        "HCDS, the result is robust to length: even when we constrain the "
        "comparison to questions where neutral_strict and explicit_cot "
        "have similar output lengths, neutral_strict still aligns with "
        "explicit_cot more than with explicit_no_cot.\n"
    )
    lines.append(
        "If a tier crosses zero or has p ≥ 0.05, that's a sign the headline "
        "signal at that length range is fragile; combined with the deep-table "
        "feature-stability check (`hcds_feature_stability.md`), the team can "
        "decide which subsets to report as load-bearing in the paper.\n"
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task6-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = compute_hcds.load_rows(args.task6_csv)
    results, thresholds = compute_tiered_hcds(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "hcds_length_matched.csv"
    md_path = args.out_dir / "hcds_length_matched.md"
    rows_out = []
    for tier in TIER_ORDER:
        for model in sorted({m for (m, _) in results.keys()}):
            row = results.get((model, tier))
            if row is not None:
                rows_out.append(row)
    write_csv(csv_path, rows_out)
    write_report(md_path, results, thresholds)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print()
    for model, (lo, hi) in sorted(thresholds.items()):
        print(f"{model}: tier thresholds short<={lo:.0f} medium<={hi:.0f} long>{hi:.0f}")
        for tier in TIER_ORDER:
            row = results.get((model, tier))
            if row is None:
                continue
            print(
                f"  {tier:6} n={row['n']:2} HCDS={row['mean_hcds']:+.3f} "
                f"CI=[{row['ci_low']:+.3f},{row['ci_high']:+.3f}] "
                f"p={row['p_value']:.3e} {row['verdict']}"
            )


if __name__ == "__main__":
    main()
