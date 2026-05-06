#!/usr/bin/env python
"""Compute per-question HCDS, bootstrap CIs, and a paired t-test.

HCDS (Hidden CoT Detection Score) per question q for a given model is:

    HCDS_q = D(f_neutral_q, f_no_cot_q) - D(f_neutral_q, f_cot_q)

where f_<prompt>_q is the feature vector for question q under prompt
<prompt>, and D is Euclidean distance in z-scored feature space. A
positive HCDS_q means neutral behaviour on that question is closer to
the explicit-CoT condition than to the explicit-no-CoT condition,
which is the proposal's signature of latent CoT use.

Methodology decisions (auditable, defensible, and explicit so the team
can adjust if needed):

1. Z-scoring is done per-model across all 150 rows (3 prompts x 50
   questions) for that model. This puts every feature on a comparable
   scale within a model while preserving cross-prompt differences. We
   do not z-score across both models together because the two variants
   have very different baseline behaviour (Thinking outputs ~3x the
   tokens of Instruct) and pooled z-scoring would dilute the signal.

2. Distance is Euclidean. We considered cosine but Euclidean is
   simpler to interpret and the proposal does not specify; we report
   both per-feature and aggregate distances so the choice is auditable.

3. Missing-data handling: a per-question HCDS is computed over only
   the features that are populated for ALL THREE prompts for that
   question. Features that are missing in any of the three prompts
   are dropped for that question only (not for others). This trades
   slightly varying feature dimensionality across questions for not
   losing entire questions. In practice this affects ~46/300 rows
   where explicit_no_cot produced no reasoning anchors, and ~30
   paraphrase rows that were filtered for numbers_preserved=false.

4. Bootstrap is over questions, not over feature values: we resample
   the 50 questions with replacement 1000 times and recompute the
   per-question HCDS mean for each resample. Reported intervals are
   2.5%/97.5% percentiles. This treats questions as the sampling unit,
   which is appropriate because they are the unit of independent
   variation in this dataset.

5. The paired t-test is one-sample on the 50 per-question HCDS values
   against zero, which is mathematically equivalent to the paired test
   of D(N,NoCoT) vs D(N,CoT) across questions. Reports two-sided p.

Outputs:
  <out_dir>/hcds_per_question.csv
      Wide table: one row per (model, original_sample_idx) with
      d_n_nocot, d_n_cot, hcds_q, plus a column per feature pair
      indicating whether that feature was usable for that question.
  <out_dir>/hcds_summary.csv
      One row per model: mean HCDS, 95% CI (bootstrap), t-stat, p-value,
      n_questions, n_features_avg.

Usage:
  python3 scripts/compute_hcds.py \\
      --task6-csv results/runs/<run_id>/task6_table.csv \\
      --out-dir   results/runs/<run_id>/
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


# Feature columns that go into the HCDS feature vector. Listed here so the
# team can drop any individual feature with a single edit and re-run to
# check robustness. The order does not matter for distance computation.
HCDS_FEATURES: List[str] = [
    "latency_per_output_token",
    "token_entropy_mean",
    "token_entropy_slope",
    "paraphrase_consistency",
    "perturbation_delta_accuracy",
    "mechanistic_intervention_delta_accuracy",
]


REQUIRED_PROMPTS = ("explicit_cot", "explicit_no_cot", "neutral_strict")


def to_float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        f = float(value)
    except (ValueError, TypeError):
        return None
    if math.isnan(f):
        return None
    return f


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def zscore_per_model(
    rows: List[Dict[str, str]],
) -> Dict[Tuple[str, str, int], Dict[str, Optional[float]]]:
    """Return per-(model, prompt, sample_idx) z-scored feature dict.

    z-score is computed per model across all 150 of that model's rows.
    """
    by_model: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    z_by_key: Dict[Tuple[str, str, int], Dict[str, Optional[float]]] = {}
    for model, model_rows in by_model.items():
        # Compute per-feature mean/std using only non-null values.
        stats_by_feature: Dict[str, Tuple[float, float]] = {}
        for feat in HCDS_FEATURES:
            vals = [to_float(r.get(feat)) for r in model_rows]
            non_null = [v for v in vals if v is not None]
            if len(non_null) < 2:
                stats_by_feature[feat] = (0.0, 0.0)
                continue
            mu = statistics.mean(non_null)
            sigma = statistics.stdev(non_null)
            stats_by_feature[feat] = (mu, sigma)
        for r in model_rows:
            zvec: Dict[str, Optional[float]] = {}
            for feat in HCDS_FEATURES:
                raw = to_float(r.get(feat))
                if raw is None:
                    zvec[feat] = None
                    continue
                mu, sigma = stats_by_feature[feat]
                zvec[feat] = (raw - mu) / sigma if sigma > 0 else 0.0
            z_by_key[(model, r["prompt_condition"], int(r["sample_idx"]))] = zvec
    return z_by_key


def euclidean_partial(
    a: Dict[str, Optional[float]],
    b: Dict[str, Optional[float]],
) -> Tuple[Optional[float], int]:
    """Euclidean distance over the features populated in both vectors.

    Returns the distance and the number of features used. If no features
    are populated in both, returns (None, 0).
    """
    sq_sum = 0.0
    n_used = 0
    for feat in HCDS_FEATURES:
        av = a.get(feat)
        bv = b.get(feat)
        if av is None or bv is None:
            continue
        sq_sum += (av - bv) ** 2
        n_used += 1
    if n_used == 0:
        return None, 0
    return math.sqrt(sq_sum), n_used


def per_question_hcds(
    rows: List[Dict[str, str]],
    z_by_key: Dict[Tuple[str, str, int], Dict[str, Optional[float]]],
) -> List[Dict[str, object]]:
    """Compute HCDS_q per (model, sample_idx).

    For each question we need all three prompt conditions to be present
    in the data. If any one is missing (which shouldn't happen in our
    dataset but is defensive) we skip that question for that model.
    """
    seen_keys = set(z_by_key.keys())
    by_model_sample: Dict[Tuple[str, int], Dict[str, Dict[str, Optional[float]]]] = (
        defaultdict(dict)
    )
    for (model, prompt, sample_idx), zvec in z_by_key.items():
        by_model_sample[(model, sample_idx)][prompt] = zvec

    out: List[Dict[str, object]] = []
    for (model, sample_idx), prompt_dict in sorted(by_model_sample.items()):
        if not all(p in prompt_dict for p in REQUIRED_PROMPTS):
            continue
        d_n_nocot, n_nocot = euclidean_partial(
            prompt_dict["neutral_strict"], prompt_dict["explicit_no_cot"]
        )
        d_n_cot, n_cot = euclidean_partial(
            prompt_dict["neutral_strict"], prompt_dict["explicit_cot"]
        )
        if d_n_nocot is None or d_n_cot is None:
            continue
        hcds_q = d_n_nocot - d_n_cot
        out.append(
            {
                "model": model,
                "sample_idx": sample_idx,
                "d_n_nocot": d_n_nocot,
                "d_n_cot": d_n_cot,
                "hcds_q": hcds_q,
                "n_features_used_n_nocot": n_nocot,
                "n_features_used_n_cot": n_cot,
            }
        )
    return out


def bootstrap_ci(
    values: List[float],
    n_boot: int = 1000,
    seed: int = 17,
    ci: float = 0.95,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values)
    if arr.size == 0:
        return float("nan"), float("nan")
    boots = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    low = float(np.percentile(boots, (1 - ci) / 2 * 100))
    high = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return low, high


def model_summary(per_question_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_model: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in per_question_rows:
        by_model[str(row["model"])].append(row)

    summaries: List[Dict[str, object]] = []
    for model, model_rows in sorted(by_model.items()):
        hcds_values = [float(r["hcds_q"]) for r in model_rows]
        n = len(hcds_values)
        if n < 2:
            continue
        mean_hcds = statistics.mean(hcds_values)
        ci_low, ci_high = bootstrap_ci(hcds_values, n_boot=1000)
        # One-sample t-test against zero. Two-sided.
        t_stat, p_value = stats.ttest_1samp(hcds_values, popmean=0.0)
        feat_avg = statistics.mean(
            float(r["n_features_used_n_nocot"]) + float(r["n_features_used_n_cot"])
            for r in model_rows
        ) / 2.0
        verdict = (
            "neutral aligns with CoT (HCDS > 0, p < 0.05)"
            if mean_hcds > 0 and p_value < 0.05
            else "neutral aligns with NoCoT (HCDS < 0, p < 0.05)"
            if mean_hcds < 0 and p_value < 0.05
            else "ambiguous (CI crosses zero or p >= 0.05)"
        )
        summaries.append(
            {
                "model": model,
                "n_questions": n,
                "mean_hcds": mean_hcds,
                "ci_low_95": ci_low,
                "ci_high_95": ci_high,
                "t_statistic": float(t_stat),
                "p_value_two_sided": float(p_value),
                "n_features_avg": feat_avg,
                "verdict": verdict,
            }
        )
    return summaries


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
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


def print_summary(summaries: List[Dict[str, object]]) -> None:
    print()
    print("HCDS — per-model summary")
    print()
    print(
        f"{'model':<10} {'n':>3} {'mean':>7} {'95% CI':>20} {'t':>7} {'p':>10} {'verdict':<48}"
    )
    print("-" * 110)
    for s in summaries:
        ci = f"[{s['ci_low_95']:+.3f}, {s['ci_high_95']:+.3f}]"
        print(
            f"{s['model']:<10} {s['n_questions']:>3} {s['mean_hcds']:>+7.3f} {ci:>20} "
            f"{s['t_statistic']:>+7.2f} {s['p_value_two_sided']:>10.3e} {s['verdict']:<48}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task6-csv", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for hcds_per_question.csv and hcds_summary.csv.",
    )
    parser.add_argument(
        "--exclude-features",
        default="",
        help=(
            "Comma-separated list of features to drop from the HCDS feature "
            "vector for this run. Useful for robustness checks (e.g. drop "
            "paraphrase_consistency to see if the result holds without it). "
            "Default: empty (all 6 features)."
        ),
    )
    parser.add_argument(
        "--label",
        default="",
        help=(
            "Optional label appended to output filenames so multiple feature-"
            "subset runs can coexist in the same out-dir. e.g. label='no_para' "
            "writes hcds_summary_no_para.csv."
        ),
    )
    args = parser.parse_args()

    excluded = {s.strip() for s in args.exclude_features.split(",") if s.strip()}
    if excluded:
        global HCDS_FEATURES
        unknown = excluded - set(HCDS_FEATURES)
        if unknown:
            raise SystemExit(
                f"Unknown features in --exclude-features: {sorted(unknown)}. "
                f"Valid features: {HCDS_FEATURES}"
            )
        HCDS_FEATURES = [f for f in HCDS_FEATURES if f not in excluded]
        print(f"Excluded features: {sorted(excluded)}")
        print(f"Active features ({len(HCDS_FEATURES)}): {HCDS_FEATURES}")

    rows = load_rows(args.task6_csv)
    print(f"Loaded {len(rows)} rows from {args.task6_csv}")
    z_by_key = zscore_per_model(rows)
    pq = per_question_hcds(rows, z_by_key)
    summaries = model_summary(pq)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.label}" if args.label else ""
    pq_path = args.out_dir / f"hcds_per_question{suffix}.csv"
    sum_path = args.out_dir / f"hcds_summary{suffix}.csv"
    write_csv(pq_path, pq)
    write_csv(sum_path, summaries)
    print(f"Wrote {len(pq)} per-question rows to {pq_path}")
    print(f"Wrote {len(summaries)} summary rows to {sum_path}")
    print_summary(summaries)


if __name__ == "__main__":
    main()
