"""Sensitivity analysis: compute HCDS for all 2^k - 1 non-empty
subsets of the feature vector and report per-(dataset, model) coverage
of positive-sign and significance.

For GSM8K n=50 and StrategyQA n=50 (6-feature data): 63 subsets each.
For GSM8K n=500 (5-feature; mechanistic feature unavailable): 31 subsets.

Output:
  results/runs/sensitivity/hcds_subset_summary.csv  -- one row per
    (dataset, model, subset_signature) with mean_hcds, p, sign, sig.
  results/runs/sensitivity/hcds_subset_coverage.csv -- one row per
    (dataset, model) with %positive and %significant across all
    its subsets.

This addresses the PI's "why equal weighting?" question with a
discrete sensitivity analysis. If a high fraction of all
non-empty feature subsets gives the same sign, equal-weighting is
empirically vindicated.
"""

from __future__ import annotations

import csv
import itertools
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path("/Users/edward/Projects/AJAR")
OUT = ROOT / "results/runs/sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

# Feature ordering must match compute_hcds.py
ALL_FEATURES = [
    "latency_per_output_token",
    "token_entropy_mean",
    "token_entropy_slope",
    "paraphrase_consistency",
    "perturbation_delta_accuracy",
    "mechanistic_intervention_delta_accuracy",
]

DATASETS = [
    ("GSM8K n=50",      "results/runs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/task6_table.csv"),
    ("StrategyQA n=50", "results/runs/2026-05-06_2121_strategyqa50_gsm8k50_qwen3-4b_deep-table/task6_table.csv"),
    ("GSM8K n=500",     "results/runs/2026-05-07_0025_gsm8k500_qwen3-4b_deep-table-ext/task6_table.csv"),
]

PROMPTS = ("explicit_cot", "explicit_no_cot", "neutral_strict")


def to_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_rows(csv_path: Path) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            out.append(r)
    return out


def zscore_per_model(rows: Sequence[Dict[str, object]],
                      features: Sequence[str]) -> Dict[Tuple[str, int, str], Dict[str, Optional[float]]]:
    """Returns {(model, sample_idx, prompt): {feature: z}} where z is
    None if the underlying feature was None."""
    by_model_feature: Dict[Tuple[str, str], List[float]] = {}
    for r in rows:
        model = str(r.get("model", ""))
        for feat in features:
            v = to_float(r.get(feat))
            if v is not None:
                by_model_feature.setdefault((model, feat), []).append(v)

    stats: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for (model, feat), vals in by_model_feature.items():
        if len(vals) < 2:
            stats[(model, feat)] = (0.0, 1.0)
        else:
            mean = statistics.mean(vals)
            sd = statistics.pstdev(vals) or 1.0
            stats[(model, feat)] = (mean, sd)

    z_by_key: Dict[Tuple[str, int, str], Dict[str, Optional[float]]] = {}
    for r in rows:
        model = str(r.get("model", ""))
        try:
            sid = int(str(r.get("sample_idx") or 0))
        except ValueError:
            continue
        prompt = str(r.get("prompt_condition", "") or r.get("prompt_name", ""))
        if prompt not in PROMPTS:
            continue
        z_by_key[(model, sid, prompt)] = {}
        for feat in features:
            v = to_float(r.get(feat))
            if v is None:
                z_by_key[(model, sid, prompt)][feat] = None
                continue
            mean, sd = stats.get((model, feat), (0.0, 1.0))
            z_by_key[(model, sid, prompt)][feat] = (v - mean) / sd
    return z_by_key


def euclidean_partial(a: Dict[str, Optional[float]],
                       b: Dict[str, Optional[float]],
                       features: Sequence[str]) -> Optional[float]:
    sq = 0.0
    n = 0
    for f in features:
        av = a.get(f)
        bv = b.get(f)
        if av is None or bv is None:
            continue
        sq += (av - bv) ** 2
        n += 1
    if n == 0:
        return None
    return math.sqrt(sq)


def per_question_hcds(z_by_key, features, models) -> Dict[str, List[float]]:
    by_model_q: Dict[Tuple[str, int], Dict[str, Dict[str, Optional[float]]]] = {}
    for (model, sid, prompt), feats in z_by_key.items():
        by_model_q.setdefault((model, sid), {})[prompt] = feats

    out: Dict[str, List[float]] = {m: [] for m in models}
    for (model, sid), prompts in by_model_q.items():
        if model not in models:
            continue
        if not all(p in prompts for p in ("neutral_strict", "explicit_no_cot", "explicit_cot")):
            continue
        d_n_nocot = euclidean_partial(prompts["neutral_strict"], prompts["explicit_no_cot"], features)
        d_n_cot = euclidean_partial(prompts["neutral_strict"], prompts["explicit_cot"], features)
        if d_n_nocot is None or d_n_cot is None:
            continue
        out[model].append(d_n_nocot - d_n_cot)
    return out


def t_test_one_sample(values: List[float]) -> Tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values) or 1e-12
    se = sd / math.sqrt(n)
    t = mean / se
    try:
        from scipy.stats import t as t_dist
        p = 2 * (1 - t_dist.cdf(abs(t), df=n - 1))
    except ImportError:
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return t, p


def features_available(rows: Sequence[Dict[str, object]],
                        features: Sequence[str]) -> List[str]:
    """Return the subset of features that have at least one non-null value."""
    out = []
    for f in features:
        if any(to_float(r.get(f)) is not None for r in rows):
            out.append(f)
    return out


def all_subsets(features: Sequence[str]) -> List[Tuple[str, ...]]:
    out: List[Tuple[str, ...]] = []
    for k in range(1, len(features) + 1):
        for subset in itertools.combinations(features, k):
            out.append(tuple(subset))
    return out


def main() -> None:
    summary_rows = []
    coverage_rows = []
    for label, rel_path in DATASETS:
        csv_path = ROOT / rel_path
        if not csv_path.exists():
            print(f"  [skip] {label}: {csv_path} not found")
            continue
        rows = load_rows(csv_path)
        if not rows:
            print(f"  [skip] {label}: empty")
            continue
        feats_present = features_available(rows, ALL_FEATURES)
        subsets = all_subsets(feats_present)
        models = sorted({str(r.get("model", "")) for r in rows})
        print(f"  [{label}] features available: {len(feats_present)} -> {2**len(feats_present)-1} subsets, models: {models}")

        # Pre-compute z-scores once over the FULL feature set; this z-score
        # is computed across all (q, prompt) rows per model and is identical
        # regardless of which subset we then take a distance over.
        z_by_key = zscore_per_model(rows, feats_present)

        for model in models:
            counts = {"positive": 0, "significant_positive": 0, "negative": 0, "significant": 0, "total": 0}
            for subset in subsets:
                per_q_by_model = per_question_hcds(z_by_key, list(subset), [model])
                vals = per_q_by_model.get(model, [])
                if not vals:
                    continue
                mean = statistics.mean(vals)
                t, p = t_test_one_sample(vals)
                positive = mean > 0
                significant = p < 0.05
                counts["total"] += 1
                if positive:
                    counts["positive"] += 1
                else:
                    counts["negative"] += 1
                if significant:
                    counts["significant"] += 1
                if positive and significant:
                    counts["significant_positive"] += 1

                # Build a short signature for this subset
                sig = "+".join(f.replace("_", "") for f in subset)
                summary_rows.append({
                    "dataset": label,
                    "model": model,
                    "subset_size": len(subset),
                    "subset": ",".join(subset),
                    "n_questions": len(vals),
                    "mean_hcds": round(mean, 4),
                    "t_statistic": round(t, 3),
                    "p_value_two_sided": p,
                    "positive": int(positive),
                    "significant_positive": int(positive and significant),
                })
            coverage_rows.append({
                "dataset": label,
                "model": model,
                "n_subsets": counts["total"],
                "n_positive": counts["positive"],
                "pct_positive": round(100 * counts["positive"] / counts["total"], 1) if counts["total"] else 0.0,
                "n_significant_positive": counts["significant_positive"],
                "pct_significant_positive": round(100 * counts["significant_positive"] / counts["total"], 1) if counts["total"] else 0.0,
            })

    # Write per-subset CSV
    if summary_rows:
        path = OUT / "hcds_subset_summary.csv"
        fields = ["dataset", "model", "subset_size", "subset", "n_questions",
                  "mean_hcds", "t_statistic", "p_value_two_sided",
                  "positive", "significant_positive"]
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)
        print(f"\nWrote {path} ({len(summary_rows)} rows)")

    # Write coverage CSV
    if coverage_rows:
        path = OUT / "hcds_subset_coverage.csv"
        fields = ["dataset", "model", "n_subsets", "n_positive", "pct_positive",
                  "n_significant_positive", "pct_significant_positive"]
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in coverage_rows:
                w.writerow(r)
        print(f"Wrote {path}\n")

        # Pretty print
        print(f"{'Dataset':<18}{'Model':<10}{'subsets':>8}{'+sign':>8}{'+sign %':>10}{'+sig':>8}{'+sig %':>10}")
        print("-" * 75)
        for r in coverage_rows:
            print(f"{r['dataset']:<18}{r['model']:<10}{r['n_subsets']:>8}"
                  f"{r['n_positive']:>8}{r['pct_positive']:>9.1f}%"
                  f"{r['n_significant_positive']:>8}{r['pct_significant_positive']:>9.1f}%")


if __name__ == "__main__":
    main()
