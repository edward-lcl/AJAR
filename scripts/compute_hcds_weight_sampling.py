"""Continuous-weight sensitivity analysis for HCDS.

The subset analysis (compute_hcds_all_subsets.py) tests robustness to
*discrete* feature inclusion (each feature weight in {0, 1}). This script
generalizes that to *continuous* weights: we draw feature-weight vectors
uniformly from the probability simplex (Dirichlet(1,...,1)) and recompute
HCDS under a weighted Euclidean metric

    D_w(a, b) = sqrt( sum_j w_j (a_j - b_j)^2 ),   HCDS_q = D_w(N,NoCoT) - D_w(N,CoT).

For each (dataset, model) we report the fraction of sampled weightings that
keep HCDS positive and significantly positive, plus the 5/50/95th percentiles
of the resulting mean HCDS. The equal-weight point (all w_j equal) is reported
as the anchor.

Why sign/significance are the right summaries: HCDS = D1 - D2 and both distances
scale by sqrt(c) under w -> c*w, so the sign and the one-sample t-statistic are
invariant to the overall weight scale. A simplex weight vector (sum = 1) is
therefore directly comparable to the paper's equal weighting (all w_j = 1); only
the *relative* feature weights vary. This is a purely post-hoc computation on the
committed z-scored feature tables -- no model runs.

This does NOT learn or optimize weights. HCDS is unsupervised (there is no
per-question hidden-CoT label), so optimizing weights toward a larger effect
would be circular. We only test robustness to how the fixed features are weighted.

Output (results/runs/sensitivity/):
  hcds_weight_sampling_coverage.csv -- one row per (dataset, model).
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import t as t_dist

ROOT = Path("/Users/edward/Projects/AJAR")
OUT = ROOT / "results/runs/sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

# Must match compute_hcds.py ordering.
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
N_SAMPLES = 10000
SEED = 17


def to_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def features_available(rows: Sequence[Dict[str, str]]) -> List[str]:
    return [f for f in ALL_FEATURES if any(to_float(r.get(f)) is not None for r in rows)]


def zscore_per_model(
    rows: Sequence[Dict[str, str]], features: Sequence[str]
) -> Dict[Tuple[str, int, str], Dict[str, Optional[float]]]:
    """{(model, sample_idx, prompt): {feature: z or None}}, z-scored per model."""
    vals_by: Dict[Tuple[str, str], List[float]] = {}
    for r in rows:
        model = str(r.get("model", ""))
        for feat in features:
            v = to_float(r.get(feat))
            if v is not None:
                vals_by.setdefault((model, feat), []).append(v)
    # Match compute_hcds.py exactly: sample std (ddof=1); sigma==0 -> z=0.
    stat: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for k, vals in vals_by.items():
        stat[k] = (statistics.mean(vals), statistics.stdev(vals)) if len(vals) >= 2 else (0.0, 0.0)

    z: Dict[Tuple[str, int, str], Dict[str, Optional[float]]] = {}
    for r in rows:
        model = str(r.get("model", ""))
        prompt = str(r.get("prompt_condition", "") or r.get("prompt_name", ""))
        if prompt not in PROMPTS:
            continue
        try:
            sid = int(str(r.get("sample_idx") or 0))
        except ValueError:
            continue
        cell: Dict[str, Optional[float]] = {}
        for feat in features:
            v = to_float(r.get(feat))
            if v is None:
                cell[feat] = None
            else:
                mu, sd = stat.get((model, feat), (0.0, 0.0))
                cell[feat] = (v - mu) / sd if sd > 0 else 0.0
        z[(model, sid, prompt)] = cell
    return z


def build_diff_matrices(
    z: Dict[Tuple[str, int, str], Dict[str, Optional[float]]],
    features: Sequence[str],
    model: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-question squared-difference matrices (Q, F) with NaN for a feature
    that is missing in either arm of the pair. Rows: questions with all three
    prompts and >=1 shared feature in each pair.

    Returns (diff_nocot, diff_cot); a weighted distance is
    sqrt(nansum(w * diff)) per row.
    """
    by_q: Dict[int, Dict[str, Dict[str, Optional[float]]]] = {}
    for (m, sid, prompt), cell in z.items():
        if m == model:
            by_q.setdefault(sid, {})[prompt] = cell

    nocot_rows: List[List[float]] = []
    cot_rows: List[List[float]] = []
    for sid in sorted(by_q):
        arms = by_q[sid]
        if not all(p in arms for p in PROMPTS):
            continue
        n_, no, co = arms["neutral_strict"], arms["explicit_no_cot"], arms["explicit_cot"]

        def sqdiff(a, b):
            row = []
            for f in features:
                av, bv = a.get(f), b.get(f)
                row.append((av - bv) ** 2 if (av is not None and bv is not None) else np.nan)
            return row

        d_no = sqdiff(n_, no)
        d_co = sqdiff(n_, co)
        # require at least one shared feature in each pair
        if all(math.isnan(x) for x in d_no) or all(math.isnan(x) for x in d_co):
            continue
        nocot_rows.append(d_no)
        cot_rows.append(d_co)
    return np.array(nocot_rows, dtype=float), np.array(cot_rows, dtype=float)


def weighted_hcds_stats(
    diff_nocot: np.ndarray, diff_cot: np.ndarray, weights: np.ndarray
) -> Tuple[float, float]:
    """Return (mean_hcds, two_sided_p) for one weight vector (F,)."""
    # nansum over features: treat missing (NaN) as zero contribution (dropped).
    d_no = np.sqrt(np.nansum(diff_nocot * weights, axis=1))
    d_co = np.sqrt(np.nansum(diff_cot * weights, axis=1))
    hcds = d_no - d_co
    n = hcds.size
    if n < 2:
        return float("nan"), float("nan")
    mean = float(hcds.mean())
    sd = float(hcds.std(ddof=1)) or 1e-12
    t = mean / (sd / math.sqrt(n))
    p = float(2 * (1 - t_dist.cdf(abs(t), df=n - 1)))
    return mean, p


def main() -> None:
    rng = np.random.default_rng(SEED)
    coverage_rows: List[Dict[str, object]] = []

    for label, rel in DATASETS:
        path = ROOT / rel
        if not path.exists():
            print(f"  [skip] {label}: {path} not found")
            continue
        rows = load_rows(path)
        feats = features_available(rows)
        k = len(feats)
        z = zscore_per_model(rows, feats)
        models = sorted({str(r.get("model", "")) for r in rows})
        print(f"[{label}] {k} features, {N_SAMPLES} simplex samples, models: {models}")

        # One shared weight sample set per dataset (uniform over the k-simplex).
        weight_samples = rng.dirichlet(np.ones(k), size=N_SAMPLES)  # (S, k)
        equal_w = np.ones(k)  # sign/p invariant to scale, so all-ones == uniform mean

        for model in models:
            diff_no, diff_co = build_diff_matrices(z, feats, model)
            if diff_no.size == 0:
                print(f"  [skip] {label}/{model}: no usable questions")
                continue
            means = np.empty(N_SAMPLES)
            ps = np.empty(N_SAMPLES)
            for i in range(N_SAMPLES):
                means[i], ps[i] = weighted_hcds_stats(diff_no, diff_co, weight_samples[i])
            eq_mean, eq_p = weighted_hcds_stats(diff_no, diff_co, equal_w)

            pos = means > 0
            sig_pos = pos & (ps < 0.05)
            coverage_rows.append({
                "dataset": label,
                "model": model,
                "n_questions": int(diff_no.shape[0]),
                "n_features": k,
                "n_samples": N_SAMPLES,
                "pct_positive": round(100 * float(pos.mean()), 1),
                "pct_significant_positive": round(100 * float(sig_pos.mean()), 1),
                "hcds_p05": round(float(np.percentile(means, 5)), 3),
                "hcds_p50": round(float(np.percentile(means, 50)), 3),
                "hcds_p95": round(float(np.percentile(means, 95)), 3),
                "equal_weight_hcds": round(eq_mean, 3),
                "equal_weight_p": eq_p,
            })

    if coverage_rows:
        path = OUT / "hcds_weight_sampling_coverage.csv"
        fields = list(coverage_rows[0].keys())
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(coverage_rows)
        print(f"\nWrote {path}\n")
        hdr = f"{'Dataset':<16}{'Model':<10}{'+sign %':>9}{'+sig %':>9}{'HCDS p5/50/95':>22}{'equal-w':>9}"
        print(hdr)
        print("-" * len(hdr))
        for r in coverage_rows:
            pct = f"{r['hcds_p05']:+.2f}/{r['hcds_p50']:+.2f}/{r['hcds_p95']:+.2f}"
            print(f"{r['dataset']:<16}{r['model']:<10}{r['pct_positive']:>8.1f}%"
                  f"{r['pct_significant_positive']:>8.1f}%{pct:>22}{r['equal_weight_hcds']:>+9.2f}")


if __name__ == "__main__":
    main()
