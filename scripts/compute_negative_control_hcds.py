"""Compute latency-only HCDS on the negative-control runs and compare
to the GSM8K latency-only HCDS we already report.

Usage:
  python3 scripts/compute_negative_control_hcds.py

Reads baselines from:
  outputs/neg_control_arithmetic/<model>/<prompt>/sample_*/baseline.json
  outputs/neg_control_factual/<model>/<prompt>/sample_*/baseline.json

Computes HCDS using only latency_per_output_token (the 'latency_only'
variant we already use in the GSM8K Robustness section). Writes results
to:
  results/runs/negative_control/hcds_summary_latency_only.csv
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path("/Users/edward/Projects/AJAR")
RUNS = {
    "arithmetic": ROOT / "outputs/neg_control_arithmetic",
    "factual":    ROOT / "outputs/neg_control_factual",
}
PROMPTS = ("explicit_cot", "explicit_no_cot", "neutral_strict")
MODELS = ("instruct", "thinking")
OUT = ROOT / "results/runs/negative_control"
OUT.mkdir(parents=True, exist_ok=True)


def load_baseline_rows(run_dir: Path) -> List[Dict[str, object]]:
    rows = []
    for model in MODELS:
        for prompt in PROMPTS:
            base = run_dir / model / prompt
            if not base.exists():
                continue
            for sample_dir in sorted(base.glob("sample_*")):
                bj = sample_dir / "baseline.json"
                if not bj.exists():
                    continue
                with bj.open() as f:
                    d = json.load(f)
                gen_secs = d.get("generation_seconds")
                # output_tokens isn't directly in baseline.json on omlx;
                # estimate from generated_text length / 4 chars per token
                # OR from usage if present.
                usage = d.get("usage") or {}
                out_tok = usage.get("completion_tokens") or usage.get("output_tokens")
                if out_tok is None:
                    text = d.get("generated_text") or ""
                    out_tok = max(1, len(text) // 4)
                if gen_secs is None or gen_secs <= 0 or out_tok <= 0:
                    continue
                latency_per_token = float(gen_secs) / float(out_tok)
                rows.append({
                    "model": model,
                    "prompt": prompt,
                    "sample_idx": int(d.get("sample_idx", 0)),
                    "latency_per_output_token": latency_per_token,
                    "output_tokens": out_tok,
                    "generation_seconds": float(gen_secs),
                    "correct": int(bool(d.get("correct"))),
                })
    return rows


def zscore_per_model(rows: List[Dict[str, object]]) -> Dict[Tuple[str, int, str], Optional[float]]:
    """Return {(model, sample_idx, prompt): zscore_latency}."""
    by_model: Dict[str, List[float]] = {m: [] for m in MODELS}
    for r in rows:
        by_model[str(r["model"])].append(float(r["latency_per_output_token"]))
    stats = {}
    for m, vals in by_model.items():
        if len(vals) < 2:
            stats[m] = (0.0, 1.0)
        else:
            mean = statistics.mean(vals)
            sd = statistics.pstdev(vals) or 1.0
            stats[m] = (mean, sd)
    out = {}
    for r in rows:
        m = str(r["model"])
        mean, sd = stats[m]
        z = (float(r["latency_per_output_token"]) - mean) / sd
        out[(m, int(r["sample_idx"]), str(r["prompt"]))] = z
    return out


def compute_hcds_per_question(rows, zscores) -> List[Dict[str, object]]:
    """For each (model, sample_idx), compute
    HCDS_q = D(N, NoCoT) - D(N, CoT)
    where D is just |z_a - z_b| since we have 1 feature.
    """
    out: List[Dict[str, object]] = []
    by_key: Dict[Tuple[str, int], Dict[str, float]] = {}
    for r in rows:
        key = (str(r["model"]), int(r["sample_idx"]))
        by_key.setdefault(key, {})[str(r["prompt"])] = zscores[(key[0], key[1], str(r["prompt"]))]

    for (model, sid), prompts in sorted(by_key.items()):
        if not all(p in prompts for p in ("neutral_strict", "explicit_no_cot", "explicit_cot")):
            continue
        d_n_nocot = abs(prompts["neutral_strict"] - prompts["explicit_no_cot"])
        d_n_cot = abs(prompts["neutral_strict"] - prompts["explicit_cot"])
        hcds_q = d_n_nocot - d_n_cot
        out.append({
            "model": model,
            "sample_idx": sid,
            "d_n_nocot": d_n_nocot,
            "d_n_cot": d_n_cot,
            "hcds_q": hcds_q,
        })
    return out


def bootstrap_ci(values: List[float], n_resamples: int = 1000, seed: int = 17, alpha: float = 0.05):
    import random
    rng = random.Random(seed)
    if not values:
        return float("nan"), float("nan")
    means = []
    for _ in range(n_resamples):
        sample = [rng.choice(values) for _ in range(len(values))]
        means.append(statistics.mean(sample))
    means.sort()
    lo = means[int(alpha / 2 * n_resamples)]
    hi = means[int((1 - alpha / 2) * n_resamples)]
    return lo, hi


def t_test_one_sample(values: List[float]) -> Tuple[float, float]:
    """Return (t_stat, two_sided_p) for H0: mean = 0."""
    if len(values) < 2:
        return float("nan"), float("nan")
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values) or 1e-12
    se = sd / math.sqrt(n)
    t = mean / se
    # two-sided p via approx using survival of |t| under t_{n-1}
    # use scipy if available; else approx with normal for n large
    try:
        from scipy.stats import t as t_dist
        p = 2 * (1 - t_dist.cdf(abs(t), df=n - 1))
    except ImportError:
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return t, p


def summarize(per_q_rows: List[Dict[str, object]], dataset_label: str) -> List[Dict[str, object]]:
    out = []
    for model in MODELS:
        vals = [float(r["hcds_q"]) for r in per_q_rows if r["model"] == model]
        if not vals:
            continue
        mean = statistics.mean(vals)
        ci_lo, ci_hi = bootstrap_ci(vals)
        t, p = t_test_one_sample(vals)
        out.append({
            "dataset": dataset_label,
            "model": model,
            "n_questions": len(vals),
            "mean_hcds": round(mean, 4),
            "ci_low_95": round(ci_lo, 4),
            "ci_high_95": round(ci_hi, 4),
            "t_statistic": round(t, 3),
            "p_value_two_sided": p,
        })
    return out


def main() -> None:
    summaries: List[Dict[str, object]] = []
    for label, run_dir in RUNS.items():
        if not run_dir.exists():
            print(f"  [skip] {label}: {run_dir} not present")
            continue
        rows = load_baseline_rows(run_dir)
        if not rows:
            print(f"  [skip] {label}: no baseline rows found in {run_dir}")
            continue
        print(f"  [{label}] loaded {len(rows)} baseline rows")
        zscores = zscore_per_model(rows)
        per_q = compute_hcds_per_question(rows, zscores)
        print(f"  [{label}] computed {len(per_q)} per-question HCDS rows")
        summaries.extend(summarize(per_q, label))

    out_path = OUT / "hcds_summary_latency_only.csv"
    if not summaries:
        print("No summaries to write — runs not yet complete.")
        return
    fields = ["dataset", "model", "n_questions", "mean_hcds",
              "ci_low_95", "ci_high_95", "t_statistic", "p_value_two_sided"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summaries:
            w.writerow(r)
    print(f"\nWrote {out_path}")
    print()
    # Pretty print
    print(f"{'Dataset':<14}{'Model':<10}{'n':>5}{'HCDS':>10}{'95% CI':>22}{'p':>14}")
    print("-" * 75)
    for r in summaries:
        ci = f"[{r['ci_low_95']:+.3f}, {r['ci_high_95']:+.3f}]"
        p_str = f"{r['p_value_two_sided']:.2e}" if r["p_value_two_sided"] < 0.01 else f"{r['p_value_two_sided']:.3f}"
        print(f"{r['dataset']:<14}{r['model']:<10}{r['n_questions']:>5}"
              f"{r['mean_hcds']:>+10.3f}{ci:>22}{p_str:>14}")


if __name__ == "__main__":
    main()
