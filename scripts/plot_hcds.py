#!/usr/bin/env python
"""Generate the headline HCDS figure plus a per-condition accuracy strip.

The HCDS panel shows mean per-question HCDS per model with bootstrap
95% CIs. The accuracy panel shows mean accuracy per (model, prompt)
condition. Both written as a single PNG suitable for the paper.

Usage:
  python3 scripts/plot_hcds.py \\
      --task6-csv results/runs/<run_id>/task6_table.csv \\
      --hcds-summary-csv results/runs/<run_id>/hcds_summary.csv \\
      --out-png results/runs/<run_id>/hcds_figure.png
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def to_float(v):
    if v in (None, ""):
        return None
    try:
        f = float(v)
    except (ValueError, TypeError):
        return None
    if math.isnan(f):
        return None
    return f


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


PROMPT_ORDER = ["explicit_no_cot", "neutral_strict", "explicit_cot"]
PROMPT_LABELS = {
    "explicit_no_cot": "Explicit\nNoCoT",
    "neutral_strict": "Neutral",
    "explicit_cot": "Explicit\nCoT",
}
PROMPT_COLOURS = {
    "explicit_no_cot": "#E07A5F",
    "neutral_strict": "#3D5A80",
    "explicit_cot": "#81B29A",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task6-csv", type=Path, required=True)
    parser.add_argument("--hcds-summary-csv", type=Path, required=True)
    parser.add_argument("--out-png", type=Path, required=True)
    args = parser.parse_args()

    task6 = load_csv(args.task6_csv)
    hcds = load_csv(args.hcds_summary_csv)

    # --- HCDS panel data ---
    models_in_order = [r["model"] for r in hcds]
    means = [float(r["mean_hcds"]) for r in hcds]
    ci_low = [float(r["ci_low_95"]) for r in hcds]
    ci_high = [float(r["ci_high_95"]) for r in hcds]
    p_values = [float(r["p_value_two_sided"]) for r in hcds]
    yerr_low = [m - lo for m, lo in zip(means, ci_low)]
    yerr_high = [hi - m for m, hi in zip(means, ci_high)]

    # --- Accuracy panel data ---
    acc_by = defaultdict(list)
    for row in task6:
        v = to_float(row["accuracy"])
        if v is None:
            continue
        acc_by[(row["model"], row["prompt_condition"])].append(v)
    acc_means = {k: mean(vs) for k, vs in acc_by.items() if vs}

    fig, (ax_hcds, ax_acc) = plt.subplots(
        1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1, 1.4]}
    )

    # --- HCDS bar chart ---
    x = np.arange(len(models_in_order))
    bar_colours = [
        "#3D5A80" if m == "instruct" else "#81B29A" for m in models_in_order
    ]
    ax_hcds.bar(
        x,
        means,
        yerr=[yerr_low, yerr_high],
        capsize=8,
        color=bar_colours,
        edgecolor="black",
        linewidth=0.8,
        width=0.55,
    )
    ax_hcds.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax_hcds.set_xticks(x)
    ax_hcds.set_xticklabels(models_in_order, fontsize=11)
    ax_hcds.set_ylabel("HCDS  (= D(N, NoCoT) - D(N, CoT))", fontsize=11)
    ax_hcds.set_title("Hidden CoT Detection Score\nper-question, n=50, 95% bootstrap CI", fontsize=12)
    ax_hcds.spines[["top", "right"]].set_visible(False)
    for xi, m, p in zip(x, means, p_values):
        ax_hcds.annotate(
            f"p = {p:.1e}",
            xy=(xi, m),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="black",
        )

    # --- Accuracy by condition strip ---
    width = 0.25
    bar_x = np.arange(len(models_in_order))
    for i, prompt in enumerate(PROMPT_ORDER):
        offset = (i - 1) * width
        heights = [
            acc_means.get((m, prompt), 0.0) for m in models_in_order
        ]
        ax_acc.bar(
            bar_x + offset,
            heights,
            width=width,
            label=PROMPT_LABELS[prompt].replace("\n", " "),
            color=PROMPT_COLOURS[prompt],
            edgecolor="black",
            linewidth=0.6,
        )
        for bi, h in zip(bar_x + offset, heights):
            ax_acc.annotate(
                f"{h:.2f}",
                xy=(bi, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    ax_acc.set_xticks(bar_x)
    ax_acc.set_xticklabels(models_in_order, fontsize=11)
    ax_acc.set_ylabel("Accuracy on GSM8K (n=50)", fontsize=11)
    ax_acc.set_ylim(0, 1.05)
    ax_acc.set_title("Accuracy by prompt condition", fontsize=12)
    ax_acc.legend(loc="lower center", frameon=False, ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.22))
    ax_acc.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.out_png}")


if __name__ == "__main__":
    main()
