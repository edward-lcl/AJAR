"""Build paper-ready figures for the AJAR submission.

Outputs vector PDFs (matplotlib) into figures/. Numbers come from the
authoritative CSVs in results/runs/, not the spreadsheet.

Figures produced:
  fig1_cross_dataset_hcds.pdf   — HCDS bars w/ 95% CI across datasets
  fig2_robustness_ablation.pdf  — LOO feature ablations
  fig3_length_matched.pdf       — HCDS by output-length tier
  fig4_anchor_control.pdf       — anchor − control across prompts
  fig5_output_lengths.pdf       — output-length distributions by prompt
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

ROOT = Path("/Users/edward/Projects/AJAR")
STAGE1 = ROOT / "results/runs/2026-05-06_2121_strategyqa50_gsm8k50_qwen3-4b_deep-table"
STAGE2 = ROOT / "results/runs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table"
STAGE3 = ROOT / "results/runs/2026-05-07_0025_gsm8k500_qwen3-4b_deep-table-ext"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

INSTRUCT_COLOR = "#5B7C99"   # muted grey-blue (from blueprint palette)
THINKING_COLOR = "#D9A35A"   # warm amber
NEUTRAL_GREY = "#888888"
HCDS_BLUE = "#2166AC"
TEAL = "#3F8E8C"
PURPLE = "#6E5797"
GREEN = "#2C7A39"
FEATURE_COLOR_DARK = "#B07820"  # darker amber (for feature-vector box: white text)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def read_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def load_summary(path: Path) -> Dict[str, Dict[str, float]]:
    return {r["model"]: {k: (float(v) if k != "verdict" and k != "model" else v)
                          for k, v in r.items()}
            for r in read_dicts(path)}


def fig1_cross_dataset() -> None:
    gsm50 = load_summary(STAGE2 / "hcds_summary.csv")
    sqa50 = load_summary(STAGE1 / "hcds_summary.csv")
    gsm500 = load_summary(STAGE3 / "hcds_summary.csv")

    datasets = [
        ("GSM8K\n(n=50)", gsm50),
        ("StrategyQA\n(n=50)", sqa50),
        ("GSM8K\n(n=500)", gsm500),
    ]
    models = ["instruct", "thinking"]
    colors = {"instruct": INSTRUCT_COLOR, "thinking": THINKING_COLOR}

    fig, ax = plt.subplots(figsize=(7, 4.2))
    width = 0.36
    x = np.arange(len(datasets))

    for i, m in enumerate(models):
        means = [d[1][m]["mean_hcds"] for d in datasets]
        ci_lo = [d[1][m]["ci_low_95"] for d in datasets]
        ci_hi = [d[1][m]["ci_high_95"] for d in datasets]
        err_lo = [mn - lo for mn, lo in zip(means, ci_lo)]
        err_hi = [hi - mn for mn, hi in zip(means, ci_hi)]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=[err_lo, err_hi],
                      capsize=4, label=m.capitalize(), color=colors[m],
                      edgecolor="black", linewidth=0.5,
                      error_kw={"elinewidth": 1.0, "ecolor": "#333"})

        for j, (b, mn, lbl) in enumerate(zip(bars, means, datasets)):
            p = lbl[1][m]["p_value_two_sided"]
            p_str = f"p={p:.2g}" if p > 1e-10 else f"p={p:.0e}"
            ax.text(b.get_x() + b.get_width() / 2,
                    mn + err_hi[j] + 0.08,
                    p_str, ha="center", va="bottom", fontsize=8, color="#333")

    ax.axhline(0, color="black", linewidth=0.5, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels([d[0] for d in datasets])
    ax.set_ylabel("HCDS  (D(N, NoCoT) − D(N, CoT))")
    ax.set_title("Hidden CoT Detection Score — cross-dataset replication", pad=12)
    ax.legend(loc="upper right", title="Model")
    ax.set_ylim(-0.3, max(d[1]["instruct"]["ci_high_95"] for d in datasets) + 0.55)

    fig.text(0.5, -0.02,
             "HCDS > 0 ⇒ neutral prompt behaves more like CoT than NoCoT in feature space. "
             "Error bars: 95% bootstrap CI (1000 samples, seed=17).",
             ha="center", fontsize=8.5, style="italic", color="#444")

    fig.tight_layout()
    out = OUT / "fig1_cross_dataset_hcds.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def fig2_robustness() -> None:
    variants = [
        ("full", "hcds_summary.csv", "All 6 features"),
        ("no_paraphrase", "hcds_summary_no_paraphrase.csv", "− paraphrase"),
        ("no_perturb", "hcds_summary_no_perturb.csv", "− perturbation"),
        ("no_entropy", "hcds_summary_no_entropy.csv", "− entropy"),
        ("no_mech", "hcds_summary_no_mech.csv", "− mechanistic"),
        ("entropy_only", "hcds_summary_entropy_only.csv", "entropy only"),
        ("latency_only", "hcds_summary_latency_only.csv", "latency only"),
    ]
    rows: List[tuple] = []
    for v, fname, label in variants:
        d = load_summary(STAGE2 / fname)
        rows.append((label, d["instruct"], d["thinking"]))

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    y = np.arange(len(rows))
    h = 0.35

    inst_means = [r[1]["mean_hcds"] for r in rows]
    inst_lo = [r[1]["ci_low_95"] for r in rows]
    inst_hi = [r[1]["ci_high_95"] for r in rows]
    inst_p = [r[1]["p_value_two_sided"] for r in rows]
    think_means = [r[2]["mean_hcds"] for r in rows]
    think_lo = [r[2]["ci_low_95"] for r in rows]
    think_hi = [r[2]["ci_high_95"] for r in rows]
    think_p = [r[2]["p_value_two_sided"] for r in rows]

    inst_errlo = [m - lo for m, lo in zip(inst_means, inst_lo)]
    inst_errhi = [hi - m for hi, m in zip(inst_hi, inst_means)]
    think_errlo = [m - lo for m, lo in zip(think_means, think_lo)]
    think_errhi = [hi - m for hi, m in zip(think_hi, think_means)]

    ax.barh(y - h/2, inst_means, h, xerr=[inst_errlo, inst_errhi],
            capsize=3, label="Instruct", color=INSTRUCT_COLOR,
            edgecolor="black", linewidth=0.4,
            error_kw={"elinewidth": 0.8, "ecolor": "#333"})
    ax.barh(y + h/2, think_means, h, xerr=[think_errlo, think_errhi],
            capsize=3, label="Thinking", color=THINKING_COLOR,
            edgecolor="black", linewidth=0.4,
            error_kw={"elinewidth": 0.8, "ecolor": "#333"})

    # Mark non-significant Thinking variant
    for i, p in enumerate(think_p):
        if p >= 0.05:
            ax.scatter([think_means[i]], [y[i] + h/2], marker="x", s=60,
                       color="#C00000", zorder=5, linewidths=2)

    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("HCDS")
    ax.set_title("Robustness across HCDS feature subsets (GSM8K, n=50)", pad=10)
    ax.legend(loc="lower right")

    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], marker="x", color="#C00000",
                              linestyle="", markersize=8, label="p ≥ 0.05"))
    ax.legend(handles=handles, loc="lower right")

    fig.text(0.5, -0.02,
             "All variants positive on Instruct. Dropping entropy features collapses Thinking signal "
             "(HCDS=+0.09, p=0.62) — entropy is load-bearing for the Thinking pathway.",
             ha="center", fontsize=8.5, style="italic", color="#444")
    fig.tight_layout()
    out = OUT / "fig2_robustness_ablation.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def fig3_length_matched() -> None:
    rows = read_dicts(STAGE2 / "hcds_length_matched.csv")
    by_model: Dict[str, List[Dict[str, str]]] = {"instruct": [], "thinking": []}
    for r in rows:
        by_model[r["model"]].append(r)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    tier_order = ["short", "medium", "long"]
    x_pos = np.arange(len(tier_order))
    width = 0.36

    for i, (m, color) in enumerate([("instruct", INSTRUCT_COLOR),
                                     ("thinking", THINKING_COLOR)]):
        rows_m = {r["length_tier"]: r for r in by_model[m]}
        means, errlo, errhi, ns, ps = [], [], [], [], []
        for tier in tier_order:
            if tier not in rows_m:
                means.append(np.nan); errlo.append(0); errhi.append(0)
                ns.append(0); ps.append(np.nan)
                continue
            r = rows_m[tier]
            mean = float(r["mean_hcds"])
            means.append(mean)
            errlo.append(mean - float(r["ci_low"]))
            errhi.append(float(r["ci_high"]) - mean)
            ns.append(int(r["n"]))
            ps.append(float(r["p_value"]))
        offset = (i - 0.5) * width
        bars = ax.bar(x_pos + offset, means, width, yerr=[errlo, errhi],
                      capsize=4, label=m.capitalize(), color=color,
                      edgecolor="black", linewidth=0.4,
                      error_kw={"elinewidth": 1.0, "ecolor": "#333"})
        for j, (b, mn, p, n) in enumerate(zip(bars, means, ps, ns)):
            if np.isnan(mn):
                ax.text(b.get_x() + b.get_width()/2, 0.05, "n/a",
                        ha="center", fontsize=7, color="#888")
                continue
            if p >= 0.05:
                ax.scatter([b.get_x() + b.get_width()/2], [mn], marker="x",
                           s=50, color="#C00000", zorder=5, linewidths=2)
            ax.text(b.get_x() + b.get_width()/2, mn + errhi[j] + 0.05,
                    f"n={n}", ha="center", fontsize=7, color="#555")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Short\n(~175 tok)", "Medium\n(~280 tok)", "Long\n(~470 tok)"])
    ax.set_ylabel("HCDS")
    ax.set_title("Length-matched HCDS by output-length tier (GSM8K, n=50)", pad=10)
    ax.set_ylim(-0.6, 4.4)

    handles, _ = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], marker="x", color="#C00000",
                              linestyle="", markersize=8, label="p ≥ 0.05"))
    ax.legend(handles=handles, title="Model",
              loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=3, frameon=False)

    fig.text(0.5, -0.04,
             "Instruct holds across all length tiers; Thinking signal is significant on short outputs "
             "but ambiguous on medium (cap-affected). Long Thinking tier not in source CSV.",
             ha="center", fontsize=8.5, style="italic", color="#444")
    fig.tight_layout()
    out = OUT / "fig3_length_matched.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def fig4_anchor_control() -> None:
    gsm = read_dicts(STAGE2 / "task10_anchor_sensitivity.csv")
    sqa = read_dicts(STAGE1 / "task10_anchor_sensitivity.csv")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.0), sharey=True)
    prompt_order = ["explicit_cot", "explicit_no_cot", "neutral_strict"]
    prompt_labels = ["explicit_cot", "explicit_no_cot", "neutral_strict"]
    width = 0.36

    for ax, rows, title in [(axes[0], gsm, "GSM8K (n=50)"),
                             (axes[1], sqa, "StrategyQA (n=50)")]:
        x = np.arange(len(prompt_order))
        all_vals: Dict[str, List[float]] = {"instruct": [], "thinking": []}
        for i, (m, color) in enumerate([("instruct", INSTRUCT_COLOR),
                                         ("thinking", THINKING_COLOR)]):
            vals: List[float] = []
            for p in prompt_order:
                rec = next((r for r in rows
                            if r["model_key"] == m and r["prompt_name"] == p), None)
                vals.append(float(rec["anchor_minus_control_drop"]) if rec else np.nan)
            all_vals[m] = vals
            offset = (i - 0.5) * width
            ax.bar(x + offset, vals, width, label=m.capitalize(),
                   color=color, edgecolor="black", linewidth=0.4)
            for xi, v in zip(x + offset, vals):
                if np.isnan(v):
                    continue
                # Place labels above zero bars; below for negatives, with extra
                # padding so they clear the x-axis tick labels.
                if abs(v) < 0.005:
                    ax.text(xi, 0.012, f"{v:+.3f}", ha="center", fontsize=7,
                            color="#888")
                elif v > 0:
                    ax.text(xi, v + 0.010, f"{v:+.3f}", ha="center",
                            fontsize=7, color="#333")
                else:
                    ax.text(xi, v - 0.012, f"{v:+.3f}", ha="center",
                            fontsize=7, color="#C00000", va="top")

        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(prompt_labels, rotation=12, ha="right")
        ax.set_title(title)
        ax.set_ylim(-0.40, 0.20)

    axes[0].set_ylabel("anchor_drop − control_drop\n(positive ⇒ anchors causally privileged)")
    axes[1].legend(loc="lower right", title="Model")
    fig.suptitle("Mechanistic anchor sensitivity by prompt", y=1.02, fontsize=12)
    fig.text(0.5, -0.05,
             "Negative bar (Thinking + explicit_cot) = control disruption hurts more than anchor disruption: "
             "long reasoning chains distribute causal load across many steps.",
             ha="center", fontsize=8.5, style="italic", color="#444")
    fig.tight_layout()
    out = OUT / "fig4_anchor_control.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def fig5_output_lengths() -> None:
    # Load per-question features (n=500 has output_tokens; n=50 only has summary).
    pq_path = STAGE3 / "task6_table.csv"
    rows = read_dicts(pq_path)
    by_key: Dict[tuple, List[float]] = {}
    for r in rows:
        try:
            key = (r["model"], r["prompt_condition"])
            by_key.setdefault(key, []).append(float(r["output_tokens"]))
        except (KeyError, ValueError):
            continue

    prompt_order = ["explicit_cot", "neutral_strict", "explicit_no_cot"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, m, color in [(axes[0], "instruct", INSTRUCT_COLOR),
                          (axes[1], "thinking", THINKING_COLOR)]:
        bp_data = [by_key.get((m, p), []) for p in prompt_order]
        positions = np.arange(len(prompt_order))
        bp = ax.boxplot(bp_data, positions=positions, widths=0.55,
                        patch_artist=True,
                        medianprops={"color": "white", "linewidth": 1.5},
                        flierprops={"marker": ".", "markersize": 2.5,
                                    "markerfacecolor": "#666", "markeredgecolor": "none"})
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(prompt_order, rotation=12, ha="right")
        ax.set_title(f"Qwen3-4B {m.capitalize()}")
        ax.axhline(1024, color="#C00000", linewidth=0.6, linestyle="--")
        # Inline cap annotation, placed above the line in the upper-left of
        # each panel so it never overlaps the boxes or the line itself.
        ax.text(0.01, 1024, "  1024-token cap", color="#C00000",
                fontsize=7.5, va="bottom", ha="left",
                transform=ax.get_yaxis_transform())

        for pos, vals in zip(positions, bp_data):
            if not vals:
                continue
            mean = float(np.mean(vals))
            ax.scatter([pos], [mean], marker="D", color="white", s=24,
                       edgecolor="black", zorder=4, linewidths=0.6)
            ax.text(pos + 0.32, mean, f"μ={mean:.0f}",
                    fontsize=7, color="#333", va="center")

    axes[0].set_ylabel("Output tokens per response")
    for ax in axes:
        ax.set_ylim(-30, 1140)
    fig.suptitle("Output length distribution by prompt condition (GSM8K, n=500)",
                 y=1.02, fontsize=12)
    fig.text(0.5, -0.05,
             "Instruct + explicit_no_cot ≈ 6 median tokens (true answer-only). "
             "Thinking + explicit_no_cot ≈ 491 median — model ignores the no-CoT instruction.",
             ha="center", fontsize=8.5, style="italic", color="#444")
    fig.tight_layout()
    out = OUT / "fig5_output_lengths.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def fig0_methods_pipeline() -> None:
    """6-panel methods pipeline figure for the paper."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 140)
    ax.set_axis_off()

    def box(x, y, w, h, text, color, text_color="white", fontsize=9.5,
            weight="normal", align="center"):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.0,rounding_size=1.2",
                              linewidth=1.0, edgecolor="black",
                              facecolor=color, zorder=2)
        ax.add_patch(rect)
        if align == "left":
            ax.text(x + 2.0, y + h / 2, text, ha="left", va="center",
                    color=text_color, fontsize=fontsize, weight=weight,
                    zorder=3, family="monospace")
        else:
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                    color=text_color, fontsize=fontsize, weight=weight,
                    zorder=3)

    def arrow(x1, y1, x2, y2):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle="->", color="#333", lw=1.2,
                            mutation_scale=14, zorder=1)
        ax.add_patch(a)

    def panel_label(y, n, title):
        ax.text(2.5, y, f"Panel {n}", fontsize=8, weight="bold", color="#666",
                va="center", ha="left")
        ax.text(11, y, title, fontsize=10.5, weight="bold", color="#222",
                va="center", ha="left", style="italic")

    # Layout grid (top-down, y in figure coords):
    #   panel 1: label@135, boxes 124-132
    #   panel 2: label@115, boxes 102-112, caption@98
    #   panel 3: label@93,  box   71-91
    #   panel 4: label@66,  box   48-63
    #   panel 5: label@43,  boxes 30-40
    #   panel 6: label@22,  box   8-19

    # --- Panel 1: Inputs ---
    panel_label(135, 1, "Datasets")
    box(15, 124, 33, 8, "GSM8K\nn=50 deep-table  •  n=500 wide",
        INSTRUCT_COLOR, fontsize=9)
    box(52, 124, 33, 8, "StrategyQA\nn=50 deep-table",
        INSTRUCT_COLOR, fontsize=9)
    arrow(31.5, 124, 31.5, 113)
    arrow(68.5, 124, 68.5, 113)

    # --- Panel 2: 6 conditions per question ---
    panel_label(115, 2, "6 conditions per question  (2 models × 3 prompts)")
    box(6, 100, 28, 12, "Qwen3-4B Instruct", TEAL, fontsize=9)
    box(36, 100, 28, 12, "Qwen3-4B Thinking", TEAL, fontsize=9)
    box(66, 100, 28, 12,
        "Prompts\nexplicit_cot\nexplicit_no_cot\nneutral_strict",
        TEAL, fontsize=7.5)
    ax.text(50, 96.5,
            "Deterministic decoding (greedy) • caps: 1024 baseline / 1536 mech / 384 probe",
            ha="center", fontsize=7.5, style="italic", color="#444")
    arrow(50, 100, 50, 92)

    # --- Panel 3: Per-question feature vector ---
    panel_label(93, 3, "Per-question feature vector  (6 dims, z-scored per model)")
    box(8, 71, 84, 20,
        "1. Latency per output token             ← timing\n"
        "2. Token entropy mean                   ← attention probe over response\n"
        "3. Token entropy slope                  ← attention probe over response\n"
        "4. Paraphrase consistency               ← build_paraphrases.py\n"
        "5. Perturbation Δ accuracy              ← build_perturbations.py\n"
        "6. Mechanistic Δ accuracy               ← anchor zero-out  (n=50 only)",
        FEATURE_COLOR_DARK, fontsize=8.5, align="left")
    arrow(50, 71, 50, 64)

    # --- Panel 4: HCDS computation ---
    panel_label(66, 4, "HCDS computation  (per question)")
    box(15, 48, 70, 15,
        "f_neutral, f_no_cot, f_cot     (one z-vector per prompt)\n\n"
        "HCDS_q  =  D(f_neutral, f_no_cot)  −  D(f_neutral, f_cot)\n\n"
        "Euclidean distance • partial-feature aware",
        HCDS_BLUE, fontsize=9)
    arrow(50, 48, 50, 41)

    # --- Panel 5: Statistical testing ---
    panel_label(43, 5, "Statistical testing  (per model × dataset)")
    box(13, 28, 33, 11,
        "Bootstrap CI\n1000 resamples • seed=17\n2.5% / 97.5% percentile",
        PURPLE, fontsize=8)
    box(54, 28, 33, 11,
        "One-sample t-test\nH₀: HCDS = 0\ntwo-sided",
        PURPLE, fontsize=8)
    arrow(29.5, 28, 29.5, 22)
    arrow(70.5, 28, 70.5, 22)

    # --- Panel 6: Verdict ---
    panel_label(22, 6, "Headline verdict")
    box(8, 7, 84, 12,
        "GSM8K n=500     Instruct  +2.38  (p=2e-141)         Thinking  +0.33  (p=1.9e-7)\n\n"
        "StrategyQA n=50    Instruct  +1.87  (p=4e-12)        Thinking  +0.60  (p=2e-3)",
        GREEN, fontsize=8.5, weight="bold")

    fig.suptitle("AJAR — HCDS methodology pipeline",
                 fontsize=14, weight="bold", y=0.985)

    out = OUT / "fig0_methods_pipeline.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    print("Building paper figures...")
    fig0_methods_pipeline()
    fig1_cross_dataset()
    fig2_robustness()
    fig3_length_matched()
    fig4_anchor_control()
    fig5_output_lengths()
    print("Done.")


if __name__ == "__main__":
    main()
