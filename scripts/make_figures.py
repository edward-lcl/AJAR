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


def _draw_database_icon(ax, cx, cy, w=4, h=5, color="#5B7C99"):
    """Tiny database stack (3 cylinders)."""
    from matplotlib.patches import Ellipse, Rectangle
    layer_h = h / 3
    for i in range(3):
        y0 = cy - h / 2 + i * layer_h
        ax.add_patch(Rectangle((cx - w / 2, y0 + 0.15), w, layer_h - 0.3,
                                facecolor=color, edgecolor="black",
                                linewidth=0.6, zorder=4))
        ax.add_patch(Ellipse((cx, y0 + layer_h - 0.15), w, 0.6,
                              facecolor=color, edgecolor="black",
                              linewidth=0.6, zorder=5))


def _draw_robot_icon(ax, cx, cy, w=3.6, h=3.6, color="#3F8E8C"):
    """Tiny robot head (rectangle with eyes and antenna)."""
    from matplotlib.patches import Circle, Rectangle
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h * 0.85,
                            facecolor=color, edgecolor="black",
                            linewidth=0.7, zorder=4))
    ax.add_patch(Rectangle((cx - 0.15, cy + h / 2 * 0.85), 0.3, 0.6,
                            facecolor="black", zorder=5))
    ax.add_patch(Circle((cx, cy + h / 2 * 0.85 + 0.7), 0.25,
                         facecolor="#222", zorder=5))
    # Eyes
    eye_y = cy + h * 0.05
    ax.add_patch(Circle((cx - w * 0.20, eye_y), 0.32, facecolor="white",
                         edgecolor="black", linewidth=0.4, zorder=5))
    ax.add_patch(Circle((cx + w * 0.20, eye_y), 0.32, facecolor="white",
                         edgecolor="black", linewidth=0.4, zorder=5))
    ax.add_patch(Circle((cx - w * 0.20, eye_y), 0.13, facecolor="black", zorder=6))
    ax.add_patch(Circle((cx + w * 0.20, eye_y), 0.13, facecolor="black", zorder=6))


def _draw_gear_icon(ax, cx, cy, r=2.0, color="#B07820"):
    """Simple gear shape (12-tooth approximation)."""
    from matplotlib.patches import Circle, Polygon
    n_teeth = 12
    inner_r = r * 0.78
    outer_r = r
    pts: List[tuple] = []
    for i in range(n_teeth * 2):
        ang = i * np.pi / n_teeth
        rr = outer_r if (i // 2) % 2 == 0 else inner_r
        pts.append((cx + rr * np.cos(ang), cy + rr * np.sin(ang)))
    ax.add_patch(Polygon(pts, facecolor=color, edgecolor="black",
                          linewidth=0.7, zorder=4))
    ax.add_patch(Circle((cx, cy), r * 0.32, facecolor="white",
                         edgecolor="black", linewidth=0.5, zorder=5))


def _draw_feature_grid(ax, cx, cy, w=12, h=4.5, color="#B07820"):
    """6-cell horizontal grid representing the feature vector."""
    from matplotlib.patches import Rectangle
    cell_w = w / 6
    for i in range(6):
        x0 = cx - w / 2 + i * cell_w
        ax.add_patch(Rectangle((x0, cy - h / 2), cell_w, h,
                                facecolor=color, edgecolor="black",
                                linewidth=0.6, zorder=4))
        ax.text(x0 + cell_w / 2, cy, str(i + 1),
                ha="center", va="center", color="white",
                fontsize=10, weight="bold", zorder=5)


def _draw_mini_bars(ax, cx, cy, color="#6E5797", w=8, h=4):
    """A tiny bar chart with error bars to suggest stats output."""
    from matplotlib.patches import Rectangle
    heights = [3.0, 0.8, 2.4, 0.5]
    bw = w / (len(heights) * 1.6)
    base_y = cy - h / 2
    for i, hh in enumerate(heights):
        x0 = cx - w / 2 + i * (bw * 1.6) + bw * 0.3
        bar_h = hh / max(heights) * h * 0.85
        ax.add_patch(Rectangle((x0, base_y), bw, bar_h,
                                facecolor=color, edgecolor="black",
                                linewidth=0.4, zorder=4))
        ax.plot([x0 + bw / 2, x0 + bw / 2],
                [base_y + bar_h - 0.4, base_y + bar_h + 0.5],
                color="black", lw=0.7, zorder=5)


def fig0_methods_pipeline() -> None:
    """3-row methods pipeline (Setup → Pipeline → Result).

    Wide landscape figure. Three horizontal bands separated by clear
    gaps with directional arrows. Every card uses the same pattern:
    colored title-strip on top, white body below.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 100)
    ax.set_axis_off()

    def box(x, y, w, h, text, color, text_color="white", fontsize=9.5,
            weight="normal", align="center", line_spacing=None):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.0,rounding_size=0.9",
                              linewidth=1.0, edgecolor="black",
                              facecolor=color, zorder=2)
        ax.add_patch(rect)
        if align == "left":
            ax.text(x + 1.5, y + h / 2, text, ha="left", va="center",
                    color=text_color, fontsize=fontsize, weight=weight,
                    zorder=3, family="monospace",
                    linespacing=line_spacing or 1.4)
        else:
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                    color=text_color, fontsize=fontsize, weight=weight,
                    zorder=3, linespacing=line_spacing or 1.3)

    def arrow(x1, y1, x2, y2, lw=1.4, color="#333"):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle="-|>", color=color, lw=lw,
                            mutation_scale=12, zorder=1)
        ax.add_patch(a)

    def card(x, y, w, body_h, title, body, color, *,
             body_fontsize=8.5, title_fontsize=10, body_align="left",
             body_line_spacing=1.35, title_h=5.5):
        """Title-strip + white body card — single consistent pattern."""
        # Title strip
        box(x, y + body_h, w, title_h, title, color,
            fontsize=title_fontsize, weight="bold")
        # White body
        box(x, y, w, body_h, body, "white", text_color="#222",
            fontsize=body_fontsize, align=body_align,
            line_spacing=body_line_spacing)

    def section_divider(y, text, color):
        """Section divider — accent line spans full card width with label
        on the left, padded so label + line don't overlap any card."""
        # Background line first
        ax.plot([2, 138], [y, y], color=color, lw=1.6, zorder=1,
                solid_capstyle="round", alpha=0.4)
        # Label with white background to mask the line behind it
        ax.text(4, y, text, ha="left", va="center", fontsize=11.5,
                weight="bold", color=color, zorder=3,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor="none"))

    def big_arrow(x, y_top, y_bot, label=None, lw=2.2):
        a = FancyArrowPatch((x, y_top), (x, y_bot),
                            arrowstyle="-|>", color="#444", lw=lw,
                            mutation_scale=22, zorder=1)
        ax.add_patch(a)
        if label:
            ax.text(x + 1.5, (y_top + y_bot) / 2, label,
                    fontsize=8, style="italic", color="#444", va="center")

    # Coordinate plan (y 0-110, top-down):
    #   title block        100 - 108
    #   SETUP label         97
    #   SETUP cards         75 - 95     (body 75-89, title 89-95)  body_h=14
    #   gap + arrow         68 - 74
    #   PIPELINE label      66
    #   PIPELINE caption    63
    #   PIPELINE sub cards  40 - 60     (body 40-56, title 56-60)  body_h=16
    #   agg arrow + label   33 - 39
    #   gap + arrow         26 - 32
    #   RESULT label        24
    #   RESULT cards         3 - 22     (body 3-17, title 17-22)   body_h=14
    card_w = 41

    # ==========================================
    # ROW 1 — SETUP
    # ==========================================
    section_divider(97, "SETUP", INSTRUCT_COLOR)

    setup_body_h = 16
    setup_y = 73

    card(4, setup_y, card_w, setup_body_h, "Datasets",
         "GSM8K\n  n=50    deep · full 6-feature pipeline\n"
         "  n=500   wide · 5-feature extension\n\n"
         "StrategyQA\n  n=50    deep · full 6-feature pipeline",
         INSTRUCT_COLOR, body_fontsize=8.5)

    card(49, setup_y, card_w, setup_body_h, "Models",
         "Qwen3-4B-Instruct-2507\n     (standard instruction-tuned)\n\n"
         "Qwen3-4B-Thinking\n     (reasoning-tuned variant)\n\n"
         "Greedy decoding   ·   seed = 17",
         TEAL, body_fontsize=8.5)

    card(94, setup_y, card_w, setup_body_h, "Prompts (3)",
         "explicit_cot          CoT-encouraging prefix\n\n"
         "explicit_no_cot       answer-only directive\n\n"
         "neutral_strict        minimal task-only\n\n"
         "Caps:  1024 base  ·  1536 mech  ·  384 probe",
         TEAL, body_fontsize=7.8)

    # ==========================================
    # GAP 1: Setup → Pipeline
    # ==========================================
    big_arrow(70, 74, 68, lw=2.0)
    ax.text(72, 71,
            "run  2 × 3 × n  =  300 cells (n=50)  /  3000 cells (n=500)",
            fontsize=8.5, style="italic", color="#444", va="center")

    # ==========================================
    # ROW 2 — PIPELINE
    # ==========================================
    section_divider(66, "PIPELINE", FEATURE_COLOR_DARK)

    ax.text(70, 62.5,
            "For each cell, four parallel sub-runs each produce a slice of the feature vector:",
            ha="center", fontsize=9, style="italic", color="#444")

    # 4 sub-cards spanning the same horizontal extent as the 3 SETUP cards
    # (x = 4 → 135). w=31, gap≈2.3.
    sub_w = 31
    sub_body_h = 16
    sub_y = 40
    sub_specs = [
        (4,
         "Baseline generation",
         "Greedy generation +\ntoken-level attention probe\n\n"
         "Produces:\n   · latency / token\n   · entropy mean & slope"),
        (37.3,
         "Paraphrase  (×2 / q)",
         "Re-run on 2 paraphrased\nversions per question\n\n"
         "Produces:\n   · paraphrase\n      consistency"),
        (70.7,
         "Perturbation  (×N / q)",
         "Re-run with distractor-\nirrelevant perturbations\n\n"
         "Produces:\n   · perturbation\n      Δ accuracy"),
        (104,
         "Mechanistic  (n=50)",
         "Anchor + control\nzero-out interventions\n\n"
         "Produces:\n   · mechanistic\n      Δ accuracy"),
    ]
    for x0, title, body in sub_specs:
        card(x0, sub_y, sub_w, sub_body_h, title, body,
             FEATURE_COLOR_DARK, body_fontsize=7.5,
             title_fontsize=9.5, title_h=5)

    big_arrow(70, 40, 36, lw=1.8)
    ax.text(70, 33.5,
            "z-score per model across all (question × prompt) rows   →   "
            "feature vector  f(q, prompt)  ∈ ℝ⁶  for every cell",
            ha="center", fontsize=8.8, color="#222", weight="bold")

    # ==========================================
    # GAP 2: Pipeline → Result
    # ==========================================
    big_arrow(70, 31, 25, lw=2.0)

    # ==========================================
    # ROW 3 — RESULT
    # ==========================================
    section_divider(24, "RESULT", HCDS_BLUE)

    res_y = 2
    res_body_h = 16

    card(4, res_y, card_w, res_body_h, "HCDS  (per question)",
         "Three feature vectors per question:\n"
         "f_N ,  f_NoCoT ,  f_CoT\n\n"
         "HCDS_q  =  D(f_N , f_NoCoT)\n                  −  D(f_N , f_CoT)\n\n"
         "Euclidean  ·  partial-feature aware",
         HCDS_BLUE, body_fontsize=8.5)

    card(49, res_y, card_w, res_body_h, "Statistical aggregation",
         "Bootstrap CI\n   1000 resamples · seed = 17\n"
         "   2.5% / 97.5% percentile\n\n"
         "One-sample t-test\n   H₀ : HCDS = 0    (two-sided)\n\n"
         "Reported per (model, dataset)",
         PURPLE, body_fontsize=8)

    card(94, res_y, card_w, res_body_h, "Headline verdict",
         "GSM8K   n=500\n"
         "   Instruct   +2.38      p = 2e-141\n"
         "   Thinking  +0.33      p = 1.9e-7\n\n"
         "StrategyQA   n=50\n"
         "   Instruct   +1.87      p = 4e-12\n"
         "   Thinking  +0.60      p = 2e-3",
         GREEN, body_fontsize=8)

    # Title and subtitle intentionally omitted — the LaTeX figure caption
    # in the paper renders these so the figure stays caption-agnostic.

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
