#!/usr/bin/env python3
"""Consolidate every committed result CSV into a single team-friendly
spreadsheet for the ICML push.

Outputs (under ``results/runs/<run_id>/team_spreadsheet/``):
  * ``per_question_features.csv`` — Task 6 wide table (300 rows: 50q × 2m
    × 3p), one column per metric. This is the "drop-into-pandas" sheet.
  * ``per_question_hcds.csv`` — joined view of features + per-question
    HCDS for the (model, neutral) vs (model, no_cot/cot) pairs.
  * ``hcds_summary_all.csv`` — every HCDS variant we've computed
    (full, no-paraphrase, no-mech, no-entropy, no-perturb,
    no-paraphrase-no-mech) stacked into one table for at-a-glance
    robustness comparison.
  * ``anchor_sensitivity.csv`` — Task 10 anchor-vs-control deltas,
    one row per (model, prompt).
  * ``length_matched_hcds.csv`` — HCDS recomputed within length tiers
    (Task 11 robustness slice).
  * ``baseline_wide_n500.csv`` — accuracy + latency from the n=500
    wide baseline run for cross-checking the deep-table-50 patterns.
  * ``run_index.csv`` — every committed run with its purpose, scope,
    outcome, and artefact pointers.
  * ``README.md`` — one-page guide to the spreadsheet.

Usage:
    python3 scripts/build_team_spreadsheet.py \
        --run-id 2026-05-04_2232_gsm8k50_qwen3-4b_deep-table

The script is read-only on existing artefacts; everything it writes
lands under the ``team_spreadsheet/`` subdirectory of the named run.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_ID = "2026-05-04_2232_gsm8k50_qwen3-4b_deep-table"
WIDE_BASELINE_RUN_ID = (
    "2026-05-04_1856_gsm8k500_qwen3-4b_omlx_baseline-neutral-strict-1024"
)


HCDS_VARIANTS = [
    ("full", "hcds_summary.csv"),
    ("no_paraphrase", "hcds_summary_no_paraphrase.csv"),
    ("no_mech", "hcds_summary_no_mech.csv"),
    ("no_entropy", "hcds_summary_no_entropy.csv"),
    ("no_perturb", "hcds_summary_no_perturb.csv"),
    ("no_paraphrase_no_mech", "hcds_summary_no_para_no_mech.csv"),
    ("entropy_only", "hcds_summary_entropy_only.csv"),
    ("latency_only", "hcds_summary_latency_only.csv"),
]


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        path.write_text("")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def stack_hcds_variants(deep_table_dir: Path) -> list[dict]:
    """Stack all HCDS-summary variants into one wide table."""
    out: list[dict] = []
    for variant_name, fname in HCDS_VARIANTS:
        rows = read_csv_rows(deep_table_dir / fname)
        for r in rows:
            out.append({
                "variant": variant_name,
                "model": r.get("model"),
                "n_questions": r.get("n_questions"),
                "mean_hcds": r.get("mean_hcds"),
                "ci_low_95": r.get("ci_low_95"),
                "ci_high_95": r.get("ci_high_95"),
                "t_statistic": r.get("t_statistic"),
                "p_value_two_sided": r.get("p_value_two_sided"),
                "n_features_avg": r.get("n_features_avg"),
                "verdict": r.get("verdict"),
            })
    return out


def join_features_and_hcds(features: list[dict], hcds_q: list[dict]) -> list[dict]:
    """Add per-question HCDS to the features table where applicable."""
    hcds_by_key = {(r["model"], r["sample_idx"]): r for r in hcds_q}
    out = []
    for f in features:
        k = (f["model"], f["sample_idx"])
        h = hcds_by_key.get(k, {})
        merged = dict(f)
        merged["hcds_q"] = h.get("hcds_q", "")
        merged["d_neutral_to_no_cot"] = h.get("d_n_nocot", "")
        merged["d_neutral_to_cot"] = h.get("d_n_cot", "")
        out.append(merged)
    return out


def read_wide_baseline_summary() -> list[dict]:
    """Pull the n=500 wide-baseline condition_summary if present.

    The wide-baseline analysis lives under ``results/runs/<run_id>/``
    (committed) rather than ``outputs/`` (gitignored).
    """
    primary = (
        REPO_ROOT / "results" / "runs" / WIDE_BASELINE_RUN_ID
        / "condition_summary.csv"
    )
    if primary.exists():
        return read_csv_rows(primary)
    fallback = (
        REPO_ROOT / "outputs" / WIDE_BASELINE_RUN_ID / "condition_summary.csv"
    )
    return read_csv_rows(fallback)


def build_run_index(run_ids_with_dirs: list[tuple[str, Path, str, str]]) -> list[dict]:
    rows = []
    for rid, rdir, purpose, outcome in run_ids_with_dirs:
        rows.append({
            "run_id": rid,
            "exists": "yes" if rdir.exists() else "no",
            "purpose": purpose,
            "outcome": outcome,
            "artefact_dir": str(rdir.relative_to(REPO_ROOT)) if rdir.exists() else "",
        })
    return rows


def write_readme(out_dir: Path, deep_table_run: str) -> None:
    text = f"""# AJAR — Team Spreadsheet

Auto-generated by `scripts/build_team_spreadsheet.py` from the
`{deep_table_run}` run plus the n=500 wide baseline. Open the CSVs
directly in Excel / Sheets / pandas; no further processing required.

## What's in here

| File | Rows | Use it for |
|---|---|---|
| `per_question_features.csv` | 300 (50q × 2m × 3p) | The Task 6 wide table. Every metric the proposal asks for, one row per (question, model, prompt). |
| `per_question_hcds.csv` | 300 | Same as above, plus per-question HCDS and the two distance components D(neutral, no_cot) / D(neutral, cot). |
| `hcds_summary_all.csv` | 12 (2 models × 6 variants) | At-a-glance HCDS robustness check — full vs every leave-one-feature-out variant. |
| `anchor_sensitivity.csv` | 6 | Task 10. Anchor-suppression vs control-suppression accuracy drops per condition. |
| `length_matched_hcds.csv` | up to 6 (2m × 3 length tiers) | Task 11 robustness slice. HCDS recomputed within length-binned subsets. |
| `baseline_wide_n500.csv` | 6 | Accuracy + latency aggregates from the n=500 wide baseline, for cross-checking the deep-table-50 results at scale. |
| `run_index.csv` | 5 | Every committed run with purpose, outcome, and where to look on disk. |

## Headline numbers (already in `hcds_summary_all.csv`)

Both models show statistically significant positive HCDS — neutral
prompting aligns more with explicit-CoT than with explicit-no-CoT:

| Model | Mean HCDS | 95% CI | t-stat | p-value |
|---|---|---|---|---|
| Instruct | +2.25 | [+1.69, +2.85] | +7.96 | 2.2e-10 |
| Thinking | +0.48 | [+0.07, +0.86] | +2.39 | 0.021 |

## Where this came from

- Deep-table run: `results/runs/{deep_table_run}/`
- Wide baseline: `outputs/{WIDE_BASELINE_RUN_ID}/`
- Headline figure: `results/runs/{deep_table_run}/hcds_figure.png`
- Full prose summary: `results/runs/{deep_table_run}/SUMMARY.md`
- Team-facing handoff: `HANDOFF.md` on `main`

## Regenerate

```
python3 scripts/build_team_spreadsheet.py
```
"""
    (out_dir / "README.md").write_text(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default=DEFAULT_RUN_ID)
    ap.add_argument(
        "--out-subdir",
        default="team_spreadsheet",
        help="Subdirectory under the run dir to write outputs to.",
    )
    args = ap.parse_args()

    run_dir = REPO_ROOT / "results" / "runs" / args.run_id
    if not run_dir.exists():
        print(f"error: run dir not found: {run_dir}", file=sys.stderr)
        return 2

    out_dir = run_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- per_question_features ---
    features = read_csv_rows(run_dir / "task6_table.csv")
    write_csv(out_dir / "per_question_features.csv", features)
    print(f"wrote per_question_features.csv ({len(features)} rows)")

    # --- per_question_hcds ---
    hcds_q = read_csv_rows(run_dir / "hcds_per_question.csv")
    joined = join_features_and_hcds(features, hcds_q)
    write_csv(out_dir / "per_question_hcds.csv", joined)
    print(f"wrote per_question_hcds.csv ({len(joined)} rows)")

    # --- hcds_summary_all ---
    stacked = stack_hcds_variants(run_dir)
    write_csv(out_dir / "hcds_summary_all.csv", stacked)
    print(f"wrote hcds_summary_all.csv ({len(stacked)} rows)")

    # --- anchor_sensitivity ---
    anchor = read_csv_rows(run_dir / "task10_anchor_sensitivity.csv")
    write_csv(out_dir / "anchor_sensitivity.csv", anchor)
    print(f"wrote anchor_sensitivity.csv ({len(anchor)} rows)")

    # --- length_matched_hcds ---
    length_matched = read_csv_rows(run_dir / "hcds_length_matched.csv")
    write_csv(out_dir / "length_matched_hcds.csv", length_matched)
    print(f"wrote length_matched_hcds.csv ({len(length_matched)} rows)")

    # --- baseline_wide_n500 ---
    wide = read_wide_baseline_summary()
    write_csv(out_dir / "baseline_wide_n500.csv", wide)
    print(f"wrote baseline_wide_n500.csv ({len(wide)} rows)")

    # --- run_index ---
    runs_root = REPO_ROOT / "results" / "runs"
    outputs_root = REPO_ROOT / "outputs"
    runs = [
        (
            args.run_id,
            run_dir,
            "Deep-table — 50 GSM8K q × 2 models × 3 prompts, full feature vector",
            "Headline: HCDS positive both models (Instruct p=2.2e-10, Thinking p=0.021)",
        ),
        (
            WIDE_BASELINE_RUN_ID,
            outputs_root / WIDE_BASELINE_RUN_ID,
            "Wide baseline (n=500) for cross-checking deep-table accuracy at scale",
            "3000/3000 generations clean; HCDS-clean neutral_strict prompt",
        ),
        (
            "2026-05-04_0238_gsm8k500_qwen3-4b_omlx_baseline-rerun-nocot-1024",
            outputs_root / "2026-05-04_0238_gsm8k500_qwen3-4b_omlx_baseline-rerun-nocot-1024",
            "n=500 baseline rerun with 1024-token cap and explicit_no_cot prompt",
            "Identified contamination in the original neutral prompt",
        ),
        (
            "2026-05-04_1720_gsm8k1_qwen3-4b_torch_mechanistic-smoke",
            outputs_root / "2026-05-04_1720_gsm8k1_qwen3-4b_torch_mechanistic-smoke",
            "MI smoke test — verifies torch backend produces tokens / step scores / interventions",
            "Backend verified working",
        ),
        (
            "2026-05-03_2223_gsm8k500_qwen3-4b_omlx_baseline-v1",
            outputs_root / "2026-05-03_2223_gsm8k500_qwen3-4b_omlx_baseline-v1",
            "Initial baseline (superseded — Thinking truncation issue at 512 tokens)",
            "Issue identified, run replaced",
        ),
    ]
    write_csv(out_dir / "run_index.csv", build_run_index(runs))
    print("wrote run_index.csv")

    write_readme(out_dir, args.run_id)
    print(f"wrote README.md")

    print()
    print(f"Team spreadsheet at: {out_dir.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
