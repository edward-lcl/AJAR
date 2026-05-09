"""Compute joint_anchor_drop − joint_control_drop from the joint
intervention run.

Inputs:  outputs/joint_anchor_intervention/joint_results.jsonl
Outputs:
  results/runs/joint_anchor/joint_anchor_summary.csv
  Pretty-printed comparison vs the existing single-anchor result.

Existing single-anchor numbers for Thinking + explicit_cot on
GSM8K n=50 (from task10_anchor_sensitivity.csv):
  baseline_acc   = 0.840
  anchor_drop    = 0.105   (single anchor, mean over both anchors and both modes)
  control_drop   = 0.380   (single random control, both modes)
  anchor−control = -0.275  (the negative bar we are testing)
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path("/Users/edward/Projects/AJAR")
INPUT = ROOT / "outputs/joint_anchor_intervention/joint_results.jsonl"
OUT = ROOT / "results/runs/joint_anchor"
OUT.mkdir(parents=True, exist_ok=True)

# Existing single-anchor / single-control numbers for context
EXISTING = {
    "baseline_acc": 0.840,
    "anchor_drop_single": 0.105,
    "control_drop_single": 0.380,
    "anchor_minus_control_single": -0.275,
}


def load_results(path: Path) -> List[Dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def summarize() -> None:
    rows = load_results(INPUT)
    if not rows:
        print("No results found.")
        return

    # Group by (sample, kind) and average across modes
    by_sample_kind: Dict[Tuple[int, str], List[int]] = defaultdict(list)
    baseline_correct: Dict[int, int] = {}
    errors = 0
    for r in rows:
        if "error" in r:
            errors += 1
            continue
        sid = int(r["sample_idx"])
        kind = r["target_kind"]
        baseline_correct[sid] = int(r["baseline_correct"])
        by_sample_kind[(sid, kind)].append(int(r["intervention_correct"]))

    # Per-sample, kind-averaged intervention accuracy
    per_sample_anchor_acc: Dict[int, float] = {}
    per_sample_control_acc: Dict[int, float] = {}
    for (sid, kind), vals in by_sample_kind.items():
        acc = statistics.mean(vals)
        if kind == "anchor":
            per_sample_anchor_acc[sid] = acc
        elif kind == "control":
            per_sample_control_acc[sid] = acc

    common = sorted(set(per_sample_anchor_acc) & set(per_sample_control_acc))
    if not common:
        print("No samples have both anchor and control results.")
        return

    baseline_acc = statistics.mean(baseline_correct[s] for s in common)
    joint_anchor_acc = statistics.mean(per_sample_anchor_acc[s] for s in common)
    joint_control_acc = statistics.mean(per_sample_control_acc[s] for s in common)
    joint_anchor_drop = baseline_acc - joint_anchor_acc
    joint_control_drop = baseline_acc - joint_control_acc
    joint_amc = joint_anchor_drop - joint_control_drop

    print(f"Samples used: {len(common)}  (errors: {errors})")
    print(f"\nResult: Thinking + explicit_cot on GSM8K n=50")
    print(f"{'metric':<35s}{'single (existing)':>20s}{'joint (new)':>15s}")
    print("-" * 70)
    print(f"{'baseline accuracy':<35s}{EXISTING['baseline_acc']:>20.3f}{baseline_acc:>15.3f}")
    print(f"{'anchor drop':<35s}{EXISTING['anchor_drop_single']:>20.3f}{joint_anchor_drop:>15.3f}")
    print(f"{'control drop':<35s}{EXISTING['control_drop_single']:>20.3f}{joint_control_drop:>15.3f}")
    print(f"{'anchor − control contrast':<35s}{EXISTING['anchor_minus_control_single']:>20.3f}{joint_amc:>15.3f}")
    print()

    # Per-mode breakdown for diagnostic
    by_mode: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for r in rows:
        if "error" in r:
            continue
        by_mode[(r["target_kind"], r["intervention_mode"])].append(int(r["intervention_correct"]))
    print("Per-mode intervention accuracy:")
    for (kind, mode), vals in sorted(by_mode.items()):
        print(f"  {kind:>7s} × {mode:<18s} acc={statistics.mean(vals):.3f}  n={len(vals)}")

    # Save summary CSV
    out_path = OUT / "joint_anchor_summary.csv"
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "single (existing)", "joint (new)"])
        w.writerow(["baseline_accuracy", EXISTING["baseline_acc"], round(baseline_acc, 4)])
        w.writerow(["anchor_drop", EXISTING["anchor_drop_single"], round(joint_anchor_drop, 4)])
        w.writerow(["control_drop", EXISTING["control_drop_single"], round(joint_control_drop, 4)])
        w.writerow(["anchor_minus_control", EXISTING["anchor_minus_control_single"], round(joint_amc, 4)])
        w.writerow(["n_samples", 50, len(common)])

    print(f"\nWrote {out_path}")

    # Interpretation hint
    print("\nInterpretation:")
    if joint_amc > EXISTING["anchor_minus_control_single"] + 0.10:
        print("  Joint anchor-control contrast is meaningfully MORE positive than single.")
        print("  → Hypothesis A (distributed causal load) supported.")
    elif abs(joint_amc - EXISTING["anchor_minus_control_single"]) <= 0.10:
        print("  Joint anchor-control contrast is similar to single.")
        print("  → Hypothesis B (anchor scoring picks late summary steps) consistent.")
    else:
        print("  Joint anchor-control contrast is MORE negative than single.")
        print("  → Hypothesis C (truly diffuse / anchor scoring fails) consistent.")


if __name__ == "__main__":
    summarize()
