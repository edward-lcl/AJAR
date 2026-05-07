#!/usr/bin/env python3
"""Re-extract predictions on existing baseline.json files using the
current parser. Useful when a parser bug is fixed AFTER generation
(so we don't have to re-run the generation step).

Walks the per-sample baseline.json files under a run dir, re-runs the
extractor on the saved generated_text, updates ``prediction_numeric``
(plus ``prediction_boxed_content`` and ``correct``), and writes back.

Usage:
    AJAR_DATASET_KIND=strategyqa python3 scripts/repair_predictions.py \\
        --run-dir outputs/<run_id> \\
        [--dry-run]

Specifically built for the StrategyQA run where the original parser
didn't recognise the 0=no / 1=yes binary mapping under
explicit_no_cot. With AJAR_DATASET_KIND=strategyqa the runner module
loads the yes/no extractor; this script reuses that.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_PATH = REPO_ROOT / "scripts" / "run_qwen3_gsm8k_mi.py"


def load_runner_module(kind: str):
    """Import the runner with AJAR_DATASET_KIND set so the extractor
    is bound to the right pair (numeric vs yes/no)."""
    os.environ["AJAR_DATASET_KIND"] = kind
    spec = importlib.util.spec_from_file_location("ajar_runner_repair", RUNNER_PATH)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ajar_runner_repair"] = m
    spec.loader.exec_module(m)
    return m


def repair_one(path: Path, runner, dry_run: bool) -> tuple[bool, str]:
    """Re-extract on a single baseline.json. Returns (changed, reason)."""
    try:
        d = json.loads(path.read_text())
    except Exception as e:
        return False, f"unreadable: {e}"

    text = d.get("generated_text") or ""
    gold = d.get("gold_numeric")
    old_pred = d.get("prediction_numeric")
    old_correct = bool(d.get("correct"))

    new_pred, new_boxed = runner.extract_predicted_number(text)
    new_correct = (new_pred is not None) and (new_pred == gold)

    if new_pred == old_pred and new_correct == old_correct:
        return False, "no-change"

    if dry_run:
        return True, f"would-change: pred {old_pred!r}->{new_pred!r}, correct {old_correct}->{new_correct}"

    d["prediction_numeric"] = new_pred
    d["prediction_boxed_content"] = new_boxed
    d["correct"] = new_correct
    path.write_text(json.dumps(d, indent=2, ensure_ascii=False))
    return True, f"updated: pred {old_pred!r}->{new_pred!r}, correct {old_correct}->{new_correct}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument(
        "--kind",
        default="strategyqa",
        choices=["strategyqa", "gsm8k"],
        help="Dataset kind (chooses extractor pair).",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--filename",
        default="baseline.json",
        help="Per-sample file to update (default: baseline.json).",
    )
    args = ap.parse_args()

    if not args.run_dir.exists():
        print(f"error: run-dir not found: {args.run_dir}", file=sys.stderr)
        return 2

    runner = load_runner_module(args.kind)
    files = sorted(args.run_dir.rglob(args.filename))
    print(f"found {len(files)} {args.filename} files under {args.run_dir}")

    total = 0
    changed = 0
    by_condition: dict[str, dict[str, int]] = {}
    for f in files:
        total += 1
        did_change, reason = repair_one(f, runner, args.dry_run)
        # condition label: e.g. instruct/explicit_no_cot
        rel = f.relative_to(args.run_dir)
        cond = "/".join(rel.parts[:-2]) if len(rel.parts) >= 2 else "(unknown)"
        bc = by_condition.setdefault(cond, {"updated": 0, "unchanged": 0})
        if did_change:
            changed += 1
            bc["updated"] += 1
        else:
            bc["unchanged"] += 1

    print(f"\n{'(DRY RUN) ' if args.dry_run else ''}summary: {changed}/{total} files changed")
    print()
    print(f"{'condition':<48} {'updated':>8} {'unchanged':>10}")
    for cond in sorted(by_condition):
        c = by_condition[cond]
        print(f"{cond:<48} {c['updated']:>8} {c['unchanged']:>10}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
