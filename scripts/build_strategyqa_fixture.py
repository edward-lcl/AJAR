#!/usr/bin/env python
"""Build a StrategyQA fixture in GSM8K-shape JSONL for Task 4.

StrategyQA (Geva et al. 2021) is a yes/no commonsense reasoning
benchmark. Questions like "Could a llama survive a one-day chess
tournament?" with binary answers and multi-hop rationale strings.

This fixture loader does the minimal normalisation needed so the
existing AJAR runner can consume StrategyQA the same way it consumes
GSM8K:

  * ``question`` — the StrategyQA question text
  * ``answer``   — the rationale (joined facts), ending with the
    canonical GSM8K-style line ``#### yes`` or ``#### no``
  * downstream parsing of ``#### X`` will yield the literal string
    ``"yes"`` or ``"no"``, which the runner then compares against the
    model's extracted yes/no prediction (see
    ``AJAR_DATASET_KIND=strategyqa`` mode in the runner).

The script tries the canonical HuggingFace mirror (``ChilleD/StrategyQA``)
and falls back to ``voidful/StrategyQA``. If neither is reachable
offline, point ``--local-jsonl`` at a hand-curated JSONL with
``question`` and ``answer`` (yes/no) keys.

Output (under ``--out-dir``):
  * ``strategyqa.jsonl``  — n rows in GSM8K shape
  * ``index.csv``         — sample_idx → original strategyqa_id mapping

Usage:
    python3 scripts/build_strategyqa_fixture.py \\
        --num-samples 50 \\
        --out-dir data/fixtures/strategyqa_first50

Then to run the deep-table on it:
    GSM8K_JSONL=data/fixtures/strategyqa_first50/strategyqa.jsonl \\
    AJAR_DATASET_KIND=strategyqa \\
    bash scripts/run_deep_table.sh 50
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable


DEFAULT_HF_NAMES = [
    ("ChilleD/StrategyQA", "test"),
    ("ChilleD/StrategyQA", "train"),
    ("voidful/StrategyQA", "test"),
    ("voidful/StrategyQA", "train"),
    ("wics/strategy-qa", "test"),
]


def load_strategyqa(num_samples: int, local_jsonl: Path | None) -> list[dict]:
    """Return a list of StrategyQA records normalised to:
       {"qid": str, "question": str, "answer_yesno": "yes"|"no",
        "facts": list[str]}.

    Tries local JSONL first if provided; otherwise iterates the
    HuggingFace fallbacks in DEFAULT_HF_NAMES.
    """
    if local_jsonl is not None and local_jsonl.exists():
        rows = []
        with local_jsonl.open() as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                ans = r.get("answer")
                if isinstance(ans, bool):
                    ans = "yes" if ans else "no"
                elif isinstance(ans, str):
                    ans = ans.strip().lower()
                    if ans not in ("yes", "no"):
                        # try to coerce
                        ans = "yes" if ans in ("true", "1", "y") else "no"
                rows.append({
                    "qid": str(r.get("qid", r.get("id", len(rows)))),
                    "question": r["question"],
                    "answer_yesno": ans,
                    "facts": r.get("facts", []) or [],
                })
                if len(rows) >= num_samples:
                    break
        return rows

    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "`datasets` package not installed. `pip install datasets` or "
            "supply --local-jsonl."
        )

    last_error: Exception | None = None
    for name, split in DEFAULT_HF_NAMES:
        try:
            ds = load_dataset(name, split=split)
            print(f"loaded {name} [{split}] with {len(ds)} rows")
            rows = []
            for i, r in enumerate(ds):
                if i >= num_samples:
                    break
                # Field names vary across mirrors; try a few.
                question = r.get("question") or r.get("Question")
                answer = (
                    r.get("answer")
                    if "answer" in r
                    else r.get("Answer", r.get("answer_text"))
                )
                if isinstance(answer, bool):
                    answer = "yes" if answer else "no"
                elif isinstance(answer, str):
                    answer = answer.strip().lower()
                    if answer not in ("yes", "no"):
                        # Some mirrors use "True" / "False"
                        answer = "yes" if answer in ("true", "y", "1") else "no"
                facts = r.get("facts") or r.get("evidence") or []
                if isinstance(facts, str):
                    facts = [facts]
                qid = str(r.get("qid", r.get("id", i)))
                if question is None or answer is None:
                    continue
                rows.append({
                    "qid": qid,
                    "question": question,
                    "answer_yesno": answer,
                    "facts": facts,
                })
            if rows:
                return rows
        except Exception as e:  # noqa: BLE001
            last_error = e
            print(f"  {name} [{split}] not available: {e}")
            continue

    raise RuntimeError(
        f"Could not load StrategyQA from any HF mirror. "
        f"Last error: {last_error}. "
        f"Pass --local-jsonl pointing at a JSONL with "
        f"{{question, answer}} fields."
    )


def to_gsm8k_shape(rec: dict) -> dict:
    """Format a StrategyQA record as a GSM8K-shape JSONL row.

    The ``answer`` field is the rationale (joined facts) plus the
    canonical ``#### yes`` / ``#### no`` line that
    ``parse_gsm8k_answer`` will pick up.
    """
    facts = rec.get("facts") or []
    rationale = "\n".join(str(f) for f in facts) if facts else ""
    if rationale:
        full_answer = f"{rationale}\n#### {rec['answer_yesno']}"
    else:
        full_answer = f"#### {rec['answer_yesno']}"
    return {
        "question": rec["question"],
        "answer": full_answer,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-samples", type=int, default=50)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--local-jsonl",
        type=Path,
        default=None,
        help="Path to a local StrategyQA JSONL with {question, answer} keys.",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading StrategyQA (target n={args.num_samples})...")
    rows = load_strategyqa(args.num_samples, args.local_jsonl)
    print(f"got {len(rows)} rows")

    out_jsonl = args.out_dir / "strategyqa.jsonl"
    with out_jsonl.open("w") as f:
        for r in rows:
            f.write(json.dumps(to_gsm8k_shape(r)) + "\n")
    print(f"wrote {out_jsonl}")

    out_index = args.out_dir / "index.csv"
    with out_index.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_idx", "qid", "answer_yesno"])
        for i, r in enumerate(rows):
            w.writerow([i, r["qid"], r["answer_yesno"]])
    print(f"wrote {out_index}")

    yes_count = sum(1 for r in rows if r["answer_yesno"] == "yes")
    print(
        f"\nclass balance: yes={yes_count}/{len(rows)} "
        f"({yes_count / max(len(rows), 1):.1%})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
