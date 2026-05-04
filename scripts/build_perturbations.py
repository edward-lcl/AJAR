#!/usr/bin/env python
"""Build perturbed GSM8K variants for Task 2 / Task 9 (Perturbation Fragility).

Currently implements one operator: distractor_irrelevant. It inserts an
irrelevant numeric clause into the question without changing any quantity that
appears in the gold reasoning chain, so the canonical GSM8K answer remains
correct and Delta accuracy is interpretable as fragility to distractors.

Output:
- <out_dir>/variants.jsonl  GSM8K-shaped rows (question, answer) consumable by
  the runner via GSM8K_JSONL=<path>.
- <out_dir>/index.csv       Maps each row in variants.jsonl back to the original
  sample_idx, the perturbation op, and the variant index.

Usage:
    python3 scripts/build_perturbations.py \\
        --num-samples 50 --out-dir data/variants/gsm8k_test_first50_perturb
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset


DISTRACTOR_TEMPLATES = [
    "On the same day, the temperature was 72 degrees outside.",
    "The neighbor's dog barked seven times that afternoon.",
    "There were four red balloons tied to the mailbox.",
    "She remembered that her favorite color is green.",
    "The radio played thirteen songs during breakfast.",
    "Two squirrels ran across the lawn at noon.",
    "He had read nine pages of his book the night before.",
    "The clock on the wall said it was a Tuesday.",
]


NUMERIC_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def parse_gold_numeric(answer: str) -> str:
    match = re.search(r"####\s*([^\n]+)", answer)
    if match:
        numeric = NUMERIC_RE.search(match.group(1))
        if numeric:
            return numeric.group(0).replace(",", "")
    fallback = NUMERIC_RE.findall(answer)
    return fallback[-1].replace(",", "") if fallback else ""


def insert_distractor(question: str, distractor: str) -> str:
    """Insert the distractor as a sentence before the final question sentence.

    GSM8K questions usually end with a "How many ... ?" style sentence. We split
    on the last sentence terminator and insert the distractor between the setup
    and the question itself, so the question phrasing is preserved.
    """
    stripped = question.rstrip()
    boundary = max(stripped.rfind("?"), stripped.rfind("."), stripped.rfind("!"))
    last_terminator_before_q = -1
    if "?" in stripped:
        q_pos = stripped.rfind("?")
        last_terminator_before_q = max(
            stripped.rfind(".", 0, q_pos),
            stripped.rfind("!", 0, q_pos),
            stripped.rfind("?", 0, q_pos),
        )
    if last_terminator_before_q > 0:
        head = stripped[: last_terminator_before_q + 1]
        tail = stripped[last_terminator_before_q + 1 :].lstrip()
        return f"{head} {distractor} {tail}"
    return f"{stripped} {distractor}"


def build_perturbations(
    num_samples: int,
    out_dir: Path,
    seed: int,
) -> Tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    dataset = load_dataset("gsm8k", "main", split=f"test[:{num_samples}]")

    variant_rows: List[Dict[str, str]] = []
    index_rows: List[Dict[str, str]] = []
    for sample_idx, example in enumerate(dataset):
        question = str(example["question"])
        answer = str(example["answer"])
        gold = parse_gold_numeric(answer)

        # Variant 0: original passthrough (so the runner produces matched baselines
        # under the same row indexing).
        variant_rows.append({"question": question, "answer": answer})
        index_rows.append(
            {
                "row_idx": str(len(variant_rows) - 1),
                "original_sample_idx": str(sample_idx),
                "variant_kind": "original",
                "variant_index": "0",
                "perturbation_op": "",
                "gold_numeric": gold,
            }
        )

        # Variant 1: distractor_irrelevant.
        distractor = rng.choice(DISTRACTOR_TEMPLATES)
        perturbed = insert_distractor(question, distractor)
        variant_rows.append({"question": perturbed, "answer": answer})
        index_rows.append(
            {
                "row_idx": str(len(variant_rows) - 1),
                "original_sample_idx": str(sample_idx),
                "variant_kind": "perturbation",
                "variant_index": "1",
                "perturbation_op": "distractor_irrelevant",
                "gold_numeric": gold,
            }
        )

    variants_path = out_dir / "variants.jsonl"
    with variants_path.open("w", encoding="utf-8") as f:
        for row in variant_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    index_path = out_dir / "index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row_idx",
                "original_sample_idx",
                "variant_kind",
                "variant_index",
                "perturbation_op",
                "gold_numeric",
            ],
        )
        writer.writeheader()
        writer.writerows(index_rows)

    return len(variant_rows), len(index_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    n_variants, n_index = build_perturbations(args.num_samples, args.out_dir, args.seed)
    print(
        f"Wrote {n_variants} variant row(s) and {n_index} index row(s) to {args.out_dir}."
    )


if __name__ == "__main__":
    main()
