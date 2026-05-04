#!/usr/bin/env python
"""Build paraphrased GSM8K variants for Task 6 paraphrase consistency.

Calls the local oMLX server with Qwen3-4B-Instruct (deterministic) to produce
K paraphrases per question. The paraphrase prompt is constrained to preserve
all numeric values and the underlying logical structure, so the gold answer
from the canonical GSM8K row remains valid.

Output:
- <out_dir>/variants.jsonl  GSM8K-shaped rows runnable via GSM8K_JSONL=<path>.
- <out_dir>/index.csv       Maps each row back to original sample, variant kind,
                            and variant index.

Usage:
    python3 scripts/build_paraphrases.py --num-samples 50 \\
        --num-paraphrases 2 --out-dir data/variants/gsm8k_test_first50_para
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset


NUMERIC_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


PARAPHRASE_SYSTEM_PROMPT = (
    "You are a precise paraphrasing assistant. Given a math word problem, "
    "rewrite it in different wording while satisfying ALL of the following "
    "constraints:\n"
    "1. Preserve every number that appears in the original problem, in the same "
    "units. Do not add or remove numbers, and do not round or change quantities.\n"
    "2. Preserve the logical and arithmetic structure. The correct numerical "
    "answer must remain identical to the original problem.\n"
    "3. Preserve the named entities (people, places, items) when possible. If you "
    "rename them, keep the renaming internally consistent within the problem.\n"
    "4. Preserve the final question phrasing intent (e.g., 'how many', 'how much').\n"
    "5. Do not include the answer, hints, or any text outside the rewritten "
    "problem statement. Output only the paraphrased problem text."
)


def parse_gold_numeric(answer: str) -> str:
    match = re.search(r"####\s*([^\n]+)", answer)
    if match:
        numeric = NUMERIC_RE.search(match.group(1))
        if numeric:
            return numeric.group(0).replace(",", "")
    fallback = NUMERIC_RE.findall(answer)
    return fallback[-1].replace(",", "") if fallback else ""


def numbers_in(text: str) -> List[str]:
    return [m.replace(",", "") for m in NUMERIC_RE.findall(text)]


def resolve_omlx_api_key() -> str:
    settings_path = Path.home() / ".omlx" / "settings.json"
    try:
        return str(json.loads(settings_path.read_text(encoding="utf-8"))["auth"]["api_key"])
    except Exception:
        return ""


def call_omlx(
    base_url: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    seed: int,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data["choices"][0]["message"].get("content") or "").strip()


def paraphrase_question(
    question: str,
    base_url: str,
    api_key: str,
    model: str,
    seed: int,
    style_hint: str,
) -> str:
    user_prompt = (
        f"{style_hint}\n\n"
        f"Original problem:\n{question}\n\n"
        f"Rewritten problem (numbers preserved exactly):"
    )
    return call_omlx(
        base_url=base_url,
        api_key=api_key,
        model=model,
        system=PARAPHRASE_SYSTEM_PROMPT,
        user=user_prompt,
        max_tokens=512,
        seed=seed,
    )


STYLE_HINTS = [
    "Rewrite this problem in a conversational, narrative style. Vary the "
    "sentence structure but keep all numbers exactly as they appear.",
    "Rewrite this problem more concisely. Combine sentences where natural, "
    "but keep all numbers exactly as they appear and keep the question itself "
    "explicit.",
]


def build_paraphrases(
    num_samples: int,
    num_paraphrases: int,
    out_dir: Path,
    base_url: str,
    api_key: str,
    model: str,
    seed: int,
) -> Tuple[int, int, int]:
    if num_paraphrases > len(STYLE_HINTS):
        raise ValueError(
            f"Only {len(STYLE_HINTS)} style hints are defined; either lower "
            f"--num-paraphrases or extend STYLE_HINTS."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("gsm8k", "main", split=f"test[:{num_samples}]")

    variant_rows: List[Dict[str, str]] = []
    index_rows: List[Dict[str, str]] = []
    rejected = 0

    for sample_idx, example in enumerate(dataset):
        question = str(example["question"])
        answer = str(example["answer"])
        gold = parse_gold_numeric(answer)
        original_numbers = sorted(numbers_in(question))

        # Variant 0: original passthrough.
        variant_rows.append({"question": question, "answer": answer})
        index_rows.append(
            {
                "row_idx": str(len(variant_rows) - 1),
                "original_sample_idx": str(sample_idx),
                "variant_kind": "original",
                "variant_index": "0",
                "paraphrase_style": "",
                "gold_numeric": gold,
                "numbers_preserved": "true",
            }
        )

        for variant_index in range(1, num_paraphrases + 1):
            style_hint = STYLE_HINTS[variant_index - 1]
            print(
                f"[{sample_idx + 1}/{num_samples}] paraphrase {variant_index}",
                flush=True,
            )
            try:
                paraphrased = paraphrase_question(
                    question=question,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    seed=seed + sample_idx * 100 + variant_index,
                    style_hint=style_hint,
                )
            except urllib.error.URLError as exc:
                raise RuntimeError(
                    f"Could not reach oMLX at {base_url}. Start the server or "
                    f"set OMLX_BASE_URL/OMLX_API_KEY."
                ) from exc

            preserved = sorted(numbers_in(paraphrased)) == original_numbers
            if not preserved:
                rejected += 1

            variant_rows.append({"question": paraphrased, "answer": answer})
            index_rows.append(
                {
                    "row_idx": str(len(variant_rows) - 1),
                    "original_sample_idx": str(sample_idx),
                    "variant_kind": "paraphrase",
                    "variant_index": str(variant_index),
                    "paraphrase_style": style_hint,
                    "gold_numeric": gold,
                    "numbers_preserved": "true" if preserved else "false",
                }
            )
            time.sleep(0.05)

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
                "paraphrase_style",
                "gold_numeric",
                "numbers_preserved",
            ],
        )
        writer.writeheader()
        writer.writerows(index_rows)

    return len(variant_rows), len(index_rows), rejected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--num-paraphrases", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Local oMLX base URL.",
    )
    parser.add_argument(
        "--model",
        default="Qwen3-4B-Instruct-2507-MLX-8bit",
        help="Model id served by the local oMLX runtime.",
    )
    args = parser.parse_args()
    api_key = resolve_omlx_api_key()
    if not api_key:
        raise RuntimeError(
            "Could not load oMLX API key from ~/.omlx/settings.json. Set "
            "OMLX_API_KEY explicitly if your config is elsewhere."
        )
    n_variants, n_index, rejected = build_paraphrases(
        num_samples=args.num_samples,
        num_paraphrases=args.num_paraphrases,
        out_dir=args.out_dir,
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        seed=args.seed,
    )
    print(
        f"Wrote {n_variants} variant row(s) and {n_index} index row(s) to {args.out_dir}; "
        f"{rejected} paraphrase(s) flagged numbers_preserved=false."
    )


if __name__ == "__main__":
    main()
