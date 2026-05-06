#!/usr/bin/env python
"""Numerical parity test: mlxterp+8bit vs torch+fp16 on AJAR's intervention task.

The contribution plan called for a 5-question parity test before we'd
trust mlxterp for AJAR research. This script does that:

  1. For each of the first 5 GSM8K questions, load the existing
     torch+fp16 baseline output (already on disk from the deep-table
     run, mech/instruct/explicit_cot/sample_NNNN/baseline.json).
  2. Run mlxterp-backed generation on the same prompt with the same
     deterministic decoding settings.
  3. Compare:
       a. final predicted number (do both backends agree?)
       b. correctness vs gold (do both backends solve the question?)

  Pass criterion (from mlxterp_contribution_plan.md):
       - same predicted number on >= 3/5 questions, OR
       - same correctness pattern on >= 4/5 questions

Even with 8-bit quantization differences, both backends should reach
the gold answer on a majority of GSM8K problems. That's the bar.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=r".*LibreSSL.*", module="urllib3.*")
warnings.filterwarnings("ignore", message=r".*head_dim.*", module=".*")

from mlx_lm import load, generate as mlx_generate


MLX_MODEL = "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-8bit"
TORCH_REF_DIR = Path(
    "outputs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/mech/instruct/explicit_cot"
)
NUMERIC_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def normalise(s: str) -> str:
    if not s:
        return ""
    return s.strip().replace(",", "").replace("$", "").rstrip(".")


def extract_predicted(text: str) -> str:
    """Same logic as AJAR's runner: prefer \\boxed{}, else last number."""
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        nm = NUMERIC_RE.search(m.group(1))
        if nm:
            return normalise(nm.group(0))
    matches = list(NUMERIC_RE.finditer(text))
    return normalise(matches[-1].group(0)) if matches else ""


def main() -> None:
    print("# mlxterp-vs-torch parity test on AJAR's intervention task")
    print()
    print("[setup] loading MLX-8bit Qwen3-4B-Instruct...")
    base, tok = load(MLX_MODEL)
    print()

    if not TORCH_REF_DIR.exists():
        print(f"  torch reference not found at {TORCH_REF_DIR}")
        print("  this script depends on the AJAR deep-table run output")
        return

    sample_dirs = sorted(TORCH_REF_DIR.glob("sample_000?"))[:5]
    print(f"[parity] comparing mlxterp vs torch on {len(sample_dirs)} questions")
    print()

    rows = []
    for sd in sample_dirs:
        baseline_path = sd / "baseline.json"
        if not baseline_path.exists():
            continue
        torch_baseline = json.loads(baseline_path.read_text())
        sample_idx = torch_baseline["sample_idx"]
        question = torch_baseline["question"]
        gold = torch_baseline.get("gold_numeric")
        torch_pred = torch_baseline.get("prediction_numeric")
        torch_correct = bool(torch_baseline.get("correct"))

        # The torch baseline used the explicit_cot prompt. Reconstruct it.
        prompt = (
            "Think through this step by step showing your reasoning process "
            "then provide your final answer. Put the final numeric answer in \\boxed{}.\n\n"
            f"Question: {question}"
        )
        # mlx-lm generation, deterministic
        mlx_text = mlx_generate(base, tok, prompt, max_tokens=512, verbose=False)
        # Strip the prompt prefix that mlx_lm.generate sometimes echoes
        mlx_continuation = mlx_text[len(prompt):] if mlx_text.startswith(prompt) else mlx_text
        mlx_pred = extract_predicted(mlx_continuation)
        mlx_correct = mlx_pred == gold and mlx_pred != ""

        rows.append(
            {
                "sample_idx": sample_idx,
                "gold": gold,
                "torch_pred": torch_pred,
                "mlx_pred": mlx_pred,
                "torch_correct": torch_correct,
                "mlx_correct": mlx_correct,
                "preds_match": torch_pred == mlx_pred,
                "correctness_match": torch_correct == mlx_correct,
                "torch_text": (torch_baseline.get("generated_text") or "")[:120],
                "mlx_text": mlx_continuation[:120],
            }
        )
        print(
            f"  sample {sample_idx}: gold={gold!r}  torch={torch_pred!r}  "
            f"mlx={mlx_pred!r}  torch_correct={torch_correct}  "
            f"mlx_correct={mlx_correct}"
        )
        print(f"    mlx text: {mlx_continuation[:160]!r}")

    print()
    n = len(rows)
    if n == 0:
        print("no rows; aborting")
        return

    preds_match = sum(1 for r in rows if r["preds_match"])
    correctness_match = sum(1 for r in rows if r["correctness_match"])
    both_correct = sum(1 for r in rows if r["torch_correct"] and r["mlx_correct"])
    only_torch = sum(1 for r in rows if r["torch_correct"] and not r["mlx_correct"])
    only_mlx = sum(1 for r in rows if not r["torch_correct"] and r["mlx_correct"])
    neither = sum(1 for r in rows if not r["torch_correct"] and not r["mlx_correct"])

    print(f"# Summary on n={n} questions")
    print(f"  exact prediction match:    {preds_match}/{n}")
    print(f"  correctness match:         {correctness_match}/{n}")
    print(f"  both correct:              {both_correct}")
    print(f"  only torch correct:        {only_torch}")
    print(f"  only mlxterp correct:      {only_mlx}")
    print(f"  neither correct:           {neither}")
    print()

    pred_pass = preds_match >= 3
    correct_pass = correctness_match >= 4
    print(f"  contract: same prediction on >= 3/5  → "
          f"{'PASS' if pred_pass else 'FAIL'} ({preds_match}/5)")
    print(f"  contract: same correctness on >= 4/5 → "
          f"{'PASS' if correct_pass else 'FAIL'} ({correctness_match}/5)")
    print()
    print(
        f"verdict: mlxterp {'CAN' if (pred_pass or correct_pass) else 'CANNOT'} "
        f"replace torch+fp16 for AJAR's intervention task at the contract level"
    )


if __name__ == "__main__":
    main()
