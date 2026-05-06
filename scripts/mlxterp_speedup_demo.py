#!/usr/bin/env python
"""Wall-clock + intervention demo for mlxterp.generate on AJAR's baseline task.

This is the AJAR-side empirical evidence backing PR #10
(`feature/generation-with-interventions`). It does two things on the same
five GSM8K questions used in the deep-table run:

  1. **Speedup**: Time `InterpretableModel.generate` (in-process MLX-8bit)
     against the oMLX HTTP-server `generation_seconds` recorded during the
     2026-05-04 deep-table baseline. Same model, same prompts, no
     intervention. The integration win for AJAR is removing the HTTP hop
     and the JSON serialisation around it.

  2. **Intervention-during-generation**: Run the same prompts a third time
     with a small steering vector added to layer 5 during every generated
     token's forward pass. Reports answer + first-100-char preview so we
     can eyeball whether the hook actually fires (not just "no error").

The reference baselines on disk used the oMLX server (chatcmpl-* IDs in
`raw_response_id`), so this is a realistic A/B for AJAR's actual pipeline,
not a synthetic prompt. We're measuring the same workload AJAR runs in
production.

Output: a markdown summary at `docs/mlxterp_speedup_demo.md`.
"""

from __future__ import annotations

import json
import re
import time
import warnings
from pathlib import Path
from statistics import mean

warnings.filterwarnings("ignore", message=r".*LibreSSL.*", module="urllib3.*")
warnings.filterwarnings("ignore", message=r".*head_dim.*", module=".*")

import mlx.core as mx
from mlxterp import InterpretableModel
from mlxterp import interventions as iv


MLX_MODEL = "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-8bit"
BASELINE_JSONL = Path(
    "outputs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/baseline/baselines.jsonl"
)
N_QUESTIONS = 5
MAX_TOKENS = 512
NUMERIC_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
OUT_PATH = Path("docs/mlxterp_speedup_demo.md")


def normalise(s: str) -> str:
    if not s:
        return ""
    return s.strip().replace(",", "").replace("$", "").rstrip(".")


def extract_predicted(text: str) -> str:
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        nm = NUMERIC_RE.search(m.group(1))
        if nm:
            return normalise(nm.group(0))
    matches = list(NUMERIC_RE.finditer(text))
    return normalise(matches[-1].group(0)) if matches else ""


def load_reference_baselines(path: Path, n: int) -> list[dict]:
    """Pull the first n explicit_cot baselines for the instruct model."""
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("model_key") != "instruct":
                continue
            if rec.get("prompt_name") != "explicit_cot":
                continue
            rows.append(rec)
            if len(rows) >= n:
                break
    return rows


def build_prompt(question: str) -> str:
    """Reconstruct the explicit_cot prompt body. The reference baselines
    used the chat template via oMLX; here we hand the raw user content
    to mlx_lm.generate which doesn't apply a chat template, matching
    `mlxterp_parity_vs_torch.py`'s convention so the two scripts are
    comparable."""
    return (
        "Think through this step by step showing your reasoning process "
        "then provide your final answer. Put the final numeric answer in "
        "\\boxed{}.\n\n"
        f"Question: {question}"
    )


def main() -> None:
    print("# mlxterp.generate speedup + intervention demo on AJAR baselines")
    print()

    if not BASELINE_JSONL.exists():
        print(f"  reference baselines not found at {BASELINE_JSONL}")
        print("  run the deep-table experiment first")
        return

    refs = load_reference_baselines(BASELINE_JSONL, N_QUESTIONS)
    if len(refs) < N_QUESTIONS:
        print(f"  only {len(refs)} matching reference rows; expected {N_QUESTIONS}")
        return

    print(f"[setup] loading {MLX_MODEL} via InterpretableModel")
    t0 = time.perf_counter()
    model = InterpretableModel(MLX_MODEL)
    load_secs = time.perf_counter() - t0
    print(f"[setup] loaded in {load_secs:.1f}s")
    print()

    rows: list[dict] = []
    for ref in refs:
        sample_idx = ref["sample_idx"]
        question = ref["question"]
        gold = ref.get("gold_numeric")
        omlx_secs = float(ref.get("generation_seconds") or 0.0)
        omlx_pred = ref.get("prediction_numeric")
        omlx_correct = bool(ref.get("correct"))

        prompt = build_prompt(question)

        # Pass 1: plain mlxterp.generate, no intervention.
        t0 = time.perf_counter()
        out = model.generate(prompt, max_tokens=MAX_TOKENS, verbose=False)
        mlxterp_secs = time.perf_counter() - t0
        cont = out[len(prompt):] if out.startswith(prompt) else out
        mlxterp_pred = extract_predicted(cont)
        mlxterp_correct = mlxterp_pred == gold and mlxterp_pred != ""

        # Pass 2: same generate call but with a small additive steering
        # vector at layer 5. We use a tiny scale so the model still solves
        # the problem most of the time; the demo is "the hook fires", not
        # "this particular vector breaks the answer".
        hidden = 2560  # Qwen3-4B hidden size
        steer = mx.random.normal((hidden,), key=mx.random.key(sample_idx)) * 0.05
        t0 = time.perf_counter()
        out_iv = model.generate(
            prompt,
            max_tokens=MAX_TOKENS,
            interventions={"layers.5": iv.add_vector(steer)},
            intervention_tokens="generated",
            verbose=False,
        )
        iv_secs = time.perf_counter() - t0
        cont_iv = out_iv[len(prompt):] if out_iv.startswith(prompt) else out_iv
        iv_pred = extract_predicted(cont_iv)
        iv_correct = iv_pred == gold and iv_pred != ""

        speedup = (omlx_secs / mlxterp_secs) if mlxterp_secs > 0 else float("nan")

        rows.append(
            {
                "sample_idx": sample_idx,
                "gold": gold,
                "omlx_secs": omlx_secs,
                "omlx_pred": omlx_pred,
                "omlx_correct": omlx_correct,
                "mlxterp_secs": mlxterp_secs,
                "mlxterp_pred": mlxterp_pred,
                "mlxterp_correct": mlxterp_correct,
                "iv_secs": iv_secs,
                "iv_pred": iv_pred,
                "iv_correct": iv_correct,
                "speedup_x": speedup,
                "mlxterp_text": cont[:120],
                "iv_text": cont_iv[:120],
            }
        )
        print(
            f"  sample {sample_idx}: gold={gold!r}\n"
            f"    oMLX:    {omlx_secs:6.2f}s  pred={omlx_pred!r}  correct={omlx_correct}\n"
            f"    mlxterp: {mlxterp_secs:6.2f}s  pred={mlxterp_pred!r}  correct={mlxterp_correct}  speedup={speedup:.2f}x\n"
            f"    +iv L5:  {iv_secs:6.2f}s  pred={iv_pred!r}  correct={iv_correct}"
        )

    print()
    print("# Summary")
    n = len(rows)
    omlx_mean = mean(r["omlx_secs"] for r in rows)
    mlxterp_mean = mean(r["mlxterp_secs"] for r in rows)
    iv_mean = mean(r["iv_secs"] for r in rows)
    speedup_mean = mean(r["speedup_x"] for r in rows)
    pred_match = sum(1 for r in rows if r["omlx_pred"] == r["mlxterp_pred"])
    correct_match = sum(1 for r in rows if r["omlx_correct"] == r["mlxterp_correct"])
    iv_solved = sum(1 for r in rows if r["iv_correct"])
    iv_changed = sum(
        1 for r in rows if r["iv_pred"] != r["mlxterp_pred"]
    )

    print(f"  n = {n}")
    print(f"  oMLX mean:    {omlx_mean:6.2f}s")
    print(f"  mlxterp mean: {mlxterp_mean:6.2f}s  ({speedup_mean:.2f}x faster on average)")
    print(f"  +iv mean:     {iv_mean:6.2f}s  (steering overhead vs plain mlxterp)")
    print(f"  prediction match (oMLX vs mlxterp):  {pred_match}/{n}")
    print(f"  correctness match (oMLX vs mlxterp): {correct_match}/{n}")
    print(f"  +iv still solved gold:  {iv_solved}/{n}")
    print(f"  +iv changed prediction: {iv_changed}/{n}  (evidence the hook fires)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        f.write("# mlxterp.generate speedup + intervention demo on AJAR baselines\n\n")
        f.write(
            "Same five GSM8K questions used in the 2026-05-04 deep-table run, "
            "same Qwen3-4B-Instruct-MLX-8bit weights. Reference timings are "
            "the recorded `generation_seconds` from oMLX's HTTP server during "
            "that run; mlxterp timings are from `InterpretableModel.generate` "
            "in-process.\n\n"
        )
        f.write(f"Model: `{MLX_MODEL}`\n\n")
        f.write(f"Max tokens: {MAX_TOKENS}\n\n")
        f.write(
            "| sample | gold | oMLX (s) | mlxterp (s) | speedup | +iv L5 (s) | "
            "oMLX pred | mlxterp pred | +iv pred |\n"
        )
        f.write(
            "|---|---|---|---|---|---|---|---|---|\n"
        )
        for r in rows:
            f.write(
                f"| {r['sample_idx']} | {r['gold']} | "
                f"{r['omlx_secs']:.2f} | {r['mlxterp_secs']:.2f} | "
                f"{r['speedup_x']:.2f}x | {r['iv_secs']:.2f} | "
                f"{r['omlx_pred']} | {r['mlxterp_pred']} | {r['iv_pred']} |\n"
            )
        f.write("\n")
        f.write("## Aggregate\n\n")
        f.write(f"- oMLX mean:    {omlx_mean:.2f}s\n")
        f.write(f"- mlxterp mean: {mlxterp_mean:.2f}s\n")
        f.write(f"- mean speedup: **{speedup_mean:.2f}x**\n")
        f.write(f"- +iv mean:     {iv_mean:.2f}s\n")
        f.write(f"- prediction agreement (oMLX vs mlxterp):  {pred_match}/{n}\n")
        f.write(f"- correctness agreement (oMLX vs mlxterp): {correct_match}/{n}\n")
        f.write(f"- intervention changed answer on:          {iv_changed}/{n}\n")
        f.write("\n")
        f.write("## Caveats\n\n")
        f.write(
            "- Reference oMLX timings include HTTP overhead and JSON "
            "serialisation, so the speedup is for the AJAR pipeline as a "
            "whole, not a pure inference comparison. That is the right "
            "comparison: AJAR was paying that overhead.\n"
        )
        f.write(
            "- The intervention pass uses a tiny additive vector "
            "(N(0, 0.05) at layer 5, generated tokens only). It is meant "
            "to demonstrate the hook fires, not to break the model.\n"
        )
        f.write(
            "- The reference oMLX prompts went through the chat template; "
            "the mlxterp call does not. Predicted-numeric agreement is "
            "the apples-to-apples signal.\n"
        )

    print()
    print(f"  wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
