#!/usr/bin/env python
"""M1 validation: confirm InterpretableModel.generate produces identical
output to the manual trace+mlx_generate pattern, and that interventions
fire correctly during generation.

Three checks:
  1. generate() with no interventions == raw mlx_lm.generate (sanity)
  2. generate() with interventions != baseline (interventions firing)
  3. generate() with interventions == manual trace+mlx_lm.generate
     pattern from M0 (the new API matches the validated hack)
"""

from __future__ import annotations

import time
import warnings

warnings.filterwarnings("ignore", message=r".*LibreSSL.*", module="urllib3.*")
warnings.filterwarnings("ignore", message=r".*head_dim.*", module=".*")

import mlx.core as mx
from mlx_lm import load
from mlx_lm import generate as mlx_generate
from mlxterp import InterpretableModel
from mlxterp import interventions as iv


MODEL_NAME = "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-8bit"
PROMPT = "Q: What is 7 * 8? A:"
MAX_TOKENS = 24


def main() -> None:
    print("# M1 validation — InterpretableModel.generate")
    print()
    print("[setup] load + wrap")
    base, tok = load(MODEL_NAME)
    model = InterpretableModel(base, tokenizer=tok)
    print()

    # Check 1: generate() with no interventions == raw mlx-lm
    print("[check 1] generate(no interventions) vs raw mlx_lm.generate")
    t0 = time.time()
    raw_out = mlx_generate(base, tok, PROMPT, max_tokens=MAX_TOKENS, verbose=False)
    t_raw = time.time() - t0
    t0 = time.time()
    api_out = model.generate(PROMPT, max_tokens=MAX_TOKENS)
    t_api = time.time() - t0
    print(f"  raw:  {raw_out!r}  ({t_raw:.2f}s)")
    print(f"  api:  {api_out!r}  ({t_api:.2f}s)")
    print(f"  match: {'YES' if raw_out == api_out else 'NO'}")
    print()

    # Check 2: generate() with interventions != baseline
    print("[check 2] generate(zero_out layer 5) != baseline")
    t0 = time.time()
    ablated_out = model.generate(
        PROMPT,
        max_tokens=MAX_TOKENS,
        interventions={"layers.5": iv.zero_out},
    )
    t_ablated = time.time() - t0
    print(f"  baseline: {api_out!r}")
    print(f"  ablated:  {ablated_out!r}  ({t_ablated:.2f}s)")
    print(f"  differs from baseline: {'YES' if ablated_out != api_out else 'NO'}")
    print()

    # Check 3: generate() vs manual hack pattern from M0 — should be identical
    print("[check 3] generate(interventions=...) vs manual trace+mlx_generate hack")
    print("          (the new API should match the M0-validated pattern)")
    t0 = time.time()
    with model.trace(PROMPT, interventions={"layers.5": iv.zero_out}):
        manual_out = mlx_generate(base, tok, PROMPT, max_tokens=MAX_TOKENS, verbose=False)
    t_manual = time.time() - t0
    print(f"  manual hack:  {manual_out!r}  ({t_manual:.2f}s)")
    print(f"  new API:      {ablated_out!r}")
    print(f"  match:        {'YES' if manual_out == ablated_out else 'NO'}")
    print()

    # Check 4: cleanup — interventions don't leak to subsequent calls
    print("[check 4] post-intervention generate is back to baseline")
    t0 = time.time()
    post_out = model.generate(PROMPT, max_tokens=MAX_TOKENS)
    t_post = time.time() - t0
    print(f"  post: {post_out!r}  ({t_post:.2f}s)")
    print(f"  matches baseline: {'YES' if post_out == api_out else 'NO (LEAK!)'}")
    print()

    print("# Summary")
    all_passed = (
        raw_out == api_out
        and ablated_out != api_out
        and manual_out == ablated_out
        and post_out == api_out
    )
    print(f"  all checks passed: {'YES' if all_passed else 'NO'}")
    if all_passed:
        print()
        print("  M1 implementation is correct:")
        print("  - clean API matches raw mlx-lm when no interventions")
        print("  - interventions fire during generation (output diverges)")
        print("  - new API produces identical output to the M0-validated hack")
        print("  - patches are properly cleaned up after the generate call")


if __name__ == "__main__":
    main()
