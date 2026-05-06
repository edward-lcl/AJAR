#!/usr/bin/env python
"""M2a validation: per-token intervention selectivity.

Five checks against Qwen3-4B-Instruct-MLX-8bit:
  1. intervention_tokens='all' matches the no-gating call (sanity)
  2. intervention_tokens='prompt' fires on prompt encoding only
  3. intervention_tokens='generated' fires on generated tokens only
  4. intervention_tokens=0 fires on the first generated token only
  5. intervention_tokens=[0,1,2] fires on the first three tokens only

For each, we compare against:
- baseline (no intervention): unhooked output
- all-fire (intervention on every forward): maximally hooked output
The selective version should be DIFFERENT from both, and ideally:
- prompt-only and generated-only should differ (different effective hook)
- single-position should differ from all-fire (less aggressive)
"""

from __future__ import annotations

import time
import warnings

warnings.filterwarnings("ignore", message=r".*LibreSSL.*", module="urllib3.*")
warnings.filterwarnings("ignore", message=r".*head_dim.*", module=".*")

from mlx_lm import load
from mlxterp import InterpretableModel
from mlxterp import interventions as iv


MODEL_NAME = "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-8bit"
PROMPT = "Q: What is 7 * 8? A:"
MAX_TOKENS = 24
LAYER = "layers.5"


def main() -> None:
    print("# M2a validation — per-token intervention selectivity")
    print()
    print("[setup] load + wrap")
    base, tok = load(MODEL_NAME)
    model = InterpretableModel(base, tokenizer=tok)

    print()
    print("[reference] baseline (no intervention)")
    baseline = model.generate(PROMPT, max_tokens=MAX_TOKENS)
    print(f"  {baseline!r}")

    print()
    print("[reference] all-fire (intervention on every forward)")
    all_fire = model.generate(
        PROMPT, max_tokens=MAX_TOKENS,
        interventions={LAYER: iv.zero_out},
    )
    print(f"  {all_fire!r}")
    print(f"  differs from baseline: {all_fire != baseline}")

    cases = [
        ("'all'", "all"),
        ("'prompt'", "prompt"),
        ("'generated'", "generated"),
        ("0", 0),
        ("[0,1,2]", [0, 1, 2]),
        ("range(3,8)", range(3, 8)),
    ]

    print()
    print("[gated cases]")
    results = {}
    for label, tokens in cases:
        t0 = time.time()
        out = model.generate(
            PROMPT, max_tokens=MAX_TOKENS,
            interventions={LAYER: iv.zero_out},
            intervention_tokens=tokens,
        )
        dt = time.time() - t0
        results[label] = out
        print(
            f"  intervention_tokens={label:<14} dt={dt:.2f}s  "
            f"!=baseline={out != baseline}  !=allfire={out != all_fire}"
        )
        print(f"    out: {out!r}")

    print()
    print("# Contract checks")
    checks = [
        ("'all' identical to all-fire", results["'all'"] == all_fire),
        ("'prompt' differs from baseline", results["'prompt'"] != baseline),
        ("'prompt' differs from all-fire", results["'prompt'"] != all_fire),
        ("'generated' differs from baseline", results["'generated'"] != baseline),
        ("'generated' differs from 'prompt'", results["'generated'"] != results["'prompt'"]),
        ("0 differs from baseline", results["0"] != baseline),
        ("0 differs from all-fire", results["0"] != all_fire),
        ("[0,1,2] differs from baseline", results["[0,1,2]"] != baseline),
        ("[0,1,2] differs from 0", results["[0,1,2]"] != results["0"]),
        ("[0,1,2] differs from all-fire", results["[0,1,2]"] != all_fire),
    ]
    all_passed = True
    for desc, ok in checks:
        marker = "PASS" if ok else "FAIL"
        if not ok:
            all_passed = False
        print(f"  {marker}: {desc}")
    print()
    print(f"all checks passed: {'YES' if all_passed else 'NO'}")


if __name__ == "__main__":
    main()
