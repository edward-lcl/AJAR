#!/usr/bin/env python
"""M0 follow-up: probe whether intervention-during-generation can be
hacked together with the existing mlxterp + mlx-lm API.

Strategy: enter a model.trace() context with interventions applied
(which patches the layers), then call mlx_lm.generate() while the
patches are still active inside the with-block. If the patches
survive across multiple forward passes, we get intervention-during-
generation "for free" — minus an API.

If it works, that tells us the M1 milestone is mostly an API ergonomics
job (expose `model.generate(text, interventions=...)` cleanly) rather
than a deep architectural change. If it doesn't, M1 is the real work.
"""

from __future__ import annotations

import time
import warnings

warnings.filterwarnings("ignore", message=r".*LibreSSL.*", module="urllib3.*")

import mlx.core as mx
from mlx_lm import load
from mlx_lm import generate as mlx_generate
from mlxterp import InterpretableModel
from mlxterp import interventions as iv


MODEL_NAME = "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-8bit"
PROMPT = "Q: What is 7 * 8? A:"
MAX_TOKENS = 24


def main() -> None:
    print("# mlxterp generation probe — can we get intervention-during-generation?")
    print(f"model: {MODEL_NAME}")
    print(f"prompt: {PROMPT!r}, max_tokens: {MAX_TOKENS}")
    print()

    print("[step 1] load + wrap", flush=True)
    base_model, tokenizer = load(MODEL_NAME)
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    print()
    print("[step 2] generate without any intervention (baseline) ...", flush=True)
    t0 = time.time()
    baseline_out = mlx_generate(
        base_model, tokenizer, PROMPT, max_tokens=MAX_TOKENS, verbose=False
    )
    t_baseline = time.time() - t0
    print(f"  time: {t_baseline:.2f}s")
    print(f"  output: {baseline_out!r}")

    print()
    print("[step 3] generate with intervention applied OUTSIDE a trace block")
    print("         (interventions={} on a no-op trace, then call generate)")
    print("         — testing whether patches leak past the trace context.")
    t0 = time.time()
    with model.trace(PROMPT, interventions={"layers.5": iv.zero_out}):
        pass
    # Patches should now be restored. Confirm a fresh generate is unchanged.
    posttrace_out = mlx_generate(
        base_model, tokenizer, PROMPT, max_tokens=MAX_TOKENS, verbose=False
    )
    t_posttrace = time.time() - t0
    print(f"  time: {t_posttrace:.2f}s")
    print(f"  output: {posttrace_out!r}")
    print(
        f"  matches baseline? "
        f"{'YES' if posttrace_out == baseline_out else 'NO (patches leaked!)'}"
    )

    print()
    print("[step 4] generate INSIDE a trace block where patches are still active")
    print("         — the interesting case. Trace's __enter__ runs ONE forward")
    print("         on PROMPT, but patches stay applied until __exit__. So if we")
    print("         call mlx_generate inside the with-block, each generation")
    print("         step's forward should hit the patched layers.")
    t0 = time.time()
    try:
        with model.trace(PROMPT, interventions={"layers.5": iv.zero_out}):
            inside_out = mlx_generate(
                base_model, tokenizer, PROMPT, max_tokens=MAX_TOKENS, verbose=False
            )
        t_inside = time.time() - t0
        print(f"  time: {t_inside:.2f}s")
        print(f"  output: {inside_out!r}")
        print(
            f"  differs from baseline? "
            f"{'YES (intervention is firing during generation!)' if inside_out != baseline_out else 'NO (patches not active during generate)'}"
        )
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
