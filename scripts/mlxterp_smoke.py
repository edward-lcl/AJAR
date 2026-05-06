#!/usr/bin/env python
"""M0 smoke for mlxterp contribution work.

Verifies that mlxterp can:
  1. Load Qwen3-4B-Instruct-2507-MLX-8bit via mlx-lm + InterpretableModel
  2. Run a single forward pass and capture activations
  3. Apply a single-forward intervention and produce different output
     vs no intervention

This is the baseline before we add generation-with-interventions support.
Times every step so we can compare against torch+MPS later.

Usage:
    python3 scripts/mlxterp_smoke.py
"""

from __future__ import annotations

import time
import warnings

# Mute the urllib3 LibreSSL warning that fires on system Python.
warnings.filterwarnings("ignore", message=r".*LibreSSL.*", module="urllib3.*")

import mlx.core as mx
from mlx_lm import load
from mlxterp import InterpretableModel
from mlxterp import interventions as iv


MODEL_NAME = "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-8bit"
PROMPT = "Q: What is 7 * 8? A:"


def main() -> None:
    print(f"# mlxterp M0 smoke")
    print(f"model: {MODEL_NAME}")
    print(f"prompt: {PROMPT!r}")
    print()

    print("[step 1] Loading model via mlx_lm.load ...", flush=True)
    t0 = time.time()
    base_model, tokenizer = load(MODEL_NAME)
    t_load = time.time() - t0
    print(f"  mlx-lm load: {t_load:.2f}s")

    print("[step 2] Wrapping with InterpretableModel ...", flush=True)
    t0 = time.time()
    model = InterpretableModel(base_model, tokenizer=tokenizer)
    t_wrap = time.time() - t0
    print(f"  wrap: {t_wrap:.2f}s")
    print(f"  num decoder layers: {len(model.layers)}")

    print()
    print("[step 3] Trace forward pass and capture activations ...", flush=True)
    t0 = time.time()
    with model.trace(PROMPT) as trace:
        pass
    t_trace = time.time() - t0
    print(f"  trace: {t_trace:.2f}s")
    print(f"  activations captured: {len(trace.activations)}")
    print(f"  sample activation names:")
    for name in list(trace.activations.keys())[:8]:
        shape = getattr(trace.activations[name], "shape", "?")
        print(f"    {name}  shape={shape}")
    print(f"    ... ({len(trace.activations)} total)")

    print()
    print("[step 4] Capture residual at layer 5 (clean) ...", flush=True)
    t0 = time.time()
    with model.trace(PROMPT) as clean:
        pass
    layer_key = "model.model.layers.5"
    clean_resid = clean.activations[layer_key]
    mx.eval(clean_resid)
    t_capture = time.time() - t0
    print(f"  capture: {t_capture:.2f}s")
    print(f"  residual shape at {layer_key}: {clean_resid.shape}")

    print()
    print("[step 5] Trace with intervention (zero out layer 5) ...", flush=True)
    t0 = time.time()
    with model.trace(PROMPT, interventions={"layers.5": iv.zero_out}) as ablated:
        pass
    ablated_resid = ablated.activations[layer_key]
    mx.eval(ablated_resid)
    t_ablate = time.time() - t0
    print(f"  trace + ablate: {t_ablate:.2f}s")
    diff_norm = float(mx.linalg.norm(ablated_resid - clean_resid))
    clean_norm = float(mx.linalg.norm(clean_resid))
    print(
        f"  ||ablated - clean||  = {diff_norm:.3f}, "
        f"||clean|| = {clean_norm:.3f}, ratio = {diff_norm/clean_norm:.3f}"
    )
    print(
        "  expected: ratio close to 1 since layer 5 was zeroed and downstream "
        "layers are propagating the corrupted residual."
    )

    print()
    print("# Summary")
    print(f"  load+wrap: {t_load + t_wrap:.2f}s")
    print(f"  one trace: {t_trace:.2f}s")
    print(f"  one trace + intervention: {t_ablate:.2f}s")
    print(f"  activations captured per trace: {len(trace.activations)}")
    print(
        "  Single-forward path works. "
        "Next: test what happens if we try to generate (M1)."
    )


if __name__ == "__main__":
    main()
