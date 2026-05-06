#!/usr/bin/env python
"""Validate the new activation_patching positions / metric extensions.

Three checks against Qwen3-4B-Instruct-MLX-8bit:
  1. Layer-only call (positions=None) still works (backwards compat).
  2. Position-level call with positions='all' returns an (n_layers, n_pos)
     matrix and the values are sensible (non-zero recovery somewhere).
  3. logit_diff metric works end-to-end on a Paris/London-style task.
"""

from __future__ import annotations

import time
import warnings

warnings.filterwarnings("ignore", message=r".*LibreSSL.*", module="urllib3.*")
warnings.filterwarnings("ignore", message=r".*head_dim.*", module=".*")

import numpy as np
from mlx_lm import load
from mlxterp import InterpretableModel


MODEL_NAME = "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-8bit"


def main() -> None:
    print("# activation_patching extensions validation")
    print()
    print("[setup] loading model...")
    base, tok = load(MODEL_NAME)
    model = InterpretableModel(base, tokenizer=tok)
    print()

    # Same length is REQUIRED for positions != None. Pad/select prompts so
    # the tokenised lengths line up. The Paris/London pair tokenises to
    # 6 tokens each on Qwen3.
    clean = "Paris is the capital of France"
    corrupted = "London is the capital of France"
    print(f"clean   tokens: {tok.encode(clean)}")
    print(f"corrupt tokens: {tok.encode(corrupted)}")
    print()

    # 1. Layer-only (backwards compat)
    print("[check 1] layer-only patching (positions=None) returns Dict")
    t0 = time.time()
    layer_results = model.activation_patching(
        clean_text=clean,
        corrupted_text=corrupted,
        component="output",
        layers=list(range(0, 36, 6)),  # every 6th layer for speed
        metric="l2",
    )
    print(f"  done in {time.time()-t0:.1f}s")
    print(f"  type: {type(layer_results).__name__}")
    print(f"  result: {layer_results}")
    print()

    # 2. Position-level with positions='all'
    print("[check 2] position-level patching (positions='all') returns ndarray")
    t0 = time.time()
    matrix = model.activation_patching(
        clean_text=clean,
        corrupted_text=corrupted,
        component="output",
        layers=list(range(0, 36, 6)),
        positions="all",
        metric="l2",
    )
    print(f"  done in {time.time()-t0:.1f}s")
    print(f"  type: {type(matrix).__name__}, shape: {matrix.shape}, dtype: {matrix.dtype}")
    print(f"  matrix max recovery: {matrix.max():.2f}%")
    print(f"  matrix min recovery: {matrix.min():.2f}%")
    print(f"  matrix mean: {matrix.mean():.2f}%")
    print()

    # 3. logit_diff metric
    print("[check 3] logit_diff metric on Paris/London (Eiffel Tower task variant)")
    eiffel_clean    = "When in Rome, do as the Romans do. When in Paris, do as"
    eiffel_corrupt  = "When in Paris, do as the Parisians do. When in Rome, do as"
    print(f"  clean tokens len: {len(tok.encode(eiffel_clean))}, corrupt: {len(tok.encode(eiffel_corrupt))}")
    if len(tok.encode(eiffel_clean)) != len(tok.encode(eiffel_corrupt)):
        # Fall back to the Paris/London pair which is known same-length
        eiffel_clean = clean
        eiffel_corrupt = corrupted
    t0 = time.time()
    matrix2 = model.activation_patching(
        clean_text=eiffel_clean,
        corrupted_text=eiffel_corrupt,
        component="output",
        layers=list(range(0, 36, 6)),
        positions=[0, 3, 5],  # specific positions
        metric="logit_diff",
        clean_target=" Paris",
        corrupted_target=" London",
    )
    print(f"  done in {time.time()-t0:.1f}s")
    print(f"  shape: {matrix2.shape}")
    print(f"  values:")
    layer_list = list(range(0, 36, 6))
    for li, layer_idx in enumerate(layer_list[:matrix2.shape[0]]):
        row = matrix2[li]
        print(f"    layer {layer_idx}: {[f'{v:+.1f}' for v in row]}")
    print()

    # Contract checks
    checks = [
        ("layer-only returns dict", isinstance(layer_results, dict)),
        ("matrix path returns ndarray", isinstance(matrix, np.ndarray)),
        ("matrix has expected 2D shape", matrix.ndim == 2),
        ("matrix has reasonable layer count", matrix.shape[0] == 6),
        ("matrix has all positions", matrix.shape[1] == len(tok.encode(clean))),
        ("logit_diff returns matrix", isinstance(matrix2, np.ndarray)),
        ("logit_diff matrix shape matches positions arg", matrix2.shape == (6, 3)),
        ("at least one cell has non-trivial recovery", float(np.abs(matrix).max()) > 1.0),
    ]
    print("# Contract checks")
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
