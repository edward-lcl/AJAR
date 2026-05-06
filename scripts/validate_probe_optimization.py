#!/usr/bin/env python
"""Validate that the optimised single-forward probe path produces numerically
equivalent attention values to the original prefix-forward path.

The two paths *should* be bit-identical because causal masking guarantees
that attention from position p depends only on tokens at positions <= p,
regardless of what comes after. But MPS occasionally produces tiny
numerical drift between paths that are mathematically identical, and
that drift could shift anchor selection on borderline cases.

This script runs both paths on a single small sample and reports the
maximum element-wise difference plus whether anchor selection agrees.

Usage:
    python3 scripts/validate_probe_optimization.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Force the runner module to use a consistent backend setup.
os.environ.setdefault("AJAR_BACKEND", "torch")
os.environ.setdefault("AJAR_MODELS", "instruct")
os.environ.setdefault("AJAR_PROMPTS", "neutral_strict")
os.environ.setdefault("AJAR_NUM_SAMPLES", "1")
os.environ.setdefault("AJAR_RUN_MI", "1")
os.environ.setdefault("AJAR_DTYPE", "float16")
os.environ.setdefault("AJAR_MAX_NEW_TOKENS", "128")
os.environ.setdefault("AJAR_MAX_ANSWER_PROBES", "999999")  # disable cap for clean comparison
os.environ.setdefault("AJAR_TOP_ANCHOR_STEPS", "3")
os.environ.setdefault("AJAR_NUM_CONTROL_STEPS", "0")
os.environ.setdefault("AJAR_INTERVENTIONS", "residual_zero")
os.environ.setdefault("AJAR_OUTPUT_DIR", "outputs/probe_validation")


def main():
    spec = importlib.util.spec_from_file_location(
        "run_qwen3_gsm8k_mi", REPO_ROOT / "scripts" / "run_qwen3_gsm8k_mi.py"
    )
    runner = importlib.util.module_from_spec(spec)
    sys.modules["run_qwen3_gsm8k_mi"] = runner
    spec.loader.exec_module(runner)
    runner.require_backend_dependencies("torch")

    from datasets import load_dataset
    import numpy as np
    import torch

    dataset = load_dataset("gsm8k", "main", split="test[:1]")
    example = dataset[0]
    model_name = "Qwen/Qwen3-4B-Instruct-2507"

    print(f"Loading {model_name}...")
    tokenizer, model = runner.load_model_and_tokenizer(
        model_name=model_name,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=False,
        device="mps",
    )

    messages = runner.PROMPTS["neutral_strict"](example["question"])
    prompt_text = runner.build_prompt_text(tokenizer, messages)

    args = runner.get_config()
    print("Running baseline generation (cap 128 tokens)...")
    generation = runner.generate_completion(model, tokenizer, prompt_text, args)
    full_ids = generation["full_ids"]

    span_info = runner.identify_reasoning_and_answer_spans(generation["generated_text"])
    steps, tokenization = runner.build_step_spans(
        tokenizer,
        prompt_text,
        generation["generated_text"],
        span_info["reasoning_char_span_generated"],
    )
    answer_token_positions = runner.get_answer_token_positions(
        tokenization["offset_mapping"],
        tokenization["special_tokens_mask"],
        prompt_text,
        span_info["answer_char_span_generated"],
    )
    probes = runner.build_probe_plan(
        steps, answer_token_positions, max_answer_probes=999999
    )
    print(f"Probe count: {len(probes)} ({len(steps)} steps + answer span)")
    if not probes:
        print("No probes to compare. Exiting.")
        return

    sample_dir = REPO_ROOT / "outputs" / "probe_validation"
    sample_dir.mkdir(parents=True, exist_ok=True)

    print("\nPath A: original per-probe forward...")
    legacy = runner.probe_attention_vectors(
        model,
        full_ids,
        probes,
        save_full_probe_attention=False,
        sample_dir=sample_dir,
        raw_attentions=None,
    )

    print("Path B: single-forward + indexing...")
    device = runner.get_model_input_device(model)
    with torch.inference_mode():
        outputs = model(
            input_ids=full_ids.unsqueeze(0).to(device),
            attention_mask=torch.ones((1, full_ids.shape[0]), device=device),
            output_hidden_states=False,
            output_attentions=True,
            use_cache=False,
        )
    optimized = runner.probe_attention_vectors(
        model,
        full_ids,
        probes,
        save_full_probe_attention=False,
        sample_dir=sample_dir,
        raw_attentions=outputs.attentions,
    )

    a = legacy["probe_attention"]
    b = optimized["probe_attention"]
    print(f"\nLegacy shape: {a.shape}")
    print(f"Optimized shape: {b.shape}")

    if a.shape != b.shape:
        print("FAIL: shapes differ.")
        sys.exit(1)

    diff = np.abs(a - b)
    print(f"\nMax absolute diff: {diff.max():.6e}")
    print(f"Mean absolute diff: {diff.mean():.6e}")
    print(f"Max relative diff (abs/(|a|+1e-8)): {(diff/(np.abs(a)+1e-8)).max():.6e}")

    delta_norm = np.zeros((1, full_ids.shape[0]), dtype=np.float32)
    legacy_scores, legacy_anchors = runner.summarize_step_scores(
        steps, legacy["probe_metadata"], a, delta_norm
    )
    opt_scores, opt_anchors = runner.summarize_step_scores(
        steps, optimized["probe_metadata"], b, delta_norm
    )
    print(f"\nLegacy anchor ranking:    {legacy_anchors[:5]}")
    print(f"Optimized anchor ranking: {opt_anchors[:5]}")
    if legacy_anchors == opt_anchors:
        print("PASS: anchor ranking identical.")
    elif legacy_anchors[:3] == opt_anchors[:3]:
        print("PASS: top-3 anchors identical (full ranking diverges in tail).")
    else:
        print("WARN: anchor ranking differs in top 3.")
        sys.exit(2)


if __name__ == "__main__":
    main()
