"""Joint anchor-suppression experiment for the camera-ready / paper.

For each Thinking + explicit_cot baseline (the cell with the negative
single-anchor result on GSM8K n=50), suppress BOTH top-2 anchor steps
SIMULTANEOUSLY (union of token positions, union of top-4 layers per
step), and compare to a matched joint control that suppresses 2
random non-anchor steps simultaneously.

This tests three competing interpretations of the negative bar:
  A. distributed causal load → joint anchor drop > sum of singles
  B. anchor-scoring picks late-trace summary steps → joint = single
  C. truly diffuse (no anchor scoring works) → joint anchor ≈ random

We don't rebuild baselines or step scores; we read them from the
existing artifact tree (May-04 deep-table mfix run) and only generate
new joint interventions.

Usage:
  python3 scripts/run_joint_anchor_intervention.py [--smoke]

  --smoke runs on the first 3 samples only to validate the pipeline.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse runner internals
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_qwen3_gsm8k_mi as runner  # type: ignore

# Initialize the runner's lazy imports (it stuffs torch + transformers
# into module-level globals).
runner._ensure_torch_imports() if hasattr(runner, "_ensure_torch_imports") else None
runner.torch = torch
runner.AutoModelForCausalLM = AutoModelForCausalLM
runner.AutoTokenizer = AutoTokenizer
import torch.nn.functional as F  # noqa
runner.F = F

ROOT = Path("/Users/edward/Projects/AJAR")
SOURCE_RUN = ROOT / "outputs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table-mfix-alpha0.5"
OUT_DIR = ROOT / "outputs/joint_anchor_intervention"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_KEY = "thinking"
PROMPT = "explicit_cot"
MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507"
DTYPE = torch.float16
SEED = 17
INTERVENTION_MODES = ["residual_zero", "attention_zero"]
INTERVENTION_MAX_NEW_TOKENS = 384
TOP_ANCHOR_LAYERS = 4
NUM_ANCHORS_JOINT = 2
NUM_CONTROLS_JOINT = 2


def get_sample_dirs() -> List[Path]:
    base = SOURCE_RUN / "mech" / MODEL_KEY / PROMPT
    return sorted(base.glob("sample_*"))


def load_baseline_json(sample_dir: Path) -> Optional[Dict[str, Any]]:
    bj = sample_dir / "baseline.json"
    if not bj.exists():
        return None
    with bj.open() as f:
        return json.load(f)


def parse_steps_from_baseline(
    baseline: Dict[str, Any],
    tokenizer: AutoTokenizer,
) -> Tuple[List[runner.StepSpan], str]:
    """Re-parse generated_text into StepSpans matching what the original
    run produced. We rely on the runner's own parsing so step token
    positions line up exactly with what the saved step_scores indexed."""
    prompt_text = baseline["prompt_text"]
    generated_text = baseline["generated_text"]
    reasoning_text = baseline.get("reasoning_text") or generated_text
    answer_char_span = baseline.get("answer_char_span_generated")

    # Compose full text the same way the runner does
    full_text = prompt_text + generated_text
    full_ids = tokenizer(full_text, add_special_tokens=True, return_tensors="pt").input_ids[0]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=True, return_tensors="pt").input_ids[0]
    prompt_len = len(prompt_ids)

    # Use the runner's split function
    step_segments = runner.split_text_into_steps(
        reasoning_text,
        base_offset=0,
    )
    if not step_segments:
        return [], generated_text

    # The runner's build_step_spans needs token-level alignment. Re-implement
    # a minimal version: tokenize generated_text alone, find boundary tokens
    # via offset_mapping, and assemble StepSpans.
    enc = tokenizer(
        generated_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = enc["offset_mapping"]  # list[(start_char, end_char)] over generated_text

    steps: List[runner.StepSpan] = []
    for step_index, (char_start, char_end, text) in enumerate(step_segments):
        gen_token_positions = [
            i for i, (a, b) in enumerate(offsets)
            if b > char_start and a < char_end
        ]
        if not gen_token_positions:
            continue
        full_token_positions = [prompt_len + i for i in gen_token_positions]
        steps.append(
            runner.StepSpan(
                step_index=step_index,
                text=text,
                char_start_generated=char_start,
                char_end_generated=char_end,
                generated_token_positions=gen_token_positions,
                full_token_positions=full_token_positions,
            )
        )
    return steps, generated_text


def reselect_controls(steps: List[runner.StepSpan], anchor_indices: List[int], n: int = NUM_CONTROLS_JOINT) -> List[int]:
    """Re-sample n control steps with the same seed=17 logic as the
    original runner — top-n of the shuffled non-anchor steps."""
    rng = random.Random(SEED)
    available = [s.step_index for s in steps if s.step_index not in set(anchor_indices)]
    rng.shuffle(available)
    return available[:n]


def union_layers(steps_by_index: Dict[int, runner.StepSpan],
                 step_scores_by_index: Dict[int, Dict[str, Any]],
                 indices: Sequence[int]) -> List[int]:
    """Union of top-4 step-specific layers across the indices."""
    layer_set = set()
    for idx in indices:
        score_row = step_scores_by_index.get(idx)
        if score_row is None:
            continue
        top = score_row.get("top_layers") or []
        for ly in top[:TOP_ANCHOR_LAYERS]:
            layer_set.add(int(ly))
    return sorted(layer_set)


def union_positions(steps_by_index: Dict[int, runner.StepSpan],
                    indices: Sequence[int]) -> List[int]:
    pos_set = set()
    for idx in indices:
        sp = steps_by_index.get(idx)
        if sp is None:
            continue
        for p in sp.full_token_positions:
            pos_set.add(p)
    return sorted(pos_set)


def build_prefix_through_later(prompt_text: str, generated_text: str,
                                steps_by_index: Dict[int, runner.StepSpan],
                                indices: Sequence[int]) -> str:
    """Prefix runs from start of conversation through (and including)
    the LATER of the targeted steps."""
    last_char_end = max(steps_by_index[i].char_end_generated for i in indices)
    return prompt_text + generated_text[:last_char_end]


@dataclasses.dataclass
class JointTask:
    sample_idx: int
    target_kind: str  # "anchor" or "control"
    intervention_mode: str
    target_indices: List[int]
    target_layers: List[int]
    target_token_positions: List[int]
    prefix_text: str
    gold_numeric: Optional[str]
    baseline_correct: bool
    baseline_prediction: Optional[str]


def build_joint_tasks(args, smoke: bool, tokenizer) -> List[JointTask]:
    sample_dirs = get_sample_dirs()
    if smoke:
        sample_dirs = sample_dirs[:3]
    tasks: List[JointTask] = []
    for sample_dir in sample_dirs:
        baseline = load_baseline_json(sample_dir)
        if baseline is None:
            continue
        if baseline.get("analysis_skipped"):
            continue
        sample_idx = int(baseline["sample_idx"])
        anchor_indices = baseline.get("anchor_indices_selected") or []
        if len(anchor_indices) < NUM_ANCHORS_JOINT:
            continue
        anchor_indices = anchor_indices[:NUM_ANCHORS_JOINT]

        steps, generated_text = parse_steps_from_baseline(baseline, tokenizer)
        if not steps:
            continue
        steps_by_index = {s.step_index: s for s in steps}
        if not all(i in steps_by_index for i in anchor_indices):
            continue

        score_rows = baseline.get("step_scores") or []
        scores_by_index = {int(r["step_index"]): r for r in score_rows}

        controls = reselect_controls(steps, anchor_indices, n=NUM_CONTROLS_JOINT)
        if len(controls) < NUM_CONTROLS_JOINT:
            continue

        prompt_text = baseline["prompt_text"]
        for kind, idxs in (("anchor", anchor_indices), ("control", controls)):
            layers = union_layers(steps_by_index, scores_by_index, idxs)
            positions = union_positions(steps_by_index, idxs)
            prefix = build_prefix_through_later(prompt_text, generated_text, steps_by_index, idxs)
            for mode in INTERVENTION_MODES:
                tasks.append(JointTask(
                    sample_idx=sample_idx,
                    target_kind=kind,
                    intervention_mode=mode,
                    target_indices=list(idxs),
                    target_layers=layers,
                    target_token_positions=positions,
                    prefix_text=prefix,
                    gold_numeric=str(baseline.get("gold_numeric")),
                    baseline_correct=bool(baseline.get("correct")),
                    baseline_prediction=baseline.get("prediction_numeric"),
                ))
    return tasks


def make_args() -> Any:
    """Construct the runner.ExperimentConfig for generation calls."""
    cfg = runner.ExperimentConfig(
        inference_backend="torch",
        output_dir=OUT_DIR,
        split="test",
        num_samples=50,
        seed=SEED,
        models=[MODEL_KEY],
        prompts=[PROMPT],
        max_new_tokens=1024,
        intervention_max_new_tokens=INTERVENTION_MAX_NEW_TOKENS,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        num_beams=1,
        dtype="float16",
        attn_implementation="eager",
        device_map="mps",
        parallelize_across_gpus=False,
        model_gpu_allocations={},
        omp_num_threads_per_worker=8,
        trust_remote_code=True,
        max_answer_probes=64,
        top_anchor_steps=NUM_ANCHORS_JOINT,
        num_control_steps=NUM_CONTROLS_JOINT,
        top_anchor_layers=TOP_ANCHOR_LAYERS,
        interventions=INTERVENTION_MODES,
        save_full_hidden_states=False,
        save_full_probe_attention=False,
        resume=False,
        analysis_max_seq_len=4096,
        verbose_logging=False,
        log_baseline_generation=False,
        log_analysis_stages=False,
        log_interventions=False,
        show_progress_postfix=False,
        enable_stage_heartbeat=False,
        heartbeat_interval_seconds=15,
        omlx_base_url="",
        omlx_api_key="",
        omlx_concurrency=1,
        run_mechanistic_analysis=False,
        allow_smoke_dataset=False,
    )
    return cfg


def run_joint_task(task: JointTask, model, tokenizer, args) -> Dict[str, Any]:
    t0 = time.time()
    result = runner.generate_with_intervention(
        model=model,
        tokenizer=tokenizer,
        prefix_text=task.prefix_text,
        target_token_positions=task.target_token_positions,
        target_layers=task.target_layers,
        intervention_mode=task.intervention_mode,
        args=args,
    )
    elapsed = time.time() - t0
    cont_text = result.get("continuation_text") or result.get("generated_text") or ""
    pred_num, boxed = runner.extract_predicted_number(cont_text)
    correct = (pred_num is not None and task.gold_numeric is not None
               and runner.normalize_number_str(pred_num) == runner.normalize_number_str(task.gold_numeric))
    return {
        "sample_idx": task.sample_idx,
        "target_kind": task.target_kind,
        "target_indices": task.target_indices,
        "intervention_mode": task.intervention_mode,
        "n_target_layers": len(task.target_layers),
        "target_layers": task.target_layers,
        "n_target_positions": len(task.target_token_positions),
        "prefix_token_count": len(tokenizer(task.prefix_text, add_special_tokens=True).input_ids),
        "prediction_numeric": pred_num,
        "prediction_boxed_content": boxed,
        "gold_numeric": task.gold_numeric,
        "intervention_correct": int(bool(correct)),
        "baseline_correct": int(task.baseline_correct),
        "continuation_head": cont_text[:200],
        "elapsed_seconds": round(elapsed, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    a = parser.parse_args()

    print(f"Loading tokenizer + model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("  tokenizer loaded")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
    ).to(device).eval()
    print(f"  model loaded on {device}")

    args = make_args()

    print("Building joint intervention tasks ...")
    tasks = build_joint_tasks(args, a.smoke, tokenizer)
    print(f"  built {len(tasks)} joint tasks")

    out_path = OUT_DIR / ("joint_results_smoke.jsonl" if a.smoke else "joint_results.jsonl")
    print(f"Writing to {out_path}")
    with out_path.open("w") as f:
        for i, task in enumerate(tasks):
            print(f"  [{i+1}/{len(tasks)}] sample={task.sample_idx} kind={task.target_kind} mode={task.intervention_mode} "
                  f"target_idx={task.target_indices} n_layers={len(task.target_layers)} n_pos={len(task.target_token_positions)}")
            try:
                result = run_joint_task(task, model, tokenizer, args)
            except Exception as exc:
                print(f"    ERROR: {exc!r}")
                result = {
                    "sample_idx": task.sample_idx,
                    "target_kind": task.target_kind,
                    "intervention_mode": task.intervention_mode,
                    "error": repr(exc),
                }
            f.write(json.dumps(result) + "\n")
            f.flush()
    print("Done.")


if __name__ == "__main__":
    main()
