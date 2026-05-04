#!/usr/bin/env python
"""Build the Task 6 structured table by joining baseline, paraphrase,
perturbation, and mechanistic-intervention runs.

Each output row corresponds to one (model, dataset, prompt, original_sample_idx)
and contains every column requested in the proposal:
- accuracy
- latency_per_token, generation_seconds, output_tokens
- token_entropy_mean, token_entropy_slope
- paraphrase_consistency, paraphrase_accuracy
- perturbation_delta_accuracy
- mechanistic_intervention_delta_accuracy (anchor − control), and the raw
  anchor and control means it was computed from
- mechanistic_available

The script is tolerant of missing inputs: if --paraphrase-dir is omitted, the
paraphrase columns stay blank; same for --perturbation-dir and --mi-dir.

Inputs:
- --baseline-dir   Run dir produced by run_qwen3_gsm8k_mi.py on the canonical
                   GSM8K split. Contains baselines.jsonl with one row per
                   (model, prompt, sample_idx).
- --paraphrase-dir Run dir produced on a paraphrase fixture, alongside the
                   --paraphrase-index pointing at the matching index.csv.
- --perturbation-dir Run dir produced on a perturbation fixture, plus
                     --perturbation-index.
- --mi-dir         Run dir from a torch backend MI run, used to read
                   interventions.jsonl and compute anchor-vs-control accuracy
                   drop per (model, prompt, sample_idx).

Usage:
    python3 scripts/build_task6_table.py \\
        --baseline-dir outputs/2026-05-04_0238_gsm8k500_qwen3-4b_omlx_baseline-rerun-nocot-1024 \\
        --paraphrase-dir outputs/2026-05-05_xxxx_gsm8k50_qwen3-4b_omlx_paraphrase \\
        --paraphrase-index data/variants/gsm8k_test_first50_para/index.csv \\
        --perturbation-dir outputs/2026-05-05_xxxx_gsm8k50_qwen3-4b_omlx_perturb \\
        --perturbation-index data/variants/gsm8k_test_first50_perturb/index.csv \\
        --mi-dir outputs/2026-05-05_xxxx_gsm8k50_qwen3-4b_torch_mech-slice \\
        --out results/task6_table.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def index_baselines(baselines: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    out: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for row in baselines:
        key = (str(row["model_key"]), str(row["prompt_name"]), int(row["sample_idx"]))
        out[key] = row
    return out


def safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def variant_baselines_by_original(
    variant_baselines: Iterable[Dict[str, Any]],
    index_rows: Iterable[Dict[str, str]],
    keep_kinds: Tuple[str, ...],
    label: str,
    require_numbers_preserved: bool = False,
) -> Tuple[Dict[Tuple[str, str, int], List[Dict[str, Any]]], Dict[str, int]]:
    """Group variant baseline rows by (model, prompt, original_sample_idx).

    The variant runner uses the row index inside the JSONL fixture as its
    sample_idx. We invert the index.csv mapping to recover the original
    sample id, then keep only the variant kinds we care about.

    When require_numbers_preserved is set, rows whose index meta carries
    numbers_preserved="false" are excluded. This is the right setting for
    paraphrases, where a paraphrase that adds or removes numbers may have
    silently leaked the answer or changed the underlying problem; it is
    irrelevant for perturbations, where numbers_preserved is not tracked.

    Returns the grouping plus a small audit dict that the caller can surface
    to the operator.
    """
    row_to_meta: Dict[int, Dict[str, str]] = {
        int(row["row_idx"]): row for row in index_rows
    }
    out: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    audit = {
        "baselines_without_index": 0,
        "filtered_by_kind": 0,
        "filtered_numbers_not_preserved": 0,
    }
    seen_baseline_rows = set()
    for baseline in variant_baselines:
        row_idx = int(baseline["sample_idx"])
        seen_baseline_rows.add(row_idx)
        meta = row_to_meta.get(row_idx)
        if meta is None:
            audit["baselines_without_index"] += 1
            continue
        if meta["variant_kind"] not in keep_kinds:
            audit["filtered_by_kind"] += 1
            continue
        if (
            require_numbers_preserved
            and meta.get("numbers_preserved", "true").lower() == "false"
        ):
            audit["filtered_numbers_not_preserved"] += 1
            continue
        original_idx = int(meta["original_sample_idx"])
        key = (str(baseline["model_key"]), str(baseline["prompt_name"]), original_idx)
        out[key].append({**baseline, "_variant_meta": meta})
    expected_rows = {
        int(row["row_idx"])
        for row in index_rows
        if row["variant_kind"] in keep_kinds
        and (
            not require_numbers_preserved
            or row.get("numbers_preserved", "true").lower() != "false"
        )
    }
    audit["index_rows_without_baseline"] = len(expected_rows - seen_baseline_rows)
    audit["label"] = label
    return out, audit


def compute_paraphrase_metrics(
    original_baseline: Dict[str, Any],
    paraphrase_rows: List[Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    if not paraphrase_rows:
        return {
            "paraphrase_consistency": None,
            "paraphrase_accuracy": None,
            "paraphrase_count": 0,
        }
    original_pred = original_baseline.get("prediction_numeric")
    agreements = []
    accuracies = []
    for row in paraphrase_rows:
        pred = row.get("prediction_numeric")
        agreements.append(1 if pred is not None and pred == original_pred else 0)
        accuracies.append(int(bool(row.get("correct"))))
    return {
        "paraphrase_consistency": float(mean(agreements)),
        "paraphrase_accuracy": float(mean(accuracies)),
        "paraphrase_count": len(paraphrase_rows),
    }


def compute_perturbation_metrics(
    original_baseline: Dict[str, Any],
    perturbation_rows: List[Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    if not perturbation_rows:
        return {
            "perturbation_delta_accuracy": None,
            "perturbation_accuracy": None,
            "perturbation_count": 0,
        }
    perturbed_acc = float(mean(int(bool(row.get("correct"))) for row in perturbation_rows))
    original_acc = float(int(bool(original_baseline.get("correct"))))
    return {
        # Original − perturbed; positive = perturbation hurt this question.
        "perturbation_delta_accuracy": original_acc - perturbed_acc,
        "perturbation_accuracy": perturbed_acc,
        "perturbation_count": len(perturbation_rows),
    }


def index_interventions(
    intervention_rows: Iterable[Dict[str, Any]],
) -> Dict[Tuple[str, str, int], Dict[str, List[int]]]:
    """For each (model, prompt, sample_idx) collect anchor- and control-step
    correctness lists from the intervention summaries."""
    out: Dict[Tuple[str, str, int], Dict[str, List[int]]] = defaultdict(
        lambda: {"anchor": [], "control": []}
    )
    for row in intervention_rows:
        key = (
            str(row["model_key"]),
            str(row["prompt_name"]),
            int(row["sample_idx"]),
        )
        kind = str(row.get("step_kind", ""))
        if kind not in ("anchor", "control"):
            continue
        out[key][kind].append(int(bool(row.get("correct"))))
    return out


def compute_mi_metrics(
    original_baseline: Dict[str, Any],
    intervention_buckets: Dict[str, List[int]],
) -> Dict[str, Optional[float]]:
    has_anchor = bool(intervention_buckets.get("anchor"))
    has_control = bool(intervention_buckets.get("control"))
    if not has_anchor and not has_control:
        return {
            "mechanistic_anchor_accuracy": None,
            "mechanistic_control_accuracy": None,
            "mechanistic_intervention_delta_accuracy": None,
            "mechanistic_anchor_count": 0,
            "mechanistic_control_count": 0,
            "mechanistic_available": 0,
        }
    base_correct = float(int(bool(original_baseline.get("correct"))))
    anchor_acc = float(mean(intervention_buckets["anchor"])) if has_anchor else None
    control_acc = float(mean(intervention_buckets["control"])) if has_control else None
    anchor_drop = (base_correct - anchor_acc) if anchor_acc is not None else None
    control_drop = (base_correct - control_acc) if control_acc is not None else None
    if anchor_drop is not None and control_drop is not None:
        delta = anchor_drop - control_drop
    elif anchor_drop is not None:
        delta = anchor_drop
    else:
        delta = None
    return {
        "mechanistic_anchor_accuracy": anchor_acc,
        "mechanistic_control_accuracy": control_acc,
        "mechanistic_intervention_delta_accuracy": delta,
        "mechanistic_anchor_count": len(intervention_buckets.get("anchor", [])),
        "mechanistic_control_count": len(intervention_buckets.get("control", [])),
        "mechanistic_available": 1,
    }


def _emit_audit(audit: Dict[str, int]) -> None:
    label = audit.get("label", "variant")
    msgs = []
    if audit.get("baselines_without_index"):
        msgs.append(
            f"{audit['baselines_without_index']} {label} baseline row(s) had no "
            f"matching row in the index.csv (likely the wrong --{label}-index)"
        )
    if audit.get("index_rows_without_baseline"):
        msgs.append(
            f"{audit['index_rows_without_baseline']} {label} index row(s) had "
            f"no matching baseline (the variant run likely did not cover the "
            f"full fixture or used different prompt/model coverage)"
        )
    if audit.get("filtered_numbers_not_preserved"):
        msgs.append(
            f"{audit['filtered_numbers_not_preserved']} {label} row(s) dropped "
            f"because numbers_preserved=false (faithless paraphrase)"
        )
    for msg in msgs:
        print(f"WARNING: {msg}", file=sys.stderr)


def build_table(
    baseline_dir: Path,
    paraphrase_dir: Optional[Path],
    paraphrase_index: Optional[Path],
    perturbation_dir: Optional[Path],
    perturbation_index: Optional[Path],
    mi_dir: Optional[Path],
    dataset_label: str,
    trial_id: int,
    require_paraphrase_numbers_preserved: bool = True,
) -> List[Dict[str, Any]]:
    baselines = index_baselines(read_jsonl(baseline_dir / "baselines.jsonl"))

    paraphrase_index_rows: List[Dict[str, str]] = (
        read_csv_rows(paraphrase_index) if paraphrase_index else []
    )
    paraphrase_baselines = (
        read_jsonl(paraphrase_dir / "baselines.jsonl") if paraphrase_dir else []
    )
    paraphrase_grouped, paraphrase_audit = variant_baselines_by_original(
        paraphrase_baselines,
        paraphrase_index_rows,
        keep_kinds=("paraphrase",),
        label="paraphrase",
        require_numbers_preserved=require_paraphrase_numbers_preserved,
    )

    perturbation_index_rows: List[Dict[str, str]] = (
        read_csv_rows(perturbation_index) if perturbation_index else []
    )
    perturbation_baselines = (
        read_jsonl(perturbation_dir / "baselines.jsonl") if perturbation_dir else []
    )
    perturbation_grouped, perturbation_audit = variant_baselines_by_original(
        perturbation_baselines,
        perturbation_index_rows,
        keep_kinds=("perturbation",),
        label="perturbation",
    )

    intervention_rows: List[Dict[str, Any]] = []
    mi_baselines: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    if mi_dir is not None:
        intervention_path = mi_dir / "interventions.jsonl"
        if intervention_path.exists():
            intervention_rows = read_jsonl(intervention_path)
        mi_baselines_path = mi_dir / "baselines.jsonl"
        if mi_baselines_path.exists():
            mi_baselines = index_baselines(read_jsonl(mi_baselines_path))
    intervention_index = index_interventions(intervention_rows)
    if paraphrase_dir is not None:
        _emit_audit(paraphrase_audit)
    if perturbation_dir is not None:
        _emit_audit(perturbation_audit)

    out: List[Dict[str, Any]] = []
    for key, baseline in sorted(baselines.items()):
        model_key, prompt_name, sample_idx = key
        gen_secs = baseline.get("generation_seconds")
        out_tokens = baseline.get("generated_len_tokens") or baseline.get("usage", {}).get(
            "completion_tokens"
        )
        latency_per_token = safe_div(gen_secs, out_tokens)
        para = compute_paraphrase_metrics(baseline, paraphrase_grouped.get(key, []))
        pert = compute_perturbation_metrics(baseline, perturbation_grouped.get(key, []))
        mi_buckets = intervention_index.get(key, {"anchor": [], "control": []})
        mi = compute_mi_metrics(baseline, mi_buckets)
        row = {
            "dataset": dataset_label,
            "model": model_key,
            "model_name": baseline.get("model_name"),
            "prompt_condition": prompt_name,
            "sample_idx": sample_idx,
            "trial": trial_id,
            "question": baseline.get("question"),
            "gold_numeric": baseline.get("gold_numeric"),
            "prediction_numeric": baseline.get("prediction_numeric"),
            "accuracy": int(bool(baseline.get("correct"))),
            "generation_seconds": gen_secs,
            "output_tokens": out_tokens,
            "latency_per_output_token": latency_per_token,
            "output_steps": baseline.get("num_steps"),
            "finish_reason": baseline.get("finish_reason"),
            "boxed_answer_present": int(baseline.get("prediction_boxed_content") is not None),
            "token_entropy_mean": baseline.get("mean_entropy_generated")
            or mi_baselines.get(key, {}).get("mean_entropy_generated"),
            "token_entropy_slope": baseline.get("entropy_slope_generated")
            if baseline.get("entropy_slope_generated") is not None
            else mi_baselines.get(key, {}).get("entropy_slope_generated"),
            "token_entropy_source": "baseline" if baseline.get("mean_entropy_generated") is not None else (
                "mi" if mi_baselines.get(key, {}).get("mean_entropy_generated") is not None else ""
            ),
            **para,
            **pert,
            **mi,
        }
        out.append(row)
    return out


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--paraphrase-dir", type=Path, default=None)
    parser.add_argument("--paraphrase-index", type=Path, default=None)
    parser.add_argument("--perturbation-dir", type=Path, default=None)
    parser.add_argument("--perturbation-index", type=Path, default=None)
    parser.add_argument("--mi-dir", type=Path, default=None)
    parser.add_argument("--dataset-label", default="GSM8K")
    parser.add_argument("--trial-id", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--keep-faithless-paraphrases",
        action="store_true",
        help=(
            "Include paraphrases flagged numbers_preserved=false in the "
            "consistency calculation. Default behaviour is to drop them so "
            "consistency reflects only paraphrases that preserved the "
            "underlying numeric structure."
        ),
    )
    args = parser.parse_args()

    if (args.paraphrase_dir is None) != (args.paraphrase_index is None):
        raise SystemExit("--paraphrase-dir and --paraphrase-index must be set together.")
    if (args.perturbation_dir is None) != (args.perturbation_index is None):
        raise SystemExit("--perturbation-dir and --perturbation-index must be set together.")

    rows = build_table(
        baseline_dir=args.baseline_dir,
        paraphrase_dir=args.paraphrase_dir,
        paraphrase_index=args.paraphrase_index,
        perturbation_dir=args.perturbation_dir,
        perturbation_index=args.perturbation_index,
        mi_dir=args.mi_dir,
        dataset_label=args.dataset_label,
        trial_id=args.trial_id,
        require_paraphrase_numbers_preserved=not args.keep_faithless_paraphrases,
    )
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} row(s) to {args.out}.")


if __name__ == "__main__":
    main()
