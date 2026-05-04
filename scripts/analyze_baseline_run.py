#!/usr/bin/env python3
"""Baseline analysis for an AJAR run output directory."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics as stats
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


RUN_DIR = Path(os.environ.get("AJAR_ANALYSIS_RUN_DIR") or (sys.argv[1] if len(sys.argv) > 1 else "outputs/qwen3_gsm8k_real"))
BASELINES_PATH = RUN_DIR / "baselines.jsonl"
OUT_DIR = Path(os.environ.get("AJAR_ANALYSIS_OUT_DIR", f"analysis/{RUN_DIR.name}"))


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def median(values: Iterable[float]) -> float:
    values = list(values)
    return stats.median(values) if values else float("nan")


def percentile(values: Iterable[float], p: float) -> float:
    xs = sorted(values)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100.0
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


def bootstrap_ci(values: list[float], reps: int = 5000, seed: int = 17) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    sampled = []
    for _ in range(reps):
        sampled.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    sampled.sort()
    return sampled[int(0.025 * reps)], sampled[int(0.975 * reps)]


def exact_mcnemar_p(false_to_true: int, true_to_false: int) -> float:
    n = false_to_true + true_to_false
    if n == 0:
        return 1.0
    k = min(false_to_true, true_to_false)
    p = 2.0 * sum(math.comb(n, i) * (0.5**n) for i in range(k + 1))
    return min(1.0, p)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def fmt_num(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def load_rows() -> list[dict[str, Any]]:
    rows = []
    with BASELINES_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            usage = row.get("usage") or {}
            output_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
            row["output_tokens"] = int(output_tokens)
            row["prompt_tokens"] = usage.get("prompt_tokens") or usage.get("input_tokens")
            row["total_tokens"] = usage.get("total_tokens")
            row["has_boxed"] = row.get("prediction_boxed_content") is not None
            row["hit_token_cap"] = row["output_tokens"] >= 512
            row["seconds_per_output_token"] = (
                float(row["generation_seconds"]) / row["output_tokens"]
                if row["output_tokens"]
                else float("nan")
            )
            rows.append(row)
    return rows


@dataclass
class ConditionSummary:
    model_key: str
    prompt_name: str
    n: int
    correct: int
    accuracy: float
    accuracy_ci_low: float
    accuracy_ci_high: float
    mean_seconds: float
    median_seconds: float
    p90_seconds: float
    mean_output_tokens: float
    median_output_tokens: float
    mean_steps: float
    median_steps: float
    stop_rate: float
    length_rate: float
    boxed_rate: float
    cap_rate: float


def summarize_conditions(rows: list[dict[str, Any]]) -> list[ConditionSummary]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_key"], row["prompt_name"])].append(row)

    summaries = []
    for (model_key, prompt_name), group in sorted(grouped.items()):
        correct_values = [float(bool(row["correct"])) for row in group]
        ci_low, ci_high = bootstrap_ci(correct_values, seed=31)
        finish_counts = Counter(row.get("finish_reason") for row in group)
        summaries.append(
            ConditionSummary(
                model_key=model_key,
                prompt_name=prompt_name,
                n=len(group),
                correct=sum(int(bool(row["correct"])) for row in group),
                accuracy=mean(correct_values),
                accuracy_ci_low=ci_low,
                accuracy_ci_high=ci_high,
                mean_seconds=mean(float(row["generation_seconds"]) for row in group),
                median_seconds=median(float(row["generation_seconds"]) for row in group),
                p90_seconds=percentile((float(row["generation_seconds"]) for row in group), 90),
                mean_output_tokens=mean(float(row["output_tokens"]) for row in group),
                median_output_tokens=median(float(row["output_tokens"]) for row in group),
                mean_steps=mean(float(row["num_steps"]) for row in group),
                median_steps=median(float(row["num_steps"]) for row in group),
                stop_rate=finish_counts["stop"] / len(group),
                length_rate=finish_counts["length"] / len(group),
                boxed_rate=mean(float(bool(row["has_boxed"])) for row in group),
                cap_rate=mean(float(bool(row["hit_token_cap"])) for row in group),
            )
        )
    return summaries


def write_condition_summary(summaries: list[ConditionSummary]) -> None:
    path = OUT_DIR / "condition_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ConditionSummary.__annotations__.keys()))
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.__dict__)


def finish_reason_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_key"], row["prompt_name"], str(row.get("finish_reason")))].append(row)
    output = []
    for (model_key, prompt_name, finish_reason), group in sorted(grouped.items()):
        output.append(
            {
                "model_key": model_key,
                "prompt_name": prompt_name,
                "finish_reason": finish_reason,
                "n": len(group),
                "accuracy": mean(float(bool(row["correct"])) for row in group),
                "boxed_rate": mean(float(bool(row["has_boxed"])) for row in group),
                "median_output_tokens": median(float(row["output_tokens"]) for row in group),
                "median_steps": median(float(row["num_steps"]) for row in group),
            }
        )
    return output


def write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def paired_prompt_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model_sample: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_model_sample[(row["model_key"], int(row["sample_idx"]))][row["prompt_name"]] = row

    output = []
    prompt_pairs = [
        ("neutral", "explicit_cot"),
        ("neutral", "explicit_no_cot"),
        ("explicit_cot", "explicit_no_cot"),
    ]
    for model_key in sorted({row["model_key"] for row in rows}):
        for left_prompt, right_prompt in prompt_pairs:
            pairs = [
                pair
                for (model, _), pair in by_model_sample.items()
                if model == model_key and {left_prompt, right_prompt}.issubset(pair)
            ]
            if not pairs:
                continue
            accuracy_delta = [
                float(bool(pair[left_prompt]["correct"])) - float(bool(pair[right_prompt]["correct"]))
                for pair in pairs
            ]
            ci_low, ci_high = bootstrap_ci(accuracy_delta, seed=43)
            right_wrong_left_right = sum(
                (not pair[right_prompt]["correct"]) and bool(pair[left_prompt]["correct"]) for pair in pairs
            )
            right_right_left_wrong = sum(
                bool(pair[right_prompt]["correct"]) and (not pair[left_prompt]["correct"]) for pair in pairs
            )
            output.append(
                {
                    "model_key": model_key,
                    "left_prompt": left_prompt,
                    "right_prompt": right_prompt,
                    "pairs": len(pairs),
                    "left_minus_right_accuracy": mean(accuracy_delta),
                    "accuracy_delta_ci_low": ci_low,
                    "accuracy_delta_ci_high": ci_high,
                    "right_wrong_left_right": right_wrong_left_right,
                    "right_right_left_wrong": right_right_left_wrong,
                    "mcnemar_p": exact_mcnemar_p(right_wrong_left_right, right_right_left_wrong),
                    "left_minus_right_seconds": mean(
                        float(pair[left_prompt]["generation_seconds"])
                        - float(pair[right_prompt]["generation_seconds"])
                        for pair in pairs
                    ),
                    "left_minus_right_tokens": mean(
                        float(pair[left_prompt]["output_tokens"]) - float(pair[right_prompt]["output_tokens"])
                        for pair in pairs
                    ),
                    "left_minus_right_steps": mean(
                        float(pair[left_prompt]["num_steps"]) - float(pair[right_prompt]["num_steps"])
                        for pair in pairs
                    ),
                }
            )
    return output


def paired_model_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_prompt_sample: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_prompt_sample[(row["prompt_name"], int(row["sample_idx"]))][row["model_key"]] = row

    output = []
    for prompt_name in sorted({row["prompt_name"] for row in rows}):
        pairs = [
            pair
            for (prompt, _), pair in by_prompt_sample.items()
            if prompt == prompt_name and {"thinking", "instruct"}.issubset(pair)
        ]
        accuracy_delta = [
            float(bool(pair["instruct"]["correct"])) - float(bool(pair["thinking"]["correct"]))
            for pair in pairs
        ]
        ci_low, ci_high = bootstrap_ci(accuracy_delta, seed=47)
        thinking_wrong_instruct_right = sum(
            (not pair["thinking"]["correct"]) and bool(pair["instruct"]["correct"]) for pair in pairs
        )
        thinking_right_instruct_wrong = sum(
            bool(pair["thinking"]["correct"]) and (not pair["instruct"]["correct"]) for pair in pairs
        )
        output.append(
            {
                "prompt_name": prompt_name,
                "pairs": len(pairs),
                "instruct_minus_thinking_accuracy": mean(accuracy_delta),
                "accuracy_delta_ci_low": ci_low,
                "accuracy_delta_ci_high": ci_high,
                "thinking_wrong_instruct_right": thinking_wrong_instruct_right,
                "thinking_right_instruct_wrong": thinking_right_instruct_wrong,
                "mcnemar_p": exact_mcnemar_p(
                    thinking_wrong_instruct_right,
                    thinking_right_instruct_wrong,
                ),
                "instruct_minus_thinking_seconds": mean(
                    float(pair["instruct"]["generation_seconds"])
                    - float(pair["thinking"]["generation_seconds"])
                    for pair in pairs
                ),
                "instruct_minus_thinking_tokens": mean(
                    float(pair["instruct"]["output_tokens"])
                    - float(pair["thinking"]["output_tokens"])
                    for pair in pairs
                ),
                "instruct_minus_thinking_steps": mean(
                    float(pair["instruct"]["num_steps"]) - float(pair["thinking"]["num_steps"])
                    for pair in pairs
                ),
            }
        )
    return output


def collect_disagreements(rows: list[dict[str, Any]], limit: int = 40) -> list[dict[str, Any]]:
    by_model_sample: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_model_sample[(row["model_key"], int(row["sample_idx"]))][row["prompt_name"]] = row

    disagreements = []
    for (model_key, sample_idx), pair in sorted(by_model_sample.items()):
        if not {"explicit_cot", "neutral"}.issubset(pair):
            continue
        explicit = pair["explicit_cot"]
        neutral = pair["neutral"]
        if bool(explicit["correct"]) == bool(neutral["correct"]):
            continue
        disagreements.append(
            {
                "model_key": model_key,
                "sample_idx": sample_idx,
                "gold_numeric": neutral["gold_numeric"],
                "explicit_correct": int(bool(explicit["correct"])),
                "neutral_correct": int(bool(neutral["correct"])),
                "explicit_prediction": explicit.get("prediction_numeric"),
                "neutral_prediction": neutral.get("prediction_numeric"),
                "explicit_finish_reason": explicit.get("finish_reason"),
                "neutral_finish_reason": neutral.get("finish_reason"),
                "explicit_output_tokens": explicit["output_tokens"],
                "neutral_output_tokens": neutral["output_tokens"],
                "question": neutral["question"],
            }
        )
    return disagreements[:limit]


def render_report(
    rows: list[dict[str, Any]],
    condition_summaries: list[ConditionSummary],
    finish_rows: list[dict[str, Any]],
    prompt_deltas: list[dict[str, Any]],
    model_deltas: list[dict[str, Any]],
) -> str:
    run_config = json.loads((RUN_DIR / "run_config.json").read_text(encoding="utf-8"))
    config = run_config["config"]
    sample_count = len({row["sample_idx"] for row in rows})
    prompts = ", ".join(config["prompts"])
    models = ", ".join(config["models"])
    prompt_names = {row["prompt_name"] for row in rows}
    best_summary = max(condition_summaries, key=lambda summary: summary.accuracy)
    hcds_missing = [
        label
        for label, is_missing in [
            ("explicit No-CoT condition", "explicit_no_cot" not in prompt_names),
            ("entropy/logprob traces", True),
            ("perturbation runs", True),
            ("paraphrase consistency", True),
            ("mechanistic interventions", True),
        ]
        if is_missing
    ]

    lines = [
        "# Qwen3 GSM8K oMLX Initial Analysis",
        "",
        "## Run Scope",
        "",
        f"- Output directory: `{RUN_DIR}`",
        f"- Dataset: GSM8K `{config['split']}` split, first {sample_count} examples",
        f"- Rows analyzed: {len(rows)} baseline generations",
        f"- Models: {models}",
        f"- Prompt conditions: {prompts}",
        f"- Decoding: deterministic, `temperature={config['temperature']}`, `max_new_tokens={config['max_new_tokens']}`",
        "- Mechanistic analysis and interventions: not run in oMLX mode",
        "",
        "## Main Result",
        "",
        f"The strongest condition in this run is `{best_summary.model_key}` / `{best_summary.prompt_name}` at {best_summary.correct}/{best_summary.n} correct ({fmt_pct(best_summary.accuracy)}). The Thinking model aggregate accuracy should be treated carefully because many Thinking generations hit `finish_reason=length`, often before a boxed final answer.",
        "",
        "## Condition Summary",
        "",
        "| Model | Prompt | Accuracy | 95% bootstrap CI | Stop rate | Length rate | Boxed rate | Mean seconds | Mean output tokens | Mean steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in condition_summaries:
        lines.append(
            "| "
            f"{summary.model_key} | {summary.prompt_name} | "
            f"{summary.correct}/{summary.n} ({fmt_pct(summary.accuracy)}) | "
            f"{fmt_pct(summary.accuracy_ci_low)}-{fmt_pct(summary.accuracy_ci_high)} | "
            f"{fmt_pct(summary.stop_rate)} | {fmt_pct(summary.length_rate)} | "
            f"{fmt_pct(summary.boxed_rate)} | {fmt_num(summary.mean_seconds)} | "
            f"{fmt_num(summary.mean_output_tokens, 1)} | {fmt_num(summary.mean_steps, 1)} |"
        )

    lines.extend(
        [
            "",
            "## Prompt Effect",
            "",
            "| Model | Prompt contrast | Accuracy delta | 95% bootstrap CI | Discordant pairs | McNemar p | Seconds delta | Tokens delta |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in prompt_deltas:
        lines.append(
            "| "
            f"{row['model_key']} | {row['left_prompt']} - {row['right_prompt']} | "
            f"{fmt_pct(row['left_minus_right_accuracy'])} | "
            f"{fmt_pct(row['accuracy_delta_ci_low'])}-{fmt_pct(row['accuracy_delta_ci_high'])} | "
            f"{row['right_wrong_left_right']} improved / {row['right_right_left_wrong']} regressed | "
            f"{row['mcnemar_p']:.4g} | {fmt_num(row['left_minus_right_seconds'])} | "
            f"{fmt_num(row['left_minus_right_tokens'], 1)} |"
        )

    lines.extend(
        [
            "",
            "## Model Effect",
            "",
            "| Prompt | Instruct - Thinking accuracy | 95% bootstrap CI | Discordant pairs | McNemar p | Instruct - Thinking seconds | Instruct - Thinking tokens |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in model_deltas:
        lines.append(
            "| "
            f"{row['prompt_name']} | {fmt_pct(row['instruct_minus_thinking_accuracy'])} | "
            f"{fmt_pct(row['accuracy_delta_ci_low'])}-{fmt_pct(row['accuracy_delta_ci_high'])} | "
            f"{row['thinking_wrong_instruct_right']} Instruct-only / {row['thinking_right_instruct_wrong']} Thinking-only | "
            f"{row['mcnemar_p']:.4g} | {fmt_num(row['instruct_minus_thinking_seconds'])} | "
            f"{fmt_num(row['instruct_minus_thinking_tokens'], 1)} |"
        )

    lines.extend(
        [
            "",
            "## Truncation Check",
            "",
            "| Model | Prompt | Finish reason | n | Accuracy | Boxed rate | Median tokens | Median steps |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in finish_rows:
        lines.append(
            "| "
            f"{row['model_key']} | {row['prompt_name']} | {row['finish_reason']} | "
            f"{row['n']} | {fmt_pct(row['accuracy'])} | {fmt_pct(row['boxed_rate'])} | "
            f"{fmt_num(row['median_output_tokens'], 0)} | {fmt_num(row['median_steps'], 0)} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- The strongest baseline result is `{best_summary.model_key}` under `{best_summary.prompt_name}`: {best_summary.correct}/{best_summary.n} correct, {fmt_pct(best_summary.accuracy)}.",
            "- Explicit CoT is not helping on this run. It costs tokens and latency, and it increases truncation risk.",
            "- The Thinking model should not be judged from aggregate accuracy until the max-token cap is fixed. When it stops normally, its accuracy is near the Instruct model; when it hits the cap, accuracy collapses and boxed-answer extraction usually fails.",
            f"- This run does not yet support a full Hidden CoT Detection Score because it lacks: {', '.join(hcds_missing)}.",
            "- Treat current results as a successful throughput and baseline-behavior run, not as final evidence for or against hidden CoT.",
            "",
            "## Recommended Next Run",
            "",
            "1. Add the third prompt condition: explicit No-CoT / answer-only.",
            "2. Increase Thinking model `max_new_tokens` substantially, or use a two-budget setup: small answer-only budget and larger CoT/thinking budget.",
            "3. Preserve boxed-answer formatting across all conditions so scoring remains comparable.",
            "4. Add async OpenAI-compatible oMLX requests with a configurable concurrency limit and benchmark concurrency levels 1, 2, 4, and 8.",
            "5. Add a perturbation/paraphrase slice before scaling to full GSM8K: enough to compute preliminary consistency and perturbation fragility.",
            "6. Use MLX-LM Python or a Torch backend for logits/entropy and mechanistic traces; the current oMLX API result records do not include token-level entropy.",
            "",
            "## Generated Files",
            "",
            "- `condition_summary.csv`",
            "- `finish_reason_summary.csv`",
            "- `paired_prompt_deltas.csv`",
            "- `paired_model_deltas.csv`",
            "- `prompt_disagreement_examples.csv`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    condition_summaries = summarize_conditions(rows)
    finish_rows = finish_reason_summary(rows)
    prompt_deltas = paired_prompt_deltas(rows)
    model_deltas = paired_model_deltas(rows)
    disagreements = collect_disagreements(rows)

    write_condition_summary(condition_summaries)
    write_dict_rows(OUT_DIR / "finish_reason_summary.csv", finish_rows)
    write_dict_rows(OUT_DIR / "paired_prompt_deltas.csv", prompt_deltas)
    write_dict_rows(OUT_DIR / "paired_model_deltas.csv", model_deltas)
    write_dict_rows(OUT_DIR / "prompt_disagreement_examples.csv", disagreements)
    (OUT_DIR / "initial_report.md").write_text(
        render_report(rows, condition_summaries, finish_rows, prompt_deltas, model_deltas),
        encoding="utf-8",
    )
    print(f"Wrote analysis outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
