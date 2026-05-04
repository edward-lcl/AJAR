# AJAR Hidden CoT Experiments

This repository contains the current AJAR experiment runner, baseline GSM8K results, and analysis summaries for the Qwen3 hidden chain-of-thought project.

## Current Status

We have completed two oMLX baseline runs on the first 500 GSM8K test examples:

1. Initial baseline: 2 models x 2 prompts = 2,000 generations.
2. Rerun baseline: 2 models x 3 prompts = 3,000 generations.

The rerun added the explicit No-CoT / answer-only condition, increased the token budget from 512 to 1024, and used oMLX concurrency 4.

## Key Result

The rerun finished cleanly with 3,000/3,000 generations and 0 failures.

| Model | Prompt | Accuracy |
|---|---|---:|
| Qwen3-4B-Instruct | explicit CoT | 93.6% |
| Qwen3-4B-Instruct | neutral | 92.6% |
| Qwen3-4B-Instruct | explicit No-CoT | 41.8% |
| Qwen3-4B-Thinking | explicit CoT | 54.0% |
| Qwen3-4B-Thinking | neutral | 67.6% |
| Qwen3-4B-Thinking | explicit No-CoT | 82.0% |

Interpretation: the Instruct model stays strong under explicit CoT and neutral prompting but collapses under strict answer-only prompting. The Thinking model is still heavily affected by truncation, although the larger token cap improved results substantially.

## Repository Layout

- `scripts/run_qwen3_gsm8k_mi.py`: main experiment runner.
- `scripts/run_omlx_baseline_rerun.sh`: reproducible oMLX rerun command wrapper.
- `scripts/analyze_baseline_run.py`: analysis script for any run output directory.
- `results/RUN_INDEX.md`: index of committed run summaries.
- `results/runs/2026-05-03_2223_gsm8k500_qwen3-4b_omlx_baseline-v1/`: initial run reports and summary CSVs.
- `results/runs/2026-05-04_0238_gsm8k500_qwen3-4b_omlx_baseline-rerun-nocot-1024/`: rerun reports, summary CSVs, and Task 6 starter table.
- `docs/task_list.md`: task list provided by the team.
- `docs/run_naming.md`: naming convention for future evals.
- `docs/mechanistic_backend_plan.md`: plan for generating actual mechanistic data.

Large raw generated outputs are intentionally not committed. They live under local `outputs/` and can be shared via Drive or release artifacts.

## Run Naming

New runs should use:

```text
YYYY-MM-DD_HHMM_<dataset><sample-count>_<model-family>_<backend>_<purpose>
```

Example:

```text
2026-05-04_0238_gsm8k500_qwen3-4b_omlx_baseline-rerun-nocot-1024
```

See `docs/run_naming.md` for details.

## Mechanistic Data Caveat

The oMLX runs are text-generation-only. They do not expose hidden states, attentions, logits/logprobs, or forward hooks. Therefore the current Task 6 table contains baseline metrics and placeholders for entropy, perturbation, paraphrase, and mechanistic intervention columns.

To generate actual mechanistic data, run a separate Torch backend slice with `AJAR_BACKEND=torch` and `AJAR_RUN_MI=1`.
