# AJAR Hidden CoT Experiments

This repository contains the current AJAR experiment runner, baseline GSM8K results, and analysis summaries for the Qwen3 hidden chain-of-thought project.

## Setup

Python 3.10 or newer is recommended; macOS users on the system Python 3.9 will work but pay a urllib3 / LibreSSL warning every run (the runner suppresses it now, but other tools in the team's stack may not).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt          # runtime
pip install -r requirements-dev.txt      # adds pytest

# Quick sanity: every code path the runner relies on is unit-tested.
pytest                                    # 17 tests, < 1s
```

CUDA hosts: install the matching `torch` wheel from pytorch.org first, then `pip install -r requirements.txt --no-deps` (or just rely on pip's resolver and accept that it may pick a CPU-only build).

The runner fails fast at startup if backend dependencies are missing — `AJAR_BACKEND=torch` checks numpy/torch/transformers/datasets, `AJAR_BACKEND=omlx` checks the local oMLX server is reachable and the API key resolves. You will see one clear error message instead of a half-completed run that crashes inside a worker.

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

- `scripts/run_qwen3_gsm8k_mi.py`: main experiment runner. Backends: oMLX (baseline-only) and torch (full MI).
- `scripts/run_omlx_baseline_rerun.sh`: reproducible oMLX rerun command wrapper.
- `scripts/run_torch_mechanistic_slice.sh`: torch backend slice for entropy + MI artifacts.
- `scripts/build_perturbations.py`: generate distractor-irrelevant perturbation fixtures (Task 2 / Task 9).
- `scripts/build_paraphrases.py`: generate paraphrase fixtures via local oMLX Instruct (Task 6 paraphrase consistency).
- `scripts/build_task6_table.py`: join baseline + paraphrase + perturbation + MI runs into the Task 6 structured table.
- `scripts/aggregate_anchor_sensitivity.py`: Task 10 anchor-vs-control accuracy drop aggregation.
- `scripts/analyze_baseline_run.py`: per-run report generator (oMLX-style baseline analysis).
- `results/RUN_INDEX.md`: index of committed run summaries.
- `docs/task_list.md`: task list provided by the team.
- `docs/run_naming.md`: naming convention for future evals.
- `docs/mechanistic_backend_plan.md`: plan for generating actual mechanistic data.

Large raw generated outputs are intentionally not committed. They live under local `outputs/` and can be shared via Drive or release artifacts.

## Filling the Task 6 placeholder columns

Three Task 6 columns previously sat as placeholders for everyone (not just on Mac): `token_entropy_mean/slope`, `paraphrase_consistency`, `perturbation_delta_accuracy`. The pipeline now fills all three:

1. **Token entropy** (torch backend only — oMLX does not return logprobs even when requested). The torch backend computes full-distribution entropy `H = -Σ p · log p` per generated position from logits, saves `token_entropy` to `hidden_summary.npz` and `tokens.csv`, and surfaces `mean_entropy_generated` + `entropy_slope_generated` on each baseline record.
2. **Paraphrase consistency**: `scripts/build_paraphrases.py` generates K paraphrases per question via the local oMLX Instruct model with constraints to preserve numbers and logical structure. The runner consumes the resulting JSONL via `GSM8K_JSONL=...`. The aggregator reports both agreement rate vs the original prediction and accuracy across paraphrases.
3. **Perturbation Δ accuracy**: `scripts/build_perturbations.py` produces a fixture with the original passthrough + a distractor-irrelevant variant (gold unchanged) per question. The aggregator reports `perturbation_delta_accuracy = original_correct − perturbed_correct`.

The aggregator (`scripts/build_task6_table.py`) joins all four sources by `(model_key, prompt_name, original_sample_idx)`. If the baseline dir was an oMLX run with no entropy, the aggregator falls back to entropy from the MI dir's baselines and records the source in the `token_entropy_source` column.

### Recommended deep-table workflow for HCDS

Two-tier sample budget:

- **Wide table** (accuracy, latency, output length, finish_reason): 500–8500 questions on oMLX (already done).
- **Deep table** (entropy + paraphrase + perturbation Δ + MI): a 50–100-question slice on torch, plus matched paraphrase and perturbation runs. HCDS uses the deep table.

Sketch:

```bash
# 1. Build fixtures once.
python3 scripts/build_perturbations.py --num-samples 50 \
    --out-dir data/variants/gsm8k_test_first50_perturb
python3 scripts/build_paraphrases.py --num-samples 50 --num-paraphrases 2 \
    --out-dir data/variants/gsm8k_test_first50_para

# 2. Canonical baselines (use neutral_strict for HCDS purity).
AJAR_BACKEND=omlx AJAR_MODELS=instruct,thinking \
AJAR_PROMPTS=explicit_cot,explicit_no_cot,neutral_strict \
AJAR_NUM_SAMPLES=50 AJAR_OUTPUT_DIR=outputs/<run-id>_baseline \
python3 scripts/run_qwen3_gsm8k_mi.py

# 3. Variant baselines.
GSM8K_JSONL=data/variants/gsm8k_test_first50_para/variants.jsonl \
AJAR_NUM_SAMPLES=150 AJAR_OUTPUT_DIR=outputs/<run-id>_para ...
GSM8K_JSONL=data/variants/gsm8k_test_first50_perturb/variants.jsonl \
AJAR_NUM_SAMPLES=100 AJAR_OUTPUT_DIR=outputs/<run-id>_perturb ...

# 4. Torch MI slice for entropy + interventions. Set
#    AJAR_SAVE_FULL_PROBE_ATTENTION=0 unless you need raw attention tensors.
AJAR_BACKEND=torch AJAR_RUN_MI=1 AJAR_NUM_SAMPLES=50 \
    AJAR_SAVE_FULL_PROBE_ATTENTION=0 \
    AJAR_OUTPUT_DIR=outputs/<run-id>_mech ...

# 5. Aggregate.
python3 scripts/build_task6_table.py \
    --baseline-dir outputs/<run-id>_baseline \
    --paraphrase-dir outputs/<run-id>_para \
    --paraphrase-index data/variants/gsm8k_test_first50_para/index.csv \
    --perturbation-dir outputs/<run-id>_perturb \
    --perturbation-index data/variants/gsm8k_test_first50_perturb/index.csv \
    --mi-dir outputs/<run-id>_mech \
    --out results/task6_full_table.csv
python3 scripts/aggregate_anchor_sensitivity.py \
    --mi-dir outputs/<run-id>_mech \
    --out results/task10_anchor_sensitivity.csv
```

Use `AJAR_PROMPTS=neutral_strict` for any HCDS computation: it matches the Neutral prompt as written in the proposal verbatim. The original `neutral` prompt remains supported for back-compat with existing 500-sample runs but contains directives ("Answer the question directly", `\boxed{}`) that bias the condition.

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
