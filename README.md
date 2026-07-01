# AJAR Hidden CoT Experiments

Code and result artifacts for the paper **"Detecting hidden chain-of-thought in
large language models with linguistic, behavioral, and mechanistic indicators"**
(camera-ready source: [`cameraready.tex`](cameraready.tex)).

> **Reproducing the paper?** Start with **[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)** —
> it maps every headline number and figure to the exact script and committed CSV.

This repository contains the AJAR experiment runner, the committed HCDS result
tables, and analysis summaries for the hidden chain-of-thought project. The
sections below are developer-oriented reference; the reproducibility guide is
the entry point for readers of the paper.

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

The whole flow is wrapped in `scripts/run_deep_table.sh`:

```bash
./scripts/run_deep_table.sh                  # 50 samples (default)
./scripts/run_deep_table.sh 100              # override
RUN_STAMP=2026-05-04_2232 \
    ./scripts/run_deep_table.sh              # reuse a prior run dir (resume)
```

The orchestrator builds fixtures, runs canonical/paraphrase/perturbation oMLX baselines, then the torch MI slice, then both aggregators. Each step is idempotent via runner-level resume. On Apple Silicon (M5 Pro, MPS, fp16) the full 50-sample matrix takes ~6-8 hours including the MI slice; on a single A100 80GB, ~1-2 hours (see `docs/cloud_gpu_setup.md`).

Use `AJAR_PROMPTS=neutral_strict` for HCDS: it matches the Neutral prompt as written in the proposal verbatim. The original `neutral` prompt is kept for back-compat with existing 500-sample runs but contains "Answer the question directly" + `\boxed{}` directives that bias the condition.

### Performance-relevant env vars

| Var | Default | Purpose |
|---|---|---|
| `AJAR_DTYPE` | `auto` | fp16 on MPS, bf16 on bf16-capable CUDA, fp16 on older CUDA, fp32 on CPU. |
| `AJAR_MAX_NEW_TOKENS` | 512 (torch) / 256 (omlx) | Token budget for baseline generation. |
| `AJAR_INTERVENTION_MAX_NEW_TOKENS` | 384 | Token budget for intervention continuations only. Smaller than baseline because interventions just need to reach the answer. |
| `AJAR_ANALYSIS_MAX_SEQ_LEN` | 2048 | Skip MI on samples whose full sequence exceeds this. Bump to 4096 when running Thinking with `MAX_NEW_TOKENS>=1024`. |
| `AJAR_TOP_ANCHOR_STEPS` | 3 | Anchor-step targets per baseline. |
| `AJAR_NUM_CONTROL_STEPS` | 2 | Random-step controls for the anchor−control mechanistic Δ. |
| `AJAR_INTERVENTIONS` | `residual_zero,residual_scale_0.25,attention_zero,attention_scale_0.25` | Intervention modes. |
| `AJAR_SAVE_FULL_PROBE_ATTENTION` | 0 | Off by default. On adds ~50 GB / 50 samples. |
| `AJAR_PARALLEL` | 0 | Per-GPU worker pool. Off unless you have multi-GPU CUDA. |

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

## Backends and mechanistic data

The oMLX backend is text-generation-only: it does not expose hidden states,
attentions, logits/logprobs, or forward hooks, so it fills only the baseline and
latency metrics. The **torch backend** (`AJAR_BACKEND=torch`) fills the full
six-feature table — token entropy (from logits), paraphrase consistency,
perturbation Δ-accuracy, and mechanistic intervention Δ-accuracy. The committed
deep tables under `results/runs/*deep-table*` and the Gemma cross-family run are
produced this way; all six feature columns are populated (see
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)).

Note for Apple Silicon: Qwen runs fine in fp16, but Gemma-3-4B mechanistic
interventions require `AJAR_DTYPE=float32` (fp16 yields NaN/empty output on MPS).
