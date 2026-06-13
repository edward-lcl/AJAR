#!/usr/bin/env bash
# Cross-family replication on Gemma (PI/reviewer ask: a second model family).
#
# Gemma is instruction-tuned with no reasoning variant, so this replicates the
# *validated Instruct* HCDS case on a different architecture/tokenizer/pretrain.
# oMLX is baseline-only, so this produces the 5 behavioral/linguistic features
# (latency, entropy mean, entropy slope, paraphrase consistency, perturbation
# sensitivity). The 6th feature (torch anchor intervention) is Qwen-specific and
# is intentionally skipped (SKIP_MECH=1) -- matches the spec's "5 of 6" scaled run.
#
# Fixtures are model-independent, so we REUSE the committed GSM8K n=50 deep-table
# paraphrase/perturbation fixtures rather than regenerating them.
#
# Runs gently (OMLX_CONCURRENCY=2) so it can coexist with other GPU jobs.
set -euo pipefail

cd "$(dirname "$0")/.."

# Faithful dense small Gemma at the project's scale (Qwen3-4B analog): dense 4B,
# instruction-tuned, MLX. 4-bit chosen over 8-bit for throughput; the
# cross-family test is about generality, not quantization-matching (footnote it).
export AJAR_OMLX_MODEL_GEMMA="${AJAR_OMLX_MODEL_GEMMA:-gemma-3-4b-it-4bit}"

NUM_SAMPLES="${NUM_SAMPLES:-50}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y-%m-%d_%H%M)}"
TAG="gsm8k${NUM_SAMPLES}_gemma3-4b"

# Reuse the committed Qwen GSM8K fixtures (paraphrase + perturbation are
# answer-preserving and model-independent). The deep-table build steps detect
# the existing variants.jsonl and skip rebuilding.
export FIXTURE_ROOT="${FIXTURE_ROOT:-data/variants/gsm8k50_qwen3-4b_deep-table}"

# Gemma-labelled output/result roots so nothing collides with the Qwen runs.
export RUN_ROOT="${RUN_ROOT:-outputs/${RUN_STAMP}_${TAG}_crossfamily}"
export RESULTS_ROOT="${RESULTS_ROOT:-results/runs/${RUN_STAMP}_${TAG}_crossfamily}"

export MODEL_SET="gemma"
export SKIP_MECH=1
export OMLX_CONCURRENCY="${OMLX_CONCURRENCY:-2}"

echo "[gemma-xfam] num_samples=${NUM_SAMPLES} model=gemma fixtures=${FIXTURE_ROOT}"
echo "[gemma-xfam] results -> ${RESULTS_ROOT}"

exec bash scripts/run_deep_table.sh "${NUM_SAMPLES}"
