#!/usr/bin/env bash
# Orchestrates the Task 6 deep-table run end to end:
#   1) build paraphrase + perturbation fixtures from canonical GSM8K
#   2) canonical oMLX baselines (3 prompts x 2 models)
#   3) paraphrase oMLX baselines
#   4) perturbation oMLX baselines
#   5) torch MI slice (entropy + interventions) on the same questions
#   6) aggregate the Task 6 table and Task 10 anchor sensitivity
#
# Each step is idempotent thanks to runner-level resume, so if any step is
# interrupted, re-running this script picks up where it left off.
#
# Usage:
#   ./scripts/run_deep_table.sh [num_samples]
# Defaults to 50 samples. Override with the first arg, eg. 100.

set -euo pipefail

NUM_SAMPLES="${NUM_SAMPLES:-${1:-50}}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y-%m-%d_%H%M)}"
RUN_TAG="gsm8k${NUM_SAMPLES}_qwen3-4b"

# Output layout: every artifact for this deep-table run lives under one
# parent dir so it is easy to ship as a Drive bundle later.
RUN_ROOT="${RUN_ROOT:-outputs/${RUN_STAMP}_${RUN_TAG}_deep-table}"
FIXTURE_ROOT="${FIXTURE_ROOT:-data/variants/${RUN_TAG}_deep-table}"
RESULTS_ROOT="${RESULTS_ROOT:-results/runs/${RUN_STAMP}_${RUN_TAG}_deep-table}"

NUM_PARAPHRASES="${NUM_PARAPHRASES:-2}"

# Behavioral / wide caps. These match the existing baseline rerun and let us
# compare apples to apples with the 500-sample run from earlier.
BEHAVIORAL_MAX_TOKENS="${BEHAVIORAL_MAX_TOKENS:-1024}"
OMLX_CONCURRENCY="${OMLX_CONCURRENCY:-4}"

# Mechanistic caps. Thinking model frequently truncates at 1024; 1536 catches
# most of them without blowing up MPS memory. ANALYSIS_MAX_SEQ_LEN must
# accommodate prompt + max_new_tokens or the runner silently skips MI on
# long Thinking outputs.
MECH_MAX_TOKENS="${MECH_MAX_TOKENS:-1536}"
MECH_ANALYSIS_MAX_SEQ_LEN="${MECH_ANALYSIS_MAX_SEQ_LEN:-4096}"
# Interventions only need to reach the answer, not regenerate full reasoning.
# A 384-token cap keeps Thinking-class interventions tractable; on Instruct
# the full answer usually appears in <100 tokens anyway.
MECH_INTERVENTION_MAX_TOKENS="${MECH_INTERVENTION_MAX_TOKENS:-384}"
# Intervention scope: 2 modes (one residual, one attention) x 2 anchor steps +
# 1 control step = 6 interventions per baseline, vs the 20 the runner defaults
# to. Anchor-vs-control sensitivity is preserved; we just stop sweeping every
# combination of zero/scale within each stream.
MECH_INTERVENTIONS="${MECH_INTERVENTIONS:-residual_zero,attention_zero}"
MECH_TOP_ANCHOR_STEPS="${MECH_TOP_ANCHOR_STEPS:-2}"
MECH_NUM_CONTROL_STEPS="${MECH_NUM_CONTROL_STEPS:-1}"
MECH_TOP_ANCHOR_LAYERS="${MECH_TOP_ANCHOR_LAYERS:-4}"

# HCDS-purity prompt set. The legacy 'neutral' prompt is left out because it
# leaks a \boxed{} directive that contaminates the Neutral condition.
PROMPT_SET="${PROMPT_SET:-explicit_cot,explicit_no_cot,neutral_strict}"
MODEL_SET="${MODEL_SET:-instruct,thinking}"

mkdir -p "${RUN_ROOT}" "${FIXTURE_ROOT}" "${RESULTS_ROOT}"
echo "[deep-table] num_samples=${NUM_SAMPLES} run_root=${RUN_ROOT}"
echo "[deep-table] fixtures=${FIXTURE_ROOT} results=${RESULTS_ROOT}"

# Step 1: fixtures. Paraphrases call the oMLX server (deterministic, ~5min for 50).
# Set SKIP_PARAPHRASE=1 / SKIP_PERTURBATION=1 to skip those phases — needed
# for StrategyQA where the existing fixture builders are GSM8K-specific
# (math distractors, numeric-preservation paraphrasing).
SKIP_PARAPHRASE="${SKIP_PARAPHRASE:-0}"
SKIP_PERTURBATION="${SKIP_PERTURBATION:-0}"
PARA_DIR="${FIXTURE_ROOT}/paraphrase"
PERT_DIR="${FIXTURE_ROOT}/perturbation"

if [[ "${SKIP_PARAPHRASE}" == "1" ]]; then
    echo "[deep-table] SKIP_PARAPHRASE=1; not building paraphrase fixture."
elif [[ ! -f "${PARA_DIR}/variants.jsonl" ]]; then
    echo "[deep-table] building ${NUM_SAMPLES}-question paraphrase fixture..."
    python3 scripts/build_paraphrases.py \
        --num-samples "${NUM_SAMPLES}" \
        --num-paraphrases "${NUM_PARAPHRASES}" \
        --out-dir "${PARA_DIR}"
else
    echo "[deep-table] paraphrase fixture already exists; skipping."
fi

if [[ "${SKIP_PERTURBATION}" == "1" ]]; then
    echo "[deep-table] SKIP_PERTURBATION=1; not building perturbation fixture."
elif [[ ! -f "${PERT_DIR}/variants.jsonl" ]]; then
    echo "[deep-table] building ${NUM_SAMPLES}-question perturbation fixture..."
    python3 scripts/build_perturbations.py \
        --num-samples "${NUM_SAMPLES}" \
        --out-dir "${PERT_DIR}"
else
    echo "[deep-table] perturbation fixture already exists; skipping."
fi

# Step 2: canonical baselines.
BASELINE_DIR="${RUN_ROOT}/baseline"
echo "[deep-table] running canonical baselines via oMLX..."
AJAR_BACKEND=omlx \
AJAR_MODELS="${MODEL_SET}" \
AJAR_PROMPTS="${PROMPT_SET}" \
AJAR_NUM_SAMPLES="${NUM_SAMPLES}" \
AJAR_MAX_NEW_TOKENS="${BEHAVIORAL_MAX_TOKENS}" \
AJAR_OMLX_CONCURRENCY="${OMLX_CONCURRENCY}" \
AJAR_OUTPUT_DIR="${BASELINE_DIR}" \
    python3 scripts/run_qwen3_gsm8k_mi.py

# Step 3: paraphrase variants. Variant fixture has originals + K paraphrases
# per question, so pass the row count, not the canonical question count.
PARA_RUN_DIR="${RUN_ROOT}/paraphrase"
if [[ "${SKIP_PARAPHRASE}" == "1" ]]; then
    echo "[deep-table] SKIP_PARAPHRASE=1; not running paraphrase phase."
else
    PARA_TOTAL_ROWS=$(($(wc -l < "${PARA_DIR}/variants.jsonl")))
    echo "[deep-table] running ${PARA_TOTAL_ROWS}-row paraphrase fixture via oMLX..."
    AJAR_BACKEND=omlx \
    AJAR_MODELS="${MODEL_SET}" \
    AJAR_PROMPTS="${PROMPT_SET}" \
    AJAR_NUM_SAMPLES="${PARA_TOTAL_ROWS}" \
    AJAR_MAX_NEW_TOKENS="${BEHAVIORAL_MAX_TOKENS}" \
    AJAR_OMLX_CONCURRENCY="${OMLX_CONCURRENCY}" \
    AJAR_OUTPUT_DIR="${PARA_RUN_DIR}" \
    GSM8K_JSONL="${PARA_DIR}/variants.jsonl" \
        python3 scripts/run_qwen3_gsm8k_mi.py
fi

# Step 4: perturbation variants.
PERT_RUN_DIR="${RUN_ROOT}/perturbation"
if [[ "${SKIP_PERTURBATION}" == "1" ]]; then
    echo "[deep-table] SKIP_PERTURBATION=1; not running perturbation phase."
else
    PERT_TOTAL_ROWS=$(($(wc -l < "${PERT_DIR}/variants.jsonl")))
    echo "[deep-table] running ${PERT_TOTAL_ROWS}-row perturbation fixture via oMLX..."
    AJAR_BACKEND=omlx \
    AJAR_MODELS="${MODEL_SET}" \
    AJAR_PROMPTS="${PROMPT_SET}" \
    AJAR_NUM_SAMPLES="${PERT_TOTAL_ROWS}" \
    AJAR_MAX_NEW_TOKENS="${BEHAVIORAL_MAX_TOKENS}" \
    AJAR_OMLX_CONCURRENCY="${OMLX_CONCURRENCY}" \
    AJAR_OUTPUT_DIR="${PERT_RUN_DIR}" \
    GSM8K_JSONL="${PERT_DIR}/variants.jsonl" \
        python3 scripts/run_qwen3_gsm8k_mi.py
fi

# Step 5: torch MI slice on the same canonical questions. We run one model at
# a time so the worker doesn't thrash between Instruct and Thinking weights.
# Python's mp.Queue is FIFO per-process but interleaves cross-process puts,
# so a single combined invocation sees worker-queued interventions race
# ahead of main-queued thinking baselines, causing dozens of model swaps.
# Splitting per-model + writing into the same mech/ dir gives us one swap
# total: instruct loaded once, then thinking loaded once.
MECH_DIR="${RUN_ROOT}/mech"
if [[ "${SKIP_MECH}" == "1" ]]; then
    echo "[deep-table] SKIP_MECH=1; skipping torch MI phase (5-feature run)."
    _mech_models=()
else
    IFS=',' read -ra _mech_models <<< "${MODEL_SET}"
fi
# Guard the expansion: macOS bash 3.2 errors on "${arr[@]}" for an empty array
# under `set -u`, so only enter the loop when there are models to run.
for _model in ${_mech_models[@]+"${_mech_models[@]}"}; do
    echo "[deep-table] running torch MI on model='${_model}' (${NUM_SAMPLES} questions)..."
    AJAR_BACKEND=torch \
    AJAR_MODELS="${_model}" \
    AJAR_PROMPTS="${PROMPT_SET}" \
    AJAR_NUM_SAMPLES="${NUM_SAMPLES}" \
    AJAR_MAX_NEW_TOKENS="${MECH_MAX_TOKENS}" \
    AJAR_INTERVENTION_MAX_NEW_TOKENS="${MECH_INTERVENTION_MAX_TOKENS}" \
    AJAR_INTERVENTIONS="${MECH_INTERVENTIONS}" \
    AJAR_TOP_ANCHOR_STEPS="${MECH_TOP_ANCHOR_STEPS}" \
    AJAR_NUM_CONTROL_STEPS="${MECH_NUM_CONTROL_STEPS}" \
    AJAR_TOP_ANCHOR_LAYERS="${MECH_TOP_ANCHOR_LAYERS}" \
    AJAR_MAX_ANSWER_PROBES="${MECH_MAX_ANSWER_PROBES:-64}" \
    AJAR_RUN_MI=1 \
    AJAR_DTYPE=auto \
    AJAR_SAVE_FULL_PROBE_ATTENTION=0 \
    AJAR_ANALYSIS_MAX_SEQ_LEN="${MECH_ANALYSIS_MAX_SEQ_LEN}" \
    AJAR_OUTPUT_DIR="${MECH_DIR}" \
        python3 scripts/run_qwen3_gsm8k_mi.py
done

# Step 6: aggregate.
echo "[deep-table] aggregating Task 6 table..."
AGG_ARGS=(--baseline-dir "${BASELINE_DIR}" --mi-dir "${MECH_DIR}" --out "${RESULTS_ROOT}/task6_table.csv")
if [[ "${SKIP_PARAPHRASE}" != "1" ]]; then
    AGG_ARGS+=(--paraphrase-dir "${PARA_RUN_DIR}" --paraphrase-index "${PARA_DIR}/index.csv")
fi
if [[ "${SKIP_PERTURBATION}" != "1" ]]; then
    AGG_ARGS+=(--perturbation-dir "${PERT_RUN_DIR}" --perturbation-index "${PERT_DIR}/index.csv")
fi
python3 scripts/build_task6_table.py "${AGG_ARGS[@]}"

if [[ "${SKIP_MECH}" == "1" ]]; then
    echo "[deep-table] SKIP_MECH=1; skipping Task 10 anchor sensitivity (no MI)."
else
    echo "[deep-table] aggregating Task 10 anchor sensitivity..."
    python3 scripts/aggregate_anchor_sensitivity.py \
        --mi-dir "${MECH_DIR}" \
        --out "${RESULTS_ROOT}/task10_anchor_sensitivity.csv"
fi

echo "[deep-table] done. results in ${RESULTS_ROOT}/"
