#!/usr/bin/env bash
# Rerun the factual negative-control family at BEHAVIORAL_MAX_TOKENS=4096.
#
# The original run (outputs/neg_control_factual_full/) used 1024 tokens and
# ~25% of Thinking responses were truncated before finishing self-reflection,
# causing parser fallback. This script reruns to a fresh output dir so the
# original data is preserved.
#
# Only the factual family needs this: arithmetic had only 2 truncations/150
# rows and is clean for calibration purposes.
#
# Usage:
#   ./scripts/run_negcontrol_4096.sh
#
# Runtime estimate (M5 Pro): ~45-75 min (baseline + para + pert; mech is fast
# on these short-answer questions).
set -euo pipefail

FAMILY="factual"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
NUM_PARAPHRASES="${NUM_PARAPHRASES:-2}"

BEHAVIORAL_MAX_TOKENS=4096
OMLX_CONCURRENCY="${OMLX_CONCURRENCY:-4}"
MECH_MAX_TOKENS="${MECH_MAX_TOKENS:-1536}"
MECH_ANALYSIS_MAX_SEQ_LEN="${MECH_ANALYSIS_MAX_SEQ_LEN:-4096}"
MECH_INTERVENTION_MAX_TOKENS="${MECH_INTERVENTION_MAX_TOKENS:-384}"
MECH_INTERVENTIONS="${MECH_INTERVENTIONS:-residual_zero,attention_zero}"
MECH_TOP_ANCHOR_STEPS="${MECH_TOP_ANCHOR_STEPS:-2}"
MECH_NUM_CONTROL_STEPS="${MECH_NUM_CONTROL_STEPS:-1}"
MECH_TOP_ANCHOR_LAYERS="${MECH_TOP_ANCHOR_LAYERS:-4}"

PROMPT_SET="${PROMPT_SET:-explicit_cot,explicit_no_cot,neutral_strict}"
MODEL_SET="${MODEL_SET:-instruct,thinking}"

PY="${PY:-python3}"

FIXTURE="data/fixtures/negative_control/${FAMILY}_${NUM_SAMPLES}.jsonl"
if [[ ! -f "${FIXTURE}" ]]; then
    echo "[negcontrol-4096] MISSING fixture ${FIXTURE}" >&2
    exit 1
fi

# Fresh output dir — does NOT overwrite the original 1024-token run
RUN_ROOT="outputs/neg_control_${FAMILY}_4096"
FIXTURE_ROOT="data/variants/neg_control_${FAMILY}"
RESULTS_ROOT="results/runs/negative_control/${FAMILY}_4096"
PARA_DIR="${FIXTURE_ROOT}/paraphrase"
PERT_DIR="${FIXTURE_ROOT}/perturbation"
mkdir -p "${RUN_ROOT}" "${RESULTS_ROOT}"

echo "=============================================================="
echo "[negcontrol-4096] FAMILY=${FAMILY} n=${NUM_SAMPLES} max_tokens=${BEHAVIORAL_MAX_TOKENS}"
echo "[negcontrol-4096] Output: ${RESULTS_ROOT}"
echo "=============================================================="

# Paraphrase + perturbation fixtures are shared with the original run (already built)
if [[ ! -f "${PARA_DIR}/variants.jsonl" ]]; then
    echo "[negcontrol-4096] building paraphrase fixture..."
    ${PY} scripts/build_paraphrases.py \
        --num-samples "${NUM_SAMPLES}" \
        --num-paraphrases "${NUM_PARAPHRASES}" \
        --input-jsonl "${FIXTURE}" \
        --general-prompt \
        --out-dir "${PARA_DIR}"
else
    echo "[negcontrol-4096] paraphrase fixture exists; reusing."
fi
if [[ ! -f "${PERT_DIR}/variants.jsonl" ]]; then
    echo "[negcontrol-4096] building perturbation fixture..."
    ${PY} scripts/build_perturbations.py \
        --num-samples "${NUM_SAMPLES}" \
        --input-jsonl "${FIXTURE}" \
        --out-dir "${PERT_DIR}"
else
    echo "[negcontrol-4096] perturbation fixture exists; reusing."
fi

# Step 1: canonical baselines (fresh — no resume, new dir, 4096-token budget)
BASELINE_DIR="${RUN_ROOT}/baseline"
echo "[negcontrol-4096] baseline generation (max_tokens=${BEHAVIORAL_MAX_TOKENS})..."
AJAR_BACKEND=omlx \
AJAR_MODELS="${MODEL_SET}" AJAR_PROMPTS="${PROMPT_SET}" \
AJAR_NUM_SAMPLES="${NUM_SAMPLES}" \
AJAR_MAX_NEW_TOKENS="${BEHAVIORAL_MAX_TOKENS}" \
AJAR_OMLX_CONCURRENCY="${OMLX_CONCURRENCY}" \
AJAR_OUTPUT_DIR="${BASELINE_DIR}" \
GSM8K_JSONL="${FIXTURE}" \
    ${PY} scripts/run_qwen3_gsm8k_mi.py

# Step 2: paraphrase variants
PARA_RUN_DIR="${RUN_ROOT}/paraphrase"
PARA_TOTAL_ROWS=$(($(wc -l < "${PARA_DIR}/variants.jsonl")))
echo "[negcontrol-4096] paraphrase phase (${PARA_TOTAL_ROWS} rows)..."
AJAR_BACKEND=omlx \
AJAR_MODELS="${MODEL_SET}" AJAR_PROMPTS="${PROMPT_SET}" \
AJAR_NUM_SAMPLES="${PARA_TOTAL_ROWS}" \
AJAR_MAX_NEW_TOKENS="${BEHAVIORAL_MAX_TOKENS}" \
AJAR_OMLX_CONCURRENCY="${OMLX_CONCURRENCY}" \
AJAR_OUTPUT_DIR="${PARA_RUN_DIR}" \
GSM8K_JSONL="${PARA_DIR}/variants.jsonl" \
    ${PY} scripts/run_qwen3_gsm8k_mi.py

# Step 3: perturbation variants
PERT_RUN_DIR="${RUN_ROOT}/perturbation"
PERT_TOTAL_ROWS=$(($(wc -l < "${PERT_DIR}/variants.jsonl")))
echo "[negcontrol-4096] perturbation phase (${PERT_TOTAL_ROWS} rows)..."
AJAR_BACKEND=omlx \
AJAR_MODELS="${MODEL_SET}" AJAR_PROMPTS="${PROMPT_SET}" \
AJAR_NUM_SAMPLES="${PERT_TOTAL_ROWS}" \
AJAR_MAX_NEW_TOKENS="${BEHAVIORAL_MAX_TOKENS}" \
AJAR_OMLX_CONCURRENCY="${OMLX_CONCURRENCY}" \
AJAR_OUTPUT_DIR="${PERT_RUN_DIR}" \
GSM8K_JSONL="${PERT_DIR}/variants.jsonl" \
    ${PY} scripts/run_qwen3_gsm8k_mi.py

# Step 4: torch MI (reuse mech outputs from original run — mech uses 1536-token
# budget which is unaffected, and these short-answer questions rarely truncate there)
MECH_DIR="outputs/neg_control_${FAMILY}_full/mech"
if [[ ! -d "${MECH_DIR}" ]]; then
    echo "[negcontrol-4096] mech dir not found at ${MECH_DIR}; running fresh..."
    MECH_DIR="${RUN_ROOT}/mech"
    IFS=',' read -ra _mech_models <<< "${MODEL_SET}"
    for _model in "${_mech_models[@]}"; do
        AJAR_BACKEND=torch AJAR_RESUME=1 \
        AJAR_MODELS="${_model}" AJAR_PROMPTS="${PROMPT_SET}" \
        AJAR_NUM_SAMPLES="${NUM_SAMPLES}" \
        AJAR_MAX_NEW_TOKENS="${MECH_MAX_TOKENS}" \
        AJAR_INTERVENTION_MAX_NEW_TOKENS="${MECH_INTERVENTION_MAX_TOKENS}" \
        AJAR_INTERVENTIONS="${MECH_INTERVENTIONS}" \
        AJAR_TOP_ANCHOR_STEPS="${MECH_TOP_ANCHOR_STEPS}" \
        AJAR_NUM_CONTROL_STEPS="${MECH_NUM_CONTROL_STEPS}" \
        AJAR_TOP_ANCHOR_LAYERS="${MECH_TOP_ANCHOR_LAYERS}" \
        AJAR_MAX_ANSWER_PROBES="${MECH_MAX_ANSWER_PROBES:-64}" \
        AJAR_RUN_MI=1 AJAR_DTYPE=auto AJAR_SAVE_FULL_PROBE_ATTENTION=0 \
        AJAR_ANALYSIS_MAX_SEQ_LEN="${MECH_ANALYSIS_MAX_SEQ_LEN}" \
        AJAR_OUTPUT_DIR="${MECH_DIR}" \
        GSM8K_JSONL="${FIXTURE}" \
            ${PY} scripts/run_qwen3_gsm8k_mi.py
    done
else
    echo "[negcontrol-4096] reusing existing mech outputs from original run."
fi

# Step 5: aggregate and recompute HCDS
echo "[negcontrol-4096] building task6 table..."
${PY} scripts/build_task6_table.py \
    --baseline-dir "${BASELINE_DIR}" \
    --mi-dir "${MECH_DIR}" \
    --paraphrase-dir "${PARA_RUN_DIR}" --paraphrase-index "${PARA_DIR}/index.csv" \
    --perturbation-dir "${PERT_RUN_DIR}" --perturbation-index "${PERT_DIR}/index.csv" \
    --out "${RESULTS_ROOT}/task6_table.csv"

echo "[negcontrol-4096] computing HCDS..."
${PY} scripts/compute_hcds.py \
    --task6-csv "${RESULTS_ROOT}/task6_table.csv" \
    --out-dir "${RESULTS_ROOT}"

echo "=============================================================="
echo "[negcontrol-4096] DONE -> ${RESULTS_ROOT}/hcds_summary.csv"
echo "Compare against: results/runs/negative_control/factual/hcds_summary.csv"
echo "=============================================================="
