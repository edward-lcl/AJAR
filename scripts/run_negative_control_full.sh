#!/usr/bin/env bash
# Full multi-feature (not latency-only) HCDS on the two negative-control
# families (arithmetic, factual), both Qwen3-4B variants, 3 prompts.
#
# Experiment §1 in docs/camera_ready_analysis_spec.md ("run first" — most
# likely to reframe the Instruct headline). Mirrors run_deep_table.sh but:
#   * runs once per family, pointing every stage at the neg-control fixture
#     via GSM8K_JSONL (the runner reads any {question, answer} JSONL);
#   * builds paraphrase/perturbation fixtures FROM that family
#     (build_*.py --input-jsonl, paraphraser --general-prompt);
#   * aggregates a per-family task6_table.csv, then computes full HCDS via
#     compute_hcds.py.
#
# Output dirs use a _full suffix so they do NOT collide with the existing
# May-8 latency-only neg-control data under outputs/neg_control_<family>/.
#
# Idempotent: runner-level resume (AJAR_RESUME=1). Safe to relaunch after OOM.
#
# Usage:
#   ./scripts/run_negative_control_full.sh                       # both families
#   FAMILIES=arithmetic ./scripts/run_negative_control_full.sh   # one family
set -euo pipefail

FAMILIES="${FAMILIES:-arithmetic factual}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
NUM_PARAPHRASES="${NUM_PARAPHRASES:-2}"

BEHAVIORAL_MAX_TOKENS="${BEHAVIORAL_MAX_TOKENS:-1024}"
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

for FAMILY in ${FAMILIES}; do
    FIXTURE="data/fixtures/negative_control/${FAMILY}_${NUM_SAMPLES}.jsonl"
    if [[ ! -f "${FIXTURE}" ]]; then
        echo "[neg-control] MISSING fixture ${FIXTURE}; skipping ${FAMILY}." >&2
        continue
    fi
    RUN_ROOT="outputs/neg_control_${FAMILY}_full"
    FIXTURE_ROOT="data/variants/neg_control_${FAMILY}"
    RESULTS_ROOT="results/runs/negative_control/${FAMILY}"
    PARA_DIR="${FIXTURE_ROOT}/paraphrase"
    PERT_DIR="${FIXTURE_ROOT}/perturbation"
    mkdir -p "${RUN_ROOT}" "${FIXTURE_ROOT}" "${RESULTS_ROOT}"

    echo "=============================================================="
    echo "[neg-control] FAMILY=${FAMILY} fixture=${FIXTURE} n=${NUM_SAMPLES}"
    echo "=============================================================="

    # --- Step 1: build paraphrase + perturbation fixtures from this family ---
    if [[ ! -f "${PARA_DIR}/variants.jsonl" ]]; then
        echo "[neg-control] building paraphrase fixture..."
        ${PY} scripts/build_paraphrases.py \
            --num-samples "${NUM_SAMPLES}" \
            --num-paraphrases "${NUM_PARAPHRASES}" \
            --input-jsonl "${FIXTURE}" \
            --general-prompt \
            --out-dir "${PARA_DIR}"
    else
        echo "[neg-control] paraphrase fixture exists; skipping."
    fi
    if [[ ! -f "${PERT_DIR}/variants.jsonl" ]]; then
        echo "[neg-control] building perturbation fixture..."
        ${PY} scripts/build_perturbations.py \
            --num-samples "${NUM_SAMPLES}" \
            --input-jsonl "${FIXTURE}" \
            --out-dir "${PERT_DIR}"
    else
        echo "[neg-control] perturbation fixture exists; skipping."
    fi

    # --- Step 2: canonical baselines (latency + entropy features) ---
    BASELINE_DIR="${RUN_ROOT}/baseline"
    echo "[neg-control] canonical baselines via oMLX..."
    AJAR_BACKEND=omlx AJAR_RESUME=1 \
    AJAR_MODELS="${MODEL_SET}" AJAR_PROMPTS="${PROMPT_SET}" \
    AJAR_NUM_SAMPLES="${NUM_SAMPLES}" \
    AJAR_MAX_NEW_TOKENS="${BEHAVIORAL_MAX_TOKENS}" \
    AJAR_OMLX_CONCURRENCY="${OMLX_CONCURRENCY}" \
    AJAR_OUTPUT_DIR="${BASELINE_DIR}" \
    GSM8K_JSONL="${FIXTURE}" \
        ${PY} scripts/run_qwen3_gsm8k_mi.py

    # --- Step 3: paraphrase variants (paraphrase_consistency feature) ---
    PARA_RUN_DIR="${RUN_ROOT}/paraphrase"
    PARA_TOTAL_ROWS=$(($(wc -l < "${PARA_DIR}/variants.jsonl")))
    echo "[neg-control] paraphrase phase (${PARA_TOTAL_ROWS} rows)..."
    AJAR_BACKEND=omlx AJAR_RESUME=1 \
    AJAR_MODELS="${MODEL_SET}" AJAR_PROMPTS="${PROMPT_SET}" \
    AJAR_NUM_SAMPLES="${PARA_TOTAL_ROWS}" \
    AJAR_MAX_NEW_TOKENS="${BEHAVIORAL_MAX_TOKENS}" \
    AJAR_OMLX_CONCURRENCY="${OMLX_CONCURRENCY}" \
    AJAR_OUTPUT_DIR="${PARA_RUN_DIR}" \
    GSM8K_JSONL="${PARA_DIR}/variants.jsonl" \
        ${PY} scripts/run_qwen3_gsm8k_mi.py

    # --- Step 4: perturbation variants (perturbation_delta feature) ---
    PERT_RUN_DIR="${RUN_ROOT}/perturbation"
    PERT_TOTAL_ROWS=$(($(wc -l < "${PERT_DIR}/variants.jsonl")))
    echo "[neg-control] perturbation phase (${PERT_TOTAL_ROWS} rows)..."
    AJAR_BACKEND=omlx AJAR_RESUME=1 \
    AJAR_MODELS="${MODEL_SET}" AJAR_PROMPTS="${PROMPT_SET}" \
    AJAR_NUM_SAMPLES="${PERT_TOTAL_ROWS}" \
    AJAR_MAX_NEW_TOKENS="${BEHAVIORAL_MAX_TOKENS}" \
    AJAR_OMLX_CONCURRENCY="${OMLX_CONCURRENCY}" \
    AJAR_OUTPUT_DIR="${PERT_RUN_DIR}" \
    GSM8K_JSONL="${PERT_DIR}/variants.jsonl" \
        ${PY} scripts/run_qwen3_gsm8k_mi.py

    # --- Step 5: torch MI slice (mechanistic_intervention_delta feature) ---
    # One model at a time to avoid weight thrashing on unified memory.
    # NOTE: single-step answers are ~1 token; anchor selection may find no
    # reasoning steps, leaving the mech feature null for those cells. That is
    # expected and handled by compute_hcds.py (ragged feature vectors).
    MECH_DIR="${RUN_ROOT}/mech"
    IFS=',' read -ra _mech_models <<< "${MODEL_SET}"
    for _model in "${_mech_models[@]}"; do
        echo "[neg-control] torch MI model='${_model}'..."
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

    # --- Step 6: aggregate + compute full HCDS ---
    echo "[neg-control] aggregating task6 table for ${FAMILY}..."
    ${PY} scripts/build_task6_table.py \
        --baseline-dir "${BASELINE_DIR}" \
        --mi-dir "${MECH_DIR}" \
        --paraphrase-dir "${PARA_RUN_DIR}" --paraphrase-index "${PARA_DIR}/index.csv" \
        --perturbation-dir "${PERT_RUN_DIR}" --perturbation-index "${PERT_DIR}/index.csv" \
        --out "${RESULTS_ROOT}/task6_table.csv"

    echo "[neg-control] computing full HCDS for ${FAMILY}..."
    ${PY} scripts/compute_hcds.py \
        --task6-csv "${RESULTS_ROOT}/task6_table.csv" \
        --out-dir "${RESULTS_ROOT}"

    echo "[neg-control] ${FAMILY} done -> ${RESULTS_ROOT}/hcds_summary.csv"
done

echo "[neg-control] ALL FAMILIES COMPLETE."
