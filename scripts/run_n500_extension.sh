#!/usr/bin/env bash
# Extend HCDS coverage from n=50 to n=500 by computing the
# behavioural-feature columns (paraphrase + perturbation) on the
# remaining 450 GSM8K questions. The wide baseline at n=500 (already
# committed) provides accuracy / latency / entropy. Mechanistic
# features stay at n=50 (full extension would be 50+ hours of GPU
# time, infeasible in our window).
#
# What this produces:
#   - data/variants/gsm8k500_qwen3-4b_deep-table/paraphrase/   (n=500)
#   - data/variants/gsm8k500_qwen3-4b_deep-table/perturbation/ (n=500)
#   - outputs/<RUN_STAMP>_gsm8k500_qwen3-4b_deep-table-ext/
#         baseline/      (symlink to wide n=500 baseline)
#         paraphrase/    (n=500 paraphrase runs)
#         perturbation/  (n=500 perturbation runs)
#   - results/runs/<RUN_STAMP>_gsm8k500_qwen3-4b_deep-table-ext/
#         task6_table.csv         (n=500 wide table, 5 features)
#         hcds_per_question.csv   (HCDS at n=500, 5-feature)
#         hcds_summary.csv        (n=500 summary stats)
#
# Compute estimate: ~5-7 hr on M5 Pro (oMLX is fast; this run is
# behavioural-only).
#
# Usage:
#   ./scripts/run_n500_extension.sh           # n=500
#   NUM_SAMPLES=200 ./scripts/run_n500_extension.sh   # smaller scoped extension

set -uo pipefail   # not -e so a single failure doesn't abort everything

NUM_SAMPLES="${NUM_SAMPLES:-500}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y-%m-%d_%H%M)}"
RUN_TAG="gsm8k${NUM_SAMPLES}_qwen3-4b"

RUN_ROOT="${RUN_ROOT:-outputs/${RUN_STAMP}_${RUN_TAG}_deep-table-ext}"
FIXTURE_ROOT="${FIXTURE_ROOT:-data/variants/${RUN_TAG}_deep-table}"
RESULTS_ROOT="${RESULTS_ROOT:-results/runs/${RUN_STAMP}_${RUN_TAG}_deep-table-ext}"

NUM_PARAPHRASES="${NUM_PARAPHRASES:-2}"
BEHAVIORAL_MAX_TOKENS="${BEHAVIORAL_MAX_TOKENS:-1024}"
OMLX_CONCURRENCY="${OMLX_CONCURRENCY:-4}"
PROMPT_SET="${PROMPT_SET:-explicit_cot,explicit_no_cot,neutral_strict}"
MODEL_SET="${MODEL_SET:-instruct,thinking}"

mkdir -p "${RUN_ROOT}" "${FIXTURE_ROOT}" "${RESULTS_ROOT}"
echo "[n500-ext] num_samples=${NUM_SAMPLES} run_root=${RUN_ROOT}"

# -----------------------------------------------------------------------
# Step 1 — fixtures.
# -----------------------------------------------------------------------
PARA_DIR="${FIXTURE_ROOT}/paraphrase"
PERT_DIR="${FIXTURE_ROOT}/perturbation"

if [[ ! -f "${PARA_DIR}/variants.jsonl" ]]; then
    echo "[n500-ext] building n=${NUM_SAMPLES} paraphrase fixture..."
    python3 scripts/build_paraphrases.py \
        --num-samples "${NUM_SAMPLES}" \
        --num-paraphrases "${NUM_PARAPHRASES}" \
        --out-dir "${PARA_DIR}"
else
    echo "[n500-ext] paraphrase fixture exists; skipping rebuild."
fi

if [[ ! -f "${PERT_DIR}/variants.jsonl" ]]; then
    echo "[n500-ext] building n=${NUM_SAMPLES} perturbation fixture..."
    python3 scripts/build_perturbations.py \
        --num-samples "${NUM_SAMPLES}" \
        --out-dir "${PERT_DIR}"
else
    echo "[n500-ext] perturbation fixture exists; skipping rebuild."
fi

# -----------------------------------------------------------------------
# Step 2 — baseline. Already exists from the wide-baseline run.
# -----------------------------------------------------------------------
WIDE_BASELINE_DIR="outputs/2026-05-04_1856_gsm8k500_qwen3-4b_omlx_baseline-neutral-strict-1024"
BASELINE_DIR="${RUN_ROOT}/baseline"
if [[ ! -e "${BASELINE_DIR}" ]]; then
    echo "[n500-ext] linking wide-baseline run into ${BASELINE_DIR}..."
    if [[ -d "${WIDE_BASELINE_DIR}" ]]; then
        ln -s "$(pwd)/${WIDE_BASELINE_DIR}" "${BASELINE_DIR}"
    else
        echo "[n500-ext] WARNING: wide-baseline dir not found at ${WIDE_BASELINE_DIR}."
        echo "[n500-ext] running fresh baselines..."
        AJAR_BACKEND=omlx \
        AJAR_MODELS="${MODEL_SET}" \
        AJAR_PROMPTS="${PROMPT_SET}" \
        AJAR_NUM_SAMPLES="${NUM_SAMPLES}" \
        AJAR_MAX_NEW_TOKENS="${BEHAVIORAL_MAX_TOKENS}" \
        AJAR_OMLX_CONCURRENCY="${OMLX_CONCURRENCY}" \
        AJAR_OUTPUT_DIR="${BASELINE_DIR}" \
            python3 scripts/run_qwen3_gsm8k_mi.py
    fi
fi

# -----------------------------------------------------------------------
# Step 3 — paraphrases on the full n=500 fixture.
# -----------------------------------------------------------------------
PARA_RUN_DIR="${RUN_ROOT}/paraphrase"
PARA_TOTAL_ROWS=$(($(wc -l < "${PARA_DIR}/variants.jsonl")))
echo "[n500-ext] running ${PARA_TOTAL_ROWS}-row paraphrase fixture..."
AJAR_BACKEND=omlx \
AJAR_MODELS="${MODEL_SET}" \
AJAR_PROMPTS="${PROMPT_SET}" \
AJAR_NUM_SAMPLES="${PARA_TOTAL_ROWS}" \
AJAR_MAX_NEW_TOKENS="${BEHAVIORAL_MAX_TOKENS}" \
AJAR_OMLX_CONCURRENCY="${OMLX_CONCURRENCY}" \
AJAR_OUTPUT_DIR="${PARA_RUN_DIR}" \
GSM8K_JSONL="${PARA_DIR}/variants.jsonl" \
    python3 scripts/run_qwen3_gsm8k_mi.py

# -----------------------------------------------------------------------
# Step 4 — perturbations on the full n=500 fixture.
# -----------------------------------------------------------------------
PERT_RUN_DIR="${RUN_ROOT}/perturbation"
PERT_TOTAL_ROWS=$(($(wc -l < "${PERT_DIR}/variants.jsonl")))
echo "[n500-ext] running ${PERT_TOTAL_ROWS}-row perturbation fixture..."
AJAR_BACKEND=omlx \
AJAR_MODELS="${MODEL_SET}" \
AJAR_PROMPTS="${PROMPT_SET}" \
AJAR_NUM_SAMPLES="${PERT_TOTAL_ROWS}" \
AJAR_MAX_NEW_TOKENS="${BEHAVIORAL_MAX_TOKENS}" \
AJAR_OMLX_CONCURRENCY="${OMLX_CONCURRENCY}" \
AJAR_OUTPUT_DIR="${PERT_RUN_DIR}" \
GSM8K_JSONL="${PERT_DIR}/variants.jsonl" \
    python3 scripts/run_qwen3_gsm8k_mi.py

# -----------------------------------------------------------------------
# Step 5 — aggregate. Pull mech from the deep-table-50 run; 5-feature
# HCDS at n=500.
# -----------------------------------------------------------------------
DEEP_TABLE_MECH="outputs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/mech"

echo "[n500-ext] aggregating Task 6 wide table at n=${NUM_SAMPLES}..."
python3 scripts/build_task6_table.py \
    --baseline-dir "${BASELINE_DIR}" \
    --paraphrase-dir "${PARA_RUN_DIR}" \
    --paraphrase-index "${PARA_DIR}/index.csv" \
    --perturbation-dir "${PERT_RUN_DIR}" \
    --perturbation-index "${PERT_DIR}/index.csv" \
    --mi-dir "${DEEP_TABLE_MECH}" \
    --out "${RESULTS_ROOT}/task6_table.csv"

echo "[n500-ext] computing HCDS at n=${NUM_SAMPLES}..."
python3 scripts/compute_hcds.py \
    --task6-csv "${RESULTS_ROOT}/task6_table.csv" \
    --out-dir "${RESULTS_ROOT}"

# Drop mech feature → 5-feature HCDS at full n=500 (matches the
# already-shipped no_mech variant but with much wider coverage).
python3 scripts/compute_hcds.py \
    --task6-csv "${RESULTS_ROOT}/task6_table.csv" \
    --out-dir "${RESULTS_ROOT}" \
    --label no_mech \
    --exclude-features mechanistic_intervention_delta_accuracy

echo "[n500-ext] done. results in ${RESULTS_ROOT}/"
