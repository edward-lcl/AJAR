#!/usr/bin/env bash
# Phase A re-run (Gemma 5-feature) after the MODEL_GPU_ALLOCATIONS['gemma'] fix.
# Waits for any running finish_chain.sh (Phase B / Thinking torch) to exit first,
# so two torch runs never contend for the GPU. Safe to launch and walk away.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=python3
STAMP="$(date +%Y-%m-%d_%H%M)"
log(){ echo "[phaseA $(date +%H:%M:%S)] $*"; }

log "waiting for any running finish_chain.sh / Thinking torch to finish..."
while pgrep -f 'finish_chain.sh' >/dev/null 2>&1; do sleep 30; done
# brief settle so the prior run frees model weights
sleep 10

GEMMA_RUN="outputs/2026-06-10_2131_gsm8k50_gemma3-4b_crossfamily"
GEMMA_RES="results/runs/2026-06-10_2131_gsm8k50_gemma3-4b_crossfamily"
FIX="data/variants/gsm8k50_qwen3-4b_deep-table"
GEMMA_TORCH="${GEMMA_RUN}/torch_entropy"
SUM="${GEMMA_RES}/PHASE_A_SUMMARY_${STAMP}.md"

log "full 50-sample torch baseline (entropy) for gemma"
env AJAR_BACKEND=torch AJAR_DTYPE=auto AJAR_SAVE_FULL_PROBE_ATTENTION=0 \
    AJAR_ANALYSIS_MAX_SEQ_LEN=4096 AJAR_MODELS=gemma \
    AJAR_PROMPTS=explicit_cot,explicit_no_cot,neutral_strict \
    AJAR_NUM_SAMPLES=50 AJAR_MAX_NEW_TOKENS=1024 \
    AJAR_OUTPUT_DIR="${GEMMA_TORCH}" $PY scripts/run_qwen3_gsm8k_mi.py \
    > logs/phaseA_torch_${STAMP}.log 2>&1
RC=$?
if [[ $RC -ne 0 ]]; then
    log "torch run failed rc=$RC; see logs/phaseA_torch_${STAMP}.log"
    echo "Phase A torch FAILED rc=$RC — 3-feature HCDS (+1.67) stands." > "$SUM"
    exit $RC
fi

log "re-aggregate with entropy + compute 5-feature HCDS"
$PY scripts/build_task6_table.py \
    --baseline-dir "${GEMMA_RUN}/baseline" --mi-dir "${GEMMA_TORCH}" \
    --paraphrase-dir "${GEMMA_RUN}/paraphrase" --paraphrase-index "${FIX}/paraphrase/index.csv" \
    --perturbation-dir "${GEMMA_RUN}/perturbation" --perturbation-index "${FIX}/perturbation/index.csv" \
    --out "${GEMMA_RES}/task6_table_5feat.csv" > logs/phaseA_agg_${STAMP}.log 2>&1
$PY scripts/compute_hcds.py --task6-csv "${GEMMA_RES}/task6_table_5feat.csv" \
    --out-dir "${GEMMA_RES}/hcds" \
    --exclude-features "mechanistic_intervention_delta_accuracy" \
    --label feat5 > logs/phaseA_hcds_${STAMP}.log 2>&1

{ echo "# Phase A — Gemma 5-feature HCDS (${STAMP})"; echo '```';
  cat "${GEMMA_RES}/hcds/hcds_summary_feat5.csv"; echo '```'; } > "$SUM"
log "DONE. summary: $SUM"
