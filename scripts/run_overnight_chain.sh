#!/usr/bin/env bash
# Chains the rest of the overnight work after stage 3 (n=500 extension)
# completes. Each stage is idempotent — safe to re-run if interrupted.
#
# Sequence:
#   wait — for stage 3 (run_n500_extension) to exit
#   1) StrategyQA-50 thinking-model resume (instruct already done)
#   2) Methodology fix on Thinking + explicit_cot mech
#
# Logs land alongside the n500 logs:
#   outputs/_n500_<stamp>/chain.log
#   outputs/_n500_<stamp>/stage1_resume.log
#   outputs/_n500_<stamp>/stage2_mfix.log

set -uo pipefail

LOG_DIR="${LOG_DIR:-outputs/_n500_2026-05-07_0025}"
mkdir -p "${LOG_DIR}"
CHAIN_LOG="${LOG_DIR}/chain.log"

log() {
    local msg="[$(date +%Y-%m-%d_%H:%M:%S)] $*"
    echo "${msg}"
    echo "${msg}" >> "${CHAIN_LOG}"
}

log "==== overnight chain wrapper started ===="
log "log dir: ${LOG_DIR}"

# -----------------------------------------------------------------------
# Wait for stage 3 (n=500 extension) to finish.
# -----------------------------------------------------------------------
log "waiting for stage 3 (run_n500_extension) to exit..."
WAITED=0
while pgrep -f "run_n500_extension" > /dev/null 2>&1; do
    sleep 60
    WAITED=$((WAITED + 1))
    if (( WAITED % 30 == 0 )); then
        log "still waiting on stage 3 ($((WAITED)) min elapsed)..."
    fi
done
log "stage 3 process gone after ~${WAITED} min."

# -----------------------------------------------------------------------
# Stage 1 — StrategyQA-50 thinking-model resume.
# -----------------------------------------------------------------------
log "==== launching stage 1: StrategyQA-50 thinking-model resume ===="
STAGE1_LOG="${LOG_DIR}/stage1_resume.log"
log "logs → ${STAGE1_LOG}"
STAGE1_START=$(date +%s)
{
    env GSM8K_JSONL=data/fixtures/strategyqa_test_first50/strategyqa.jsonl \
        AJAR_DATASET_KIND=strategyqa \
        SKIP_PARAPHRASE=1 \
        SKIP_PERTURBATION=1 \
        RUN_STAMP=2026-05-06_2121_strategyqa50 \
        bash scripts/run_deep_table.sh 50
} >> "${STAGE1_LOG}" 2>&1
STAGE1_RC=$?
STAGE1_END=$(date +%s)
log "stage 1 exited rc=${STAGE1_RC} after $(( (STAGE1_END - STAGE1_START) / 60 )) min"

# -----------------------------------------------------------------------
# Stage 2 — methodology fix on Thinking + explicit_cot.
# -----------------------------------------------------------------------
log "==== launching stage 2: methodology-fix (Thinking + explicit_cot mech) ===="
STAGE2_LOG="${LOG_DIR}/stage2_mfix.log"
log "logs → ${STAGE2_LOG}"
STAGE2_START=$(date +%s)
{
    bash scripts/run_methodology_fix.sh
} >> "${STAGE2_LOG}" 2>&1
STAGE2_RC=$?
STAGE2_END=$(date +%s)
log "stage 2 exited rc=${STAGE2_RC} after $(( (STAGE2_END - STAGE2_START) / 60 )) min"

log "==== overnight chain wrapper complete ===="
log "summary:"
log "  stage 1 (strategyqa thinking resume): rc=${STAGE1_RC}"
log "  stage 2 (methodology-fix):            rc=${STAGE2_RC}"
