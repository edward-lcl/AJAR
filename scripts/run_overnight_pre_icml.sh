#!/usr/bin/env bash
# Overnight launcher for the pre-ICML push. Fires three independent
# runs in sequence, each with its own log file. A failure in one
# does NOT abort the rest — that way one tooling glitch doesn't take
# the whole night down.
#
# Sequence:
#   1) StrategyQA-50 deep-table       (~5-6 hr, oMLX + torch)
#   2) Methodology fix on Thinking +  (~3-4 hr, torch+MPS only)
#      explicit_cot
#   3) n=500 partial extension        (~5-7 hr, oMLX only)
#      (paraphrase + perturbation on remaining 450)
#
# Total: ~14-17 hr if everything runs. Started at 9-10 PM, finishes
# by 11 AM-2 PM next day. Plenty of margin before the Thursday-night
# cutoff.
#
# Logs land in `outputs/_overnight_<stamp>/`:
#   - 01_strategyqa.log
#   - 02_methodology_fix.log
#   - 03_n500_extension.log
#   - launcher.log  (top-level orchestration log)
#
# Usage:
#   ./scripts/run_overnight_pre_icml.sh
#
# To skip a stage:
#   SKIP_STRATEGYQA=1 SKIP_MFIX=1 ./scripts/run_overnight_pre_icml.sh
#   SKIP_N500_EXT=1 ./scripts/run_overnight_pre_icml.sh

set -uo pipefail

STAMP="${STAMP:-$(date +%Y-%m-%d_%H%M)}"
LOG_DIR="outputs/_overnight_${STAMP}"
mkdir -p "${LOG_DIR}"

LAUNCHER_LOG="${LOG_DIR}/launcher.log"

log() {
    local msg="[$(date +%H:%M:%S)] $*"
    echo "${msg}"
    echo "${msg}" >> "${LAUNCHER_LOG}"
}

run_stage() {
    local label="$1"
    local logfile="$2"
    shift 2
    log "starting ${label}; logging to ${logfile}"
    local start_ts=$(date +%s)
    if "$@" >> "${logfile}" 2>&1; then
        local end_ts=$(date +%s)
        log "${label}: OK ($(( (end_ts - start_ts) / 60 )) min)"
        return 0
    else
        local rc=$?
        local end_ts=$(date +%s)
        log "${label}: FAILED rc=${rc} after $(( (end_ts - start_ts) / 60 )) min"
        log "  see ${logfile} for details"
        return ${rc}
    fi
}

log "==== overnight pre-ICML launcher started ===="
log "log dir: ${LOG_DIR}"
log "stamp:   ${STAMP}"

# -----------------------------------------------------------------------
# Stage 1 — StrategyQA-50 deep-table.
# -----------------------------------------------------------------------
if [[ "${SKIP_STRATEGYQA:-0}" != "1" ]]; then
    log "stage 1/3: StrategyQA-50 deep-table"
    # SKIP_PARAPHRASE / SKIP_PERTURBATION because the existing
    # builders are GSM8K-specific (math distractors,
    # numeric-preservation paraphrasing). StrategyQA HCDS will rest
    # on accuracy + latency + entropy + mech (4/6 features) — still
    # well within the leave-N-out variants we already showed
    # significant on GSM8K.
    run_stage \
        "strategyqa-50" \
        "${LOG_DIR}/01_strategyqa.log" \
        env GSM8K_JSONL=data/fixtures/strategyqa_test_first50/strategyqa.jsonl \
            AJAR_DATASET_KIND=strategyqa \
            SKIP_PARAPHRASE=1 \
            SKIP_PERTURBATION=1 \
            RUN_STAMP="${STAMP}_strategyqa50" \
            bash scripts/run_deep_table.sh 50 || true
else
    log "stage 1/3 SKIPPED (SKIP_STRATEGYQA=1)"
fi

# -----------------------------------------------------------------------
# Stage 2 — Methodology fix on Thinking + explicit_cot.
# -----------------------------------------------------------------------
if [[ "${SKIP_MFIX:-0}" != "1" ]]; then
    log "stage 2/3: methodology fix"
    run_stage \
        "methodology-fix" \
        "${LOG_DIR}/02_methodology_fix.log" \
        bash scripts/run_methodology_fix.sh || true
else
    log "stage 2/3 SKIPPED (SKIP_MFIX=1)"
fi

# -----------------------------------------------------------------------
# Stage 3 — n=500 partial extension (paraphrase + perturbation only).
# -----------------------------------------------------------------------
if [[ "${SKIP_N500_EXT:-0}" != "1" ]]; then
    log "stage 3/3: n=500 partial extension"
    run_stage \
        "n500-extension" \
        "${LOG_DIR}/03_n500_extension.log" \
        env RUN_STAMP="${STAMP}_n500" \
            bash scripts/run_n500_extension.sh || true
else
    log "stage 3/3 SKIPPED (SKIP_N500_EXT=1)"
fi

log "==== overnight launcher finished ===="
log "tail logs with: tail -f ${LOG_DIR}/*.log"
log "summary: grep -E '(starting|OK|FAILED)' ${LAUNCHER_LOG}"
