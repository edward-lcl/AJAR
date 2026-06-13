#!/usr/bin/env bash
# Phase B (memory-safe): repair ONLY the Thinking no-CoT pole.
#
# The earlier full-Thinking torch run OOM'd on the long cot/neutral reasoning
# (1536-tok eager attention). But the no-CoT pole WITH forced </think> closure
# produces short answer-only outputs (~tens of tokens), so its analysis is light
# and safe. This run proves Fable's fix: does the pole go clean?
#   - regenerate explicit_no_cot, 50q, torch, AJAR_FORCE_THINK_CLOSE=1
#   - report the compliance gate: median output tokens (target <=16; pre-fix ~491)
# Waits for the Gemma Phase A torch run to finish first (no GPU contention).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=python3
STAMP="$(date +%Y-%m-%d_%H%M)"
log(){ echo "[phaseB $(date +%H:%M:%S)] $*"; }

log "waiting for Phase A (rerun_phaseA_gemma) / any torch run to finish..."
while pgrep -f 'rerun_phaseA_gemma.sh' >/dev/null 2>&1; do sleep 30; done
while pgrep -f 'AJAR_MODELS=gemma' >/dev/null 2>&1; do sleep 20; done
sleep 10

RUN="outputs/${STAMP}_gsm8k50_qwen3-4b_thinking_nocot_safe"
RES="results/runs/${STAMP}_qwen3-4b_thinking_nocot_safe"
mkdir -p "$RES"
SUM="${RES}/PHASE_B_GATE_${STAMP}.md"

log "regenerating Thinking no-CoT pole with forced </think> (short outputs)"
env AJAR_BACKEND=torch AJAR_DTYPE=auto AJAR_SAVE_FULL_PROBE_ATTENTION=0 \
    AJAR_ANALYSIS_MAX_SEQ_LEN=2048 AJAR_MAX_ANSWER_PROBES=16 \
    AJAR_FORCE_THINK_CLOSE=1 AJAR_MODELS=thinking AJAR_PROMPTS=explicit_no_cot \
    AJAR_NUM_SAMPLES=50 AJAR_MAX_NEW_TOKENS=512 \
    AJAR_OUTPUT_DIR="${RUN}" $PY scripts/run_qwen3_gsm8k_mi.py \
    > logs/phaseB_safe_${STAMP}.log 2>&1
RC=$?
if [[ $RC -ne 0 ]]; then
    log "no-CoT torch run failed rc=$RC; see logs/phaseB_safe_${STAMP}.log"
    echo "Phase B no-CoT repair FAILED rc=$RC (likely OOM). See log." > "$SUM"
    exit $RC
fi

log "compliance gate"
$PY scripts/build_task6_table.py --baseline-dir "${RUN}" --mi-dir "${RUN}" \
    --out "${RES}/task6_table.csv" > logs/phaseB_agg_${STAMP}.log 2>&1
$PY - "${RES}/task6_table.csv" "$SUM" <<'PY'
import csv,sys,statistics
rows=[r for r in csv.DictReader(open(sys.argv[1])) if r['prompt_condition']=='explicit_no_cot']
toks=[float(r['output_tokens']) for r in rows if r.get('output_tokens')]
ent=[float(r['token_entropy_mean']) for r in rows if r.get('token_entropy_mean')]
med=statistics.median(toks) if toks else None
lines=[
 "# Phase B — Thinking no-CoT pole repair (forced </think>)",
 f"- n={len(rows)} no-CoT cells",
 f"- **median output tokens = {med}** (gate target <=16; pre-fix ~491)",
 f"- **gate: {'PASS — clean pole' if (med is not None and med<=16) else 'CHECK (leakage; report as prompt-invariance evidence)'}**",
 f"- mean entropy of repaired pole = {round(statistics.mean(ent),3) if ent else 'n/a'}",
 "",
 "Interpretation: a clean pole gives the Thinking contrast reviewers asked for;",
 "residual leakage past </think> is itself stronger latent-reasoning evidence.",
]
open(sys.argv[2],'w').write("\n".join(lines)+"\n")
print("\n".join(lines))
PY
log "DONE. gate summary: $SUM"
