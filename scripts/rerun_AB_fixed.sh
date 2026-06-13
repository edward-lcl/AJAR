#!/usr/bin/env bash
# Clean re-run of Phase A (Gemma 5-feature) + Phase B (Thinking no-CoT gate).
# Fixes from the first attempt:
#   - Gemma torch now AJAR_DTYPE=float32 (fp16 on MPS gave NaN -> empty output)
#   - Phases run sequentially in ONE process (no pgrep-on-script-name deadlock)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=python3
STAMP="$(date +%Y-%m-%d_%H%M)"
log(){ echo "[ABfix $(date +%H:%M:%S)] $*"; }

GRUN="outputs/2026-06-10_2131_gsm8k50_gemma3-4b_crossfamily"
GRES="results/runs/2026-06-10_2131_gsm8k50_gemma3-4b_crossfamily"
FIX="data/variants/gsm8k50_qwen3-4b_deep-table"
GTORCH="${GRUN}/torch_entropy_fp32"
SUM="${GRES}/AB_FIXED_SUMMARY_${STAMP}.md"
echo "# Phase A + B re-run (${STAMP})" > "$SUM"

# ── Phase A: Gemma 5-feature, float32 ───────────────────────────────────────
log "PHASE A: gemma torch entropy (float32), 50 samples"
env AJAR_BACKEND=torch AJAR_DTYPE=float32 AJAR_SAVE_FULL_PROBE_ATTENTION=0 \
    AJAR_ANALYSIS_MAX_SEQ_LEN=2048 AJAR_MAX_ANSWER_PROBES=16 AJAR_MODELS=gemma \
    AJAR_PROMPTS=explicit_cot,explicit_no_cot,neutral_strict \
    AJAR_NUM_SAMPLES=50 AJAR_MAX_NEW_TOKENS=1024 \
    AJAR_OUTPUT_DIR="${GTORCH}" $PY scripts/run_qwen3_gsm8k_mi.py \
    > logs/ABfix_A_${STAMP}.log 2>&1 && A_OK=1 || A_OK=0

if [[ "$A_OK" == "1" ]]; then
    $PY scripts/build_task6_table.py \
        --baseline-dir "${GRUN}/baseline" --mi-dir "${GTORCH}" \
        --paraphrase-dir "${GRUN}/paraphrase" --paraphrase-index "${FIX}/paraphrase/index.csv" \
        --perturbation-dir "${GRUN}/perturbation" --perturbation-index "${FIX}/perturbation/index.csv" \
        --out "${GRES}/task6_table_5feat.csv" > logs/ABfix_Aagg_${STAMP}.log 2>&1
    $PY scripts/compute_hcds.py --task6-csv "${GRES}/task6_table_5feat.csv" \
        --out-dir "${GRES}/hcds" --exclude-features "mechanistic_intervention_delta_accuracy" \
        --label feat5 > logs/ABfix_Ahcds_${STAMP}.log 2>&1
    { echo "## Phase A — Gemma 5-feature HCDS"; echo '```';
      cat "${GRES}/hcds/hcds_summary_feat5.csv"; echo '```';
      echo "- entropy populated rows:"; $PY -c "import csv;rows=list(csv.DictReader(open('${GRES}/task6_table_5feat.csv')));print(sum(1 for r in rows if r.get('token_entropy_mean','').strip()),'/',len(rows))";
    } >> "$SUM"
    log "PHASE A done."
else
    echo "## Phase A FAILED — see logs/ABfix_A_${STAMP}.log" >> "$SUM"
    log "PHASE A failed."
fi

# ── Phase B: Thinking no-CoT pole gate (short force-closed outputs) ──────────
log "PHASE B: thinking no-CoT pole, forced </think>"
BRUN="outputs/${STAMP}_gsm8k50_qwen3-4b_thinking_nocot_safe"
BRES="results/runs/${STAMP}_qwen3-4b_thinking_nocot_safe"; mkdir -p "$BRES"
env AJAR_BACKEND=torch AJAR_DTYPE=float32 AJAR_SAVE_FULL_PROBE_ATTENTION=0 \
    AJAR_ANALYSIS_MAX_SEQ_LEN=2048 AJAR_MAX_ANSWER_PROBES=16 \
    AJAR_FORCE_THINK_CLOSE=1 AJAR_MODELS=thinking AJAR_PROMPTS=explicit_no_cot \
    AJAR_NUM_SAMPLES=50 AJAR_MAX_NEW_TOKENS=512 \
    AJAR_OUTPUT_DIR="${BRUN}" $PY scripts/run_qwen3_gsm8k_mi.py \
    > logs/ABfix_B_${STAMP}.log 2>&1 && B_OK=1 || B_OK=0

if [[ "$B_OK" == "1" ]]; then
    $PY scripts/build_task6_table.py --baseline-dir "${BRUN}" --mi-dir "${BRUN}" \
        --out "${BRES}/task6_table.csv" > logs/ABfix_Bagg_${STAMP}.log 2>&1
    $PY - "${BRES}/task6_table.csv" >> "$SUM" <<'PY'
import csv,sys,statistics
rows=[r for r in csv.DictReader(open(sys.argv[1])) if r['prompt_condition']=='explicit_no_cot']
toks=[float(r['output_tokens']) for r in rows if r.get('output_tokens')]
m=statistics.median(toks) if toks else None
print("## Phase B — Thinking no-CoT pole repair (forced </think>)")
print(f"- n={len(rows)}  median output tokens = **{m}**  (gate target <=16; pre-fix ~491)")
print(f"- gate: **{'PASS — clean pole' if (m is not None and m<=16) else 'leakage -> report as prompt-invariance evidence'}**")
PY
    log "PHASE B done."
else
    echo "## Phase B FAILED — see logs/ABfix_B_${STAMP}.log" >> "$SUM"
    log "PHASE B failed."
fi
log "ALL DONE. summary: ${SUM}"
