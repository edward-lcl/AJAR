#!/usr/bin/env bash
# Autonomous camera-ready chain. Runs unattended; safe to launch and walk away.
#
#   Phase A  Gemma 5-feature HCDS  (torch entropy -> merge -> recompute)   [RELIABLE]
#   Phase B  Thinking no-CoT pole repair (NoThinking forced </think>)      [BEST-EFFORT]
#
# Phase A runs first so the in-flight deliverable lands even if B fails.
# Every step logs; B failure never touches A's results. Final summary at the end.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=python3
STAMP="$(date +%Y-%m-%d_%H%M)"
SUM="results/runs/CHAIN_SUMMARY_${STAMP}.md"
mkdir -p "$(dirname "$SUM")"
log(){ echo "[chain $(date +%H:%M:%S)] $*"; }
note(){ echo "$*" >> "$SUM"; }

note "# Camera-ready chain — ${STAMP}"
note ""

# ── existing Gemma oMLX run (behavioral features already computed) ────────────
GEMMA_RUN="outputs/2026-06-10_2131_gsm8k50_gemma3-4b_crossfamily"
GEMMA_RES="results/runs/2026-06-10_2131_gsm8k50_gemma3-4b_crossfamily"
FIX="data/variants/gsm8k50_qwen3-4b_deep-table"
GEMMA_TORCH="${GEMMA_RUN}/torch_entropy"

torch_common=( AJAR_BACKEND=torch AJAR_DTYPE=auto AJAR_SAVE_FULL_PROBE_ATTENTION=0 \
               AJAR_ANALYSIS_MAX_SEQ_LEN=4096 )

# ════════════════════════ PHASE A — Gemma 5-feature ═════════════════════════
log "PHASE A: Gemma torch entropy"
note "## Phase A — Gemma 5-feature HCDS"

log "A1: 2-sample self-check (does torch produce entropy for gemma3?)"
rm -rf /tmp/gemma_torch_check
env "${torch_common[@]}" AJAR_MODELS=gemma AJAR_PROMPTS=neutral_strict \
    AJAR_NUM_SAMPLES=2 AJAR_MAX_NEW_TOKENS=256 \
    AJAR_OUTPUT_DIR=/tmp/gemma_torch_check $PY scripts/run_qwen3_gsm8k_mi.py \
    > logs/chain_A1_check_${STAMP}.log 2>&1

ENT_OK=$($PY - <<'PY'
import json,glob,os
ok=0
for f in glob.glob('/tmp/gemma_torch_check/**/baselines.jsonl',recursive=True)+glob.glob('/tmp/gemma_torch_check/baselines.jsonl'):
    for line in open(f):
        try: r=json.loads(line)
        except: continue
        v=r.get('mean_entropy_generated')
        if v is not None: ok=1
print(ok)
PY
)

if [[ "$ENT_OK" == "1" ]]; then
    log "A1 OK: entropy produced. Running full 50-sample torch baseline."
    note "- torch entropy self-check: **PASS**"
    env "${torch_common[@]}" AJAR_MODELS=gemma \
        AJAR_PROMPTS=explicit_cot,explicit_no_cot,neutral_strict \
        AJAR_NUM_SAMPLES=50 AJAR_MAX_NEW_TOKENS=1024 \
        AJAR_OUTPUT_DIR="${GEMMA_TORCH}" $PY scripts/run_qwen3_gsm8k_mi.py \
        > logs/chain_A2_full_${STAMP}.log 2>&1

    log "A3: re-aggregate with entropy + compute 5-feature HCDS"
    $PY scripts/build_task6_table.py \
        --baseline-dir "${GEMMA_RUN}/baseline" --mi-dir "${GEMMA_TORCH}" \
        --paraphrase-dir "${GEMMA_RUN}/paraphrase" --paraphrase-index "${FIX}/paraphrase/index.csv" \
        --perturbation-dir "${GEMMA_RUN}/perturbation" --perturbation-index "${FIX}/perturbation/index.csv" \
        --out "${GEMMA_RES}/task6_table_5feat.csv" >> logs/chain_A3_agg_${STAMP}.log 2>&1
    $PY scripts/compute_hcds.py --task6-csv "${GEMMA_RES}/task6_table_5feat.csv" \
        --out-dir "${GEMMA_RES}/hcds" \
        --exclude-features "mechanistic_intervention_delta_accuracy" \
        --label feat5 > logs/chain_A3_hcds_${STAMP}.log 2>&1
    note "- 5-feature HCDS summary:"
    note '```'
    tail -n +1 "${GEMMA_RES}/hcds/hcds_summary_feat5.csv" >> "$SUM" 2>/dev/null || note "(summary missing — see logs)"
    note '```'
    log "PHASE A complete."
else
    log "A1 FAIL: no entropy from torch gemma3. Keeping 3-feature result; see logs/chain_A1."
    note "- torch entropy self-check: **FAIL** — gemma3 torch path did not yield entropy."
    note "- Fallback: 3-feature behavioral HCDS (+1.67 [1.32,2.00]) stands; entropy left as noted gap."
fi
note ""

# ════════════════════ PHASE B — Thinking no-CoT pole repair ═════════════════
log "PHASE B: Thinking no-CoT pole repair (forced </think>)"
note "## Phase B — Thinking no-CoT pole repair"
THINK_RUN="outputs/${STAMP}_gsm8k50_qwen3-4b_thinking_nocotfix"

set +e
log "B1: cot + neutral (normal template)"
env "${torch_common[@]}" AJAR_MODELS=thinking AJAR_PROMPTS=explicit_cot,neutral_strict \
    AJAR_NUM_SAMPLES=50 AJAR_MAX_NEW_TOKENS=1536 \
    AJAR_OUTPUT_DIR="${THINK_RUN}" $PY scripts/run_qwen3_gsm8k_mi.py \
    > logs/chain_B1_${STAMP}.log 2>&1

log "B2: no_cot with AJAR_FORCE_THINK_CLOSE=1"
env "${torch_common[@]}" AJAR_FORCE_THINK_CLOSE=1 AJAR_MODELS=thinking AJAR_PROMPTS=explicit_no_cot \
    AJAR_NUM_SAMPLES=50 AJAR_MAX_NEW_TOKENS=1536 \
    AJAR_OUTPUT_DIR="${THINK_RUN}" $PY scripts/run_qwen3_gsm8k_mi.py \
    > logs/chain_B2_${STAMP}.log 2>&1
set -e

log "B3: compliance gate + clean-pole HCDS"
THINK_RES="results/runs/${STAMP}_qwen3-4b_thinking_nocotfix"
mkdir -p "${THINK_RES}"
$PY scripts/build_task6_table.py --baseline-dir "${THINK_RUN}" --mi-dir "${THINK_RUN}" \
    --out "${THINK_RES}/task6_table.csv" > logs/chain_B3_agg_${STAMP}.log 2>&1

# Gate: median generated tokens in the no_cot pole (Fable's integrity check).
$PY - "$THINK_RES/task6_table.csv" >> "$SUM" 2>>logs/chain_B3_gate_${STAMP}.log <<'PY'
import csv,sys,statistics
rows=list(csv.DictReader(open(sys.argv[1])))
def med(cond):
    xs=[float(r['output_tokens']) for r in rows if r['prompt_condition']==cond and r.get('output_tokens')]
    return statistics.median(xs) if xs else None
m=med('explicit_no_cot')
print(f"- no-CoT pole median output tokens after force-close: **{m}** (gate target <=16; pre-fix ~491)")
print(f"- gate: {'PASS' if (m is not None and m<=16) else 'CHECK — see logs'}")
PY

$PY scripts/compute_hcds.py --task6-csv "${THINK_RES}/task6_table.csv" \
    --out-dir "${THINK_RES}/hcds" \
    --exclude-features "paraphrase_consistency,perturbation_delta_accuracy,mechanistic_intervention_delta_accuracy" \
    --label clean_pole > logs/chain_B3_hcds_${STAMP}.log 2>&1
note "- clean-pole Thinking HCDS (latency+entropy, torch-consistent):"
note '```'
tail -n +1 "${THINK_RES}/hcds/hcds_summary_clean_pole.csv" >> "$SUM" 2>/dev/null || note "(missing — see logs)"
note '```'
note "- NOTE: paraphrase/perturbation not regenerated on torch (Thinking-OOM risk); 2-feature pole comparison only."

log "CHAIN DONE. Summary: ${SUM}"
note ""
note "_chain complete ${STAMP}_"
