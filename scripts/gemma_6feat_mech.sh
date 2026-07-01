#!/usr/bin/env bash
# Full 6-feature Gemma HCDS — adds the missing mechanistic feature for cross-model
# parity with the Qwen deep table (Armaan's camera-ready ask).
#
# The paper reports Gemma over 5 features (feat5 = +1.95); the 6th feature,
# mechanistic_intervention_delta_accuracy, was never computed because the original
# torch_entropy run was fp16 and Gemma3-on-MPS in fp16 produces NaN interventions
# (its interventions.jsonl is 0 bytes). Interventions REQUIRE fp32 on this machine.
#
# This resumes the fp32 mechanistic run (torch_entropy_fp32, already ~6 samples in)
# to completion with the SAME settings as the Qwen deep-table mech slice:
#   dtype=float32, max_new_tokens=1024, intervention_max_new_tokens=384,
#   anchor_steps=3, control_steps=2, 4 intervention types (16 tasks/sample).
# Resume is on by default, so it survives the machine's ~20-30 min restarts:
# completed samples are skipped, only the in-progress sample is redone.
#
# Then it aggregates the full 6-feature table (latency from the oMLX baseline dir,
# entropy + mechanistic from the fp32 run, paraphrase + perturbation unchanged) and
# computes HCDS at feat6 AND feat5 (mechanistic excluded) from the SAME table, so
# the team gets an apples-to-apples 5->6 delta on identical entropy.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=python3
STAMP="$(date +%Y-%m-%d_%H%M)"
log(){ echo "[gemma6 $(date +%H:%M:%S)] $*"; }

GEMMA_RUN="outputs/2026-06-10_2131_gsm8k50_gemma3-4b_crossfamily"
GEMMA_RES="results/runs/2026-06-10_2131_gsm8k50_gemma3-4b_crossfamily"
FIX="data/variants/gsm8k50_qwen3-4b_deep-table"
FP32="${GEMMA_RUN}/torch_entropy_fp32"
SUM="${GEMMA_RES}/GEMMA_6FEAT_${STAMP}.md"
mkdir -p "$GEMMA_RES" logs

# ── 1. resume fp32 mechanistic run to completion (50 canonical Q, full interv) ─
log "1/3 resume fp32 mechanistic run -> ${FP32}"
env AJAR_BACKEND=torch AJAR_DTYPE=float32 AJAR_MODELS=gemma \
    AJAR_PROMPTS=explicit_cot,explicit_no_cot,neutral_strict \
    AJAR_NUM_SAMPLES=50 AJAR_MAX_NEW_TOKENS=1024 \
    AJAR_INTERVENTION_MAX_NEW_TOKENS=384 AJAR_MAX_ANSWER_PROBES=16 \
    AJAR_TOP_ANCHOR_STEPS=3 AJAR_NUM_CONTROL_STEPS=2 AJAR_TOP_ANCHOR_LAYERS=4 \
    AJAR_INTERVENTIONS=residual_zero,residual_scale_0.25,attention_zero,attention_scale_0.25 \
    AJAR_SAVE_FULL_PROBE_ATTENTION=0 AJAR_ANALYSIS_MAX_SEQ_LEN=2048 \
    AJAR_OUTPUT_DIR="${FP32}" $PY scripts/run_qwen3_gsm8k_mi.py \
    >> "logs/gemma6_mech_${STAMP}.log" 2>&1
MECH_RC=$?
log "   mech rc=${MECH_RC}"

INTERV_ROWS=$(wc -l < "${FP32}/interventions.jsonl" 2>/dev/null | tr -d ' ')
log "   interventions.jsonl rows: ${INTERV_ROWS:-0}"

# Only aggregate + write the completion summary on a CLEAN exit. A non-zero rc means
# the run was killed mid-pass (machine restart / memory kill); leave no summary so the
# supervisor relaunches (resumable) instead of treating partial data as final.
if [[ ${MECH_RC} -ne 0 ]]; then
    log "mech run did not complete (rc=${MECH_RC}); skipping aggregation, will resume."
    exit "${MECH_RC}"
fi

# ── 2. aggregate full 6-feature table ────────────────────────────────────────
# latency from the published oMLX baseline dir; entropy + mechanistic from fp32.
log "2/3 aggregate 6-feature task6 table"
T6="${GEMMA_RES}/task6_table_6feat.csv"
$PY scripts/build_task6_table.py \
    --baseline-dir "${GEMMA_RUN}/baseline" --mi-dir "${FP32}" \
    --paraphrase-dir "${GEMMA_RUN}/paraphrase" --paraphrase-index "${FIX}/paraphrase/index.csv" \
    --perturbation-dir "${GEMMA_RUN}/perturbation" --perturbation-index "${FIX}/perturbation/index.csv" \
    --out "${T6}" >> "logs/gemma6_agg_${STAMP}.log" 2>&1
log "   aggregate rc=$? -> ${T6}"

# ── 3. compute HCDS: feat6 (all six) and feat5 (mech excluded) on same table ──
log "3/3 compute HCDS (feat6 + feat5 A/B)"
$PY scripts/compute_hcds.py --task6-csv "${T6}" --out-dir "${GEMMA_RES}/hcds" --label feat6_full \
    >> "logs/gemma6_hcds_${STAMP}.log" 2>&1
$PY scripts/compute_hcds.py --task6-csv "${T6}" --out-dir "${GEMMA_RES}/hcds" --label feat5_from6tbl \
    --exclude-features "mechanistic_intervention_delta_accuracy" >> "logs/gemma6_hcds_${STAMP}.log" 2>&1

gemma_row(){ grep '^gemma,' "$1" 2>/dev/null || echo "(no gemma row in $1)"; }
{
  echo "# Gemma full 6-feature HCDS — ${STAMP}"
  echo
  echo "fp32 mechanistic interventions added (the 6th feature missing from the paper's"
  echo "feat5 = +1.95). Latency from oMLX baseline; entropy + mechanistic from fp32 run;"
  echo "paraphrase + perturbation unchanged. Match key: canonical 50 GSM8K test Q."
  echo
  echo "## 6-feature HCDS (gemma)"
  echo "feat6 (full):              $(gemma_row "${GEMMA_RES}/hcds/hcds_summary_feat6_full.csv")"
  echo "feat5 (same table, no mech): $(gemma_row "${GEMMA_RES}/hcds/hcds_summary_feat5_from6tbl.csv")"
  echo "feat5 (published, fp16 ent): $(gemma_row "${GEMMA_RES}/hcds/hcds_summary_feat5.csv")"
  echo
  echo "## Feature coverage (6-feat table)"
  $PY - "${T6}" <<'PY'
import csv,sys,statistics
rows=[r for r in csv.DictReader(open(sys.argv[1])) if r.get("model")=="gemma"]
feats=["latency_per_output_token","token_entropy_mean","token_entropy_slope",
       "paraphrase_consistency","perturbation_delta_accuracy","mechanistic_intervention_delta_accuracy"]
print(f"gemma rows: {len(rows)}")
for f in feats:
    print(f"  {f}: {sum(1 for r in rows if (r.get(f) or '').strip())}/{len(rows)} populated")
PY
} > "$SUM"
log "DONE. summary: $SUM"
echo "================ SUMMARY ================"; cat "$SUM"
