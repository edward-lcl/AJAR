#!/usr/bin/env bash
# Full 6-feature CLEAN-POLE Thinking HCDS (Armaan's camera-ready ask, item 8 in full).
#
# The paper currently reports the clean NoThinking pole only over latency+entropy
# (2 families). This regenerates the Thinking explicit_no_cot cell over ALL SIX
# features against the clean pole, then recomputes HCDS by splicing the clean
# no_cot rows into the canonical n=50 deep table (Thinking cot/neutral + Instruct
# left exactly as published).
#
# WHY torch for every sub-run: AJAR_FORCE_THINK_CLOSE (the NoThinking forced
# </think> closure) only fires in build_prompt_text, the torch path. oMLX applies
# its own server-side template and cannot produce the clean pole. The clean pole
# is answer-only (median ~8 tokens) so these runs are SHORT and memory-safe --
# unlike the long contaminated reasoning that OOM'd historical full-torch attempts.
#
# Three torch sub-runs, all Thinking / explicit_no_cot / clean pole:
#   mech         50 canonical Q, full interventions -> latency + entropy + mechanistic
#                (baselines.jsonl carries latency+entropy, so it doubles as baseline-dir)
#   paraphrase   variants, interventions zeroed      -> paraphrase_consistency
#   perturbation variants, interventions zeroed      -> perturbation_delta_accuracy
#
# Resumable: the harness skips samples whose artifacts already exist.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=python3
STAMP="$(date +%Y-%m-%d_%H%M)"
log(){ echo "[cleanpole6 $(date +%H:%M:%S)] $*"; }

RUN="outputs/${STAMP}_gsm8k50_thinking_cleanpole_6feat"
RES="results/runs/${STAMP}_thinking_cleanpole_6feat"
FIX="data/variants/gsm8k50_qwen3-4b_deep-table"
CANON="results/runs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/task6_table.csv"
mkdir -p "$RUN" "$RES" logs
SUM="${RES}/CLEANPOLE6_${STAMP}.md"

# Shared torch + clean-pole settings. Short outputs, so a modest token cap is safe.
COMMON_ENV=(
  AJAR_BACKEND=torch AJAR_DTYPE=auto AJAR_FORCE_THINK_CLOSE=1
  AJAR_MODELS=thinking AJAR_PROMPTS=explicit_no_cot
  AJAR_SAVE_FULL_PROBE_ATTENTION=0 AJAR_ANALYSIS_MAX_SEQ_LEN=2048
  AJAR_MAX_NEW_TOKENS=128
)
# Interventions OFF (anchor+control step budgets = 0) for the behavioral variant runs.
NOINTERV=( AJAR_TOP_ANCHOR_STEPS=0 AJAR_NUM_CONTROL_STEPS=0 )

# ── 1. mech slice: latency + entropy + mechanistic, 50 canonical Q ───────────
log "1/3 mech (latency + entropy + mechanistic) — clean pole, full interventions"
env "${COMMON_ENV[@]}" AJAR_NUM_SAMPLES=50 \
    AJAR_INTERVENTION_MAX_NEW_TOKENS=128 \
    AJAR_INTERVENTIONS=residual_zero,attention_zero \
    AJAR_TOP_ANCHOR_STEPS=2 AJAR_NUM_CONTROL_STEPS=1 AJAR_TOP_ANCHOR_LAYERS=4 \
    AJAR_MAX_ANSWER_PROBES=16 \
    AJAR_OUTPUT_DIR="${RUN}/mech" $PY scripts/run_qwen3_gsm8k_mi.py \
    >> "logs/cleanpole6_mech_${STAMP}.log" 2>&1
log "   mech rc=$?"

# ── 2. paraphrase variants (paraphrase_consistency), interventions off ───────
PARA_ROWS=$(wc -l < "${FIX}/paraphrase/variants.jsonl")
log "2/3 paraphrase (${PARA_ROWS} rows) — clean pole, no interventions"
env "${COMMON_ENV[@]}" "${NOINTERV[@]}" AJAR_NUM_SAMPLES="${PARA_ROWS}" \
    GSM8K_JSONL="${FIX}/paraphrase/variants.jsonl" \
    AJAR_OUTPUT_DIR="${RUN}/paraphrase" $PY scripts/run_qwen3_gsm8k_mi.py \
    >> "logs/cleanpole6_para_${STAMP}.log" 2>&1
log "   paraphrase rc=$?"

# ── 3. perturbation variants (perturbation_delta_accuracy), interventions off ─
PERT_ROWS=$(wc -l < "${FIX}/perturbation/variants.jsonl")
log "3/3 perturbation (${PERT_ROWS} rows) — clean pole, no interventions"
env "${COMMON_ENV[@]}" "${NOINTERV[@]}" AJAR_NUM_SAMPLES="${PERT_ROWS}" \
    GSM8K_JSONL="${FIX}/perturbation/variants.jsonl" \
    AJAR_OUTPUT_DIR="${RUN}/perturbation" $PY scripts/run_qwen3_gsm8k_mi.py \
    >> "logs/cleanpole6_perturb_${STAMP}.log" 2>&1
log "   perturbation rc=$?"

# ── 4. aggregate clean no_cot rows (mech dir doubles as baseline-dir) ─────────
log "4/5 aggregate clean no_cot task6 rows"
CLEAN_T6="${RES}/task6_cleanpole_nocot.csv"
$PY scripts/build_task6_table.py \
    --baseline-dir "${RUN}/mech" --mi-dir "${RUN}/mech" \
    --paraphrase-dir "${RUN}/paraphrase" --paraphrase-index "${FIX}/paraphrase/index.csv" \
    --perturbation-dir "${RUN}/perturbation" --perturbation-index "${FIX}/perturbation/index.csv" \
    --out "${CLEAN_T6}" >> "logs/cleanpole6_agg_${STAMP}.log" 2>&1
log "   aggregate rc=$? -> ${CLEAN_T6}"

# ── 5. splice + recompute HCDS (clean vs canonical contaminated) ─────────────
log "5/5 splice into canonical n=50 table and recompute HCDS"
SPLICED="${RES}/task6_cleanpole_spliced.csv"
$PY scripts/splice_cleanpole.py --canonical "${CANON}" --clean "${CLEAN_T6}" --out "${SPLICED}" \
    >> "logs/cleanpole6_splice_${STAMP}.log" 2>&1
log "   splice rc=$? -> ${SPLICED}"

LAT=latency_per_output_token
# 6-feature and (backend-neutral) latency-excluded HCDS, both poles, for an A/B.
$PY scripts/compute_hcds.py --task6-csv "${CANON}"   --out-dir "${RES}" --label contaminated     >> "logs/cleanpole6_hcds_${STAMP}.log" 2>&1
$PY scripts/compute_hcds.py --task6-csv "${SPLICED}" --out-dir "${RES}" --label cleanpole        >> "logs/cleanpole6_hcds_${STAMP}.log" 2>&1
$PY scripts/compute_hcds.py --task6-csv "${CANON}"   --out-dir "${RES}" --label contaminated_nolat --exclude-features "${LAT}" >> "logs/cleanpole6_hcds_${STAMP}.log" 2>&1
$PY scripts/compute_hcds.py --task6-csv "${SPLICED}" --out-dir "${RES}" --label cleanpole_nolat    --exclude-features "${LAT}" >> "logs/cleanpole6_hcds_${STAMP}.log" 2>&1

thinking_row(){ grep '^thinking,' "$1" 2>/dev/null || echo "(no thinking row in $1)"; }
{
  echo "# Thinking CLEAN-POLE HCDS (full feature set) — ${STAMP}"
  echo
  echo "Clean NoThinking pole (forced </think>, torch, all 6 features) spliced into the"
  echo "canonical n=50 deep table. Thinking cot/neutral + all Instruct rows unchanged."
  echo
  echo "## 6-feature HCDS (thinking)"
  echo "contaminated (published): $(thinking_row "${RES}/hcds_summary_contaminated.csv")"
  echo "clean pole (NoThinking):  $(thinking_row "${RES}/hcds_summary_cleanpole.csv")"
  echo
  echo "## 5-feature, latency excluded (backend-neutral robustness; clean-pole latency is torch-sourced)"
  echo "contaminated: $(thinking_row "${RES}/hcds_summary_contaminated_nolat.csv")"
  echo "clean pole:   $(thinking_row "${RES}/hcds_summary_cleanpole_nolat.csv")"
  echo
  echo "## Clean no_cot feature coverage"
  $PY - "${CLEAN_T6}" <<'PY'
import csv,sys,statistics
rows=list(csv.DictReader(open(sys.argv[1])))
feats=["latency_per_output_token","token_entropy_mean","token_entropy_slope",
       "paraphrase_consistency","perturbation_delta_accuracy","mechanistic_intervention_delta_accuracy"]
print(f"clean no_cot rows: {len(rows)}")
for f in feats:
    print(f"  {f}: {sum(1 for r in rows if (r.get(f) or '').strip())}/{len(rows)} populated")
toks=[float(r['output_tokens']) for r in rows if (r.get('output_tokens') or '').strip()]
if toks: print(f"  output_tokens: median={statistics.median(toks)} (clean-pole gate; pre-fix ~491)")
PY
} > "$SUM"
log "DONE. summary: $SUM"
echo "================ SUMMARY ================"; cat "$SUM"
