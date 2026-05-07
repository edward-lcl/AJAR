#!/usr/bin/env bash
# Re-run only the mech step for Thinking + explicit_cot with the
# late-trace position penalty applied to anchor scoring. Produces a
# parallel mech/ dir that the aggregator can fold in.
#
# Background: the original anchor-vs-control delta on Thinking +
# explicit_cot came out negative (-0.275). The combined-anchor-score
# proxy ranks late-trace summary steps highest, but late-trace steps
# are recoverable from earlier context, so suppressing them doesn't
# hurt — while suppressing earlier randomly-chosen control steps does.
# Penalising late-trace position in the score (`AJAR_ANCHOR_LATE_PENALTY`)
# is the cheap fix we documented in the SUMMARY.
#
# Usage:
#   ./scripts/run_methodology_fix.sh        # alpha=0.5 (mild)
#   AJAR_ANCHOR_LATE_PENALTY=1.0 \
#     ./scripts/run_methodology_fix.sh      # aggressive (last step → 0)
#
# Compute estimate: ~3-4 hr on M5 Pro for 50 Thinking questions
# under explicit_cot with full MI.

set -euo pipefail

ALPHA="${AJAR_ANCHOR_LATE_PENALTY:-0.5}"
SOURCE_RUN_STAMP="${SOURCE_RUN_STAMP:-2026-05-04_2232}"
SOURCE_TAG="${SOURCE_TAG:-gsm8k50_qwen3-4b_deep-table}"

RUN_ROOT_NEW="${RUN_ROOT_NEW:-outputs/${SOURCE_RUN_STAMP}_${SOURCE_TAG}-mfix-alpha${ALPHA}}"
RESULTS_ROOT="${RESULTS_ROOT:-results/runs/${SOURCE_RUN_STAMP}_${SOURCE_TAG}}"

mkdir -p "${RUN_ROOT_NEW}/mech"

echo "[mfix] re-running mech step on Thinking + explicit_cot"
echo "[mfix]   late-position penalty alpha = ${ALPHA}"
echo "[mfix]   output dir = ${RUN_ROOT_NEW}/mech"
echo "[mfix]   results dir = ${RESULTS_ROOT}"

AJAR_BACKEND=torch \
AJAR_MODELS=thinking \
AJAR_PROMPTS=explicit_cot \
AJAR_NUM_SAMPLES="${NUM_SAMPLES:-50}" \
AJAR_MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}" \
AJAR_INTERVENTION_MAX_NEW_TOKENS="${INTERVENTION_MAX_NEW_TOKENS:-384}" \
AJAR_INTERVENTIONS="${INTERVENTIONS:-residual_zero,attention_zero}" \
AJAR_TOP_ANCHOR_STEPS="${TOP_ANCHOR_STEPS:-2}" \
AJAR_NUM_CONTROL_STEPS="${NUM_CONTROL_STEPS:-1}" \
AJAR_TOP_ANCHOR_LAYERS="${TOP_ANCHOR_LAYERS:-4}" \
AJAR_MAX_ANSWER_PROBES="${MAX_ANSWER_PROBES:-64}" \
AJAR_RUN_MI=1 \
AJAR_DTYPE=auto \
AJAR_SAVE_FULL_PROBE_ATTENTION=0 \
AJAR_ANALYSIS_MAX_SEQ_LEN="${ANALYSIS_MAX_SEQ_LEN:-4096}" \
AJAR_RESUME=0 \
AJAR_ANCHOR_LATE_PENALTY="${ALPHA}" \
AJAR_OUTPUT_DIR="${RUN_ROOT_NEW}/mech" \
    python3 scripts/run_qwen3_gsm8k_mi.py

echo "[mfix] aggregating Task 10 anchor sensitivity (mfix variant)..."
python3 scripts/aggregate_anchor_sensitivity.py \
    --mi-dir "${RUN_ROOT_NEW}/mech" \
    --out "${RESULTS_ROOT}/task10_anchor_sensitivity_mfix_alpha${ALPHA}.csv"

echo "[mfix] done."
echo "[mfix] compare:"
echo "       original: ${RESULTS_ROOT}/task10_anchor_sensitivity.csv"
echo "       mfix:     ${RESULTS_ROOT}/task10_anchor_sensitivity_mfix_alpha${ALPHA}.csv"
