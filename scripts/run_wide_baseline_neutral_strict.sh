#!/usr/bin/env bash
# Wide oMLX baseline at neutral_strict (Task 1, HCDS-clean version).
#
# This is the 500-sample re-baseline that supersedes the existing
# 2026-05-04_0238 rerun for HCDS purposes. The original 'neutral' prompt
# included an "Answer the question directly" + \boxed{} directive which
# contaminated the Neutral condition. This run uses neutral_strict, which
# matches the proposal verbatim.
#
# Settings match the 2026-05-04 rerun (3 prompts, 1024 token cap, oMLX
# concurrency 4) so wide-table comparisons are apples-to-apples.

set -euo pipefail

NUM_SAMPLES="${NUM_SAMPLES:-500}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
OMLX_CONCURRENCY="${OMLX_CONCURRENCY:-4}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y-%m-%d_%H%M)}"
RUN_ID="${RUN_ID:-${RUN_STAMP}_gsm8k${NUM_SAMPLES}_qwen3-4b_omlx_baseline-neutral-strict-${MAX_TOKENS}}"

export AJAR_BACKEND=omlx
export AJAR_MODELS="${AJAR_MODELS:-instruct,thinking}"
export AJAR_PROMPTS="${AJAR_PROMPTS:-explicit_cot,explicit_no_cot,neutral_strict}"
export AJAR_NUM_SAMPLES="${NUM_SAMPLES}"
export AJAR_MAX_NEW_TOKENS="${MAX_TOKENS}"
export AJAR_OMLX_CONCURRENCY="${OMLX_CONCURRENCY}"
export AJAR_OUTPUT_DIR="${AJAR_OUTPUT_DIR:-outputs/${RUN_ID}}"

echo "[wide-baseline] launching ${RUN_ID}"
python3 scripts/run_qwen3_gsm8k_mi.py
AJAR_ANALYSIS_OUT_DIR="${AJAR_ANALYSIS_OUT_DIR:-analysis/${RUN_ID}}" \
  python3 scripts/analyze_baseline_run.py "$AJAR_OUTPUT_DIR"
echo "[wide-baseline] complete; analysis written to analysis/${RUN_ID}"
