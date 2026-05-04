#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export AJAR_BACKEND="${AJAR_BACKEND:-omlx}"
export AJAR_NUM_SAMPLES="${AJAR_NUM_SAMPLES:-500}"
export AJAR_MAX_NEW_TOKENS="${AJAR_MAX_NEW_TOKENS:-1024}"
export AJAR_PROMPTS="${AJAR_PROMPTS:-explicit_cot,explicit_no_cot,neutral}"
export AJAR_OMLX_CONCURRENCY="${AJAR_OMLX_CONCURRENCY:-4}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y-%m-%d_%H%M)}"
RUN_ID="${RUN_ID:-${RUN_STAMP}_gsm8k${AJAR_NUM_SAMPLES}_qwen3-4b_omlx_baseline-rerun-nocot-${AJAR_MAX_NEW_TOKENS}}"
export AJAR_OUTPUT_DIR="${AJAR_OUTPUT_DIR:-outputs/${RUN_ID}}"

python3 scripts/run_qwen3_gsm8k_mi.py
AJAR_ANALYSIS_OUT_DIR="${AJAR_ANALYSIS_OUT_DIR:-analysis/${RUN_ID}}" \
  python3 scripts/analyze_baseline_run.py "$AJAR_OUTPUT_DIR"
