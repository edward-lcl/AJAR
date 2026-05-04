#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export AJAR_BACKEND="${AJAR_BACKEND:-torch}"
export AJAR_RUN_MI="${AJAR_RUN_MI:-1}"
export AJAR_NUM_SAMPLES="${AJAR_NUM_SAMPLES:-5}"
export AJAR_MAX_NEW_TOKENS="${AJAR_MAX_NEW_TOKENS:-512}"
export AJAR_PROMPTS="${AJAR_PROMPTS:-explicit_cot,neutral}"
export AJAR_MODELS="${AJAR_MODELS:-instruct}"
export AJAR_DTYPE="${AJAR_DTYPE:-float16}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y-%m-%d_%H%M)}"
RUN_ID="${RUN_ID:-${RUN_STAMP}_gsm8k${AJAR_NUM_SAMPLES}_qwen3-4b_torch_mechanistic-slice}"
export AJAR_OUTPUT_DIR="${AJAR_OUTPUT_DIR:-outputs/${RUN_ID}}"

python3 scripts/run_qwen3_gsm8k_mi.py
