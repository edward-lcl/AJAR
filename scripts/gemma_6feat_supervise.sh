#!/usr/bin/env bash
# Detached supervisor for the Gemma 6-feature fp32 run. Relaunches the resumable
# gemma_6feat_mech.sh after any OOM/process kill until it exits clean (rc==0, which
# is the only path that writes the GEMMA_6FEAT summary). Run via nohup so it is
# independent of any harness background-task lifetime:
#   cd /Users/edward/Projects/AJAR && nohup bash scripts/gemma_6feat_supervise.sh \
#       >> logs/gemma6_supervisor.out 2>&1 &
cd "$(dirname "$0")/.."
n=0
until bash scripts/gemma_6feat_mech.sh; do
  n=$((n+1))
  echo "[supervise $(date +%H:%M:%S)] non-clean exit; relaunch #$n (resumable)"
  sleep 15
done
echo "[supervise $(date +%H:%M:%S)] RUN COMPLETE — summary:"
cat results/runs/2026-06-10_2131_gsm8k50_gemma3-4b_crossfamily/GEMMA_6FEAT_*.md 2>/dev/null
