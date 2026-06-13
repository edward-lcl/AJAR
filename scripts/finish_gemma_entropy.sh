#!/usr/bin/env bash
# Finish the Gemma 5-feature HCDS via the fast mlx_lm entropy path (restart-safe).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=python3; STAMP="$(date +%Y-%m-%d_%H%M)"
log(){ echo "[gemma-ent $(date +%H:%M:%S)] $*"; }

GRUN="outputs/2026-06-10_2131_gsm8k50_gemma3-4b_crossfamily"
GRES="results/runs/2026-06-10_2131_gsm8k50_gemma3-4b_crossfamily"
FIX="data/variants/gsm8k50_qwen3-4b_deep-table"
MLXDIR="${GRUN}/mlx_entropy"
SUM="${GRES}/GEMMA_5FEAT_${STAMP}.md"

log "computing entropy (mlx_lm, 4-bit, resumable)"
N_SAMPLES=50 MAXTOK=1024 $PY scripts/gemma_entropy_mlx.py "${MLXDIR}" \
    > logs/gemma_entropy_${STAMP}.log 2>&1
if ! grep -q '^DONE' logs/gemma_entropy_${STAMP}.log; then
    log "entropy did not finish (restart?); re-run this script to resume."
    echo "Entropy incomplete — re-run scripts/finish_gemma_entropy.sh to resume." > "$SUM"; exit 1
fi

log "aggregate + 5-feature HCDS"
$PY scripts/build_task6_table.py \
    --baseline-dir "${GRUN}/baseline" --mi-dir "${MLXDIR}" \
    --paraphrase-dir "${GRUN}/paraphrase" --paraphrase-index "${FIX}/paraphrase/index.csv" \
    --perturbation-dir "${GRUN}/perturbation" --perturbation-index "${FIX}/perturbation/index.csv" \
    --out "${GRES}/task6_table_5feat.csv" > logs/gemma_5feat_agg_${STAMP}.log 2>&1
$PY scripts/compute_hcds.py --task6-csv "${GRES}/task6_table_5feat.csv" \
    --out-dir "${GRES}/hcds" --exclude-features "mechanistic_intervention_delta_accuracy" \
    --label feat5 > logs/gemma_5feat_hcds_${STAMP}.log 2>&1

{ echo "# Gemma 5-feature HCDS (mlx entropy) — ${STAMP}"; echo '```';
  cat "${GRES}/hcds/hcds_summary_feat5.csv"; echo '```';
  echo -n "- entropy-populated rows: ";
  $PY -c "import csv;r=list(csv.DictReader(open('${GRES}/task6_table_5feat.csv')));print(sum(1 for x in r if x.get('token_entropy_mean','').strip()),'/',len(r))";
} > "$SUM"
log "DONE -> ${SUM}"; cat "$SUM"
