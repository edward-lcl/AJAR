#!/bin/bash
# SageMaker training-job entry point for AJAR MI slice.
# SageMaker invokes this from /opt/ml/code with env HF_TOKEN, AJAR_SAMPLES, S3_BUCKET set.
set -euo pipefail

cd /opt/ml/code

python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

export HF_HOME=/opt/ml/input/data/hfcache
mkdir -p "$HF_HOME"

RUN_STAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
export RUN_STAMP

# Existing AJAR runner — already idempotent, already resume-friendly.
bash scripts/run_torch_mechanistic_slice.sh "${AJAR_SAMPLES:-50}"

# Push outputs to S3 (SageMaker also auto-uploads /opt/ml/model/ to output_path,
# but AJAR writes to outputs/ — copy it across so SageMaker bundles it).
mkdir -p /opt/ml/model
cp -r outputs/* /opt/ml/model/ 2>/dev/null || true
echo "Entry point complete."
