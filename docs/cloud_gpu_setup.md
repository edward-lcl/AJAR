# Running the torch MI slice on a cloud GPU

The mechanistic-interpretability step is the dominant cost in the deep table
because it requires PyTorch forward hooks (which mlx-lm doesn't expose) on
2 models x 3 prompts x N questions x ~6 interventions per baseline. On an
M5 Pro Mac via MPS, even after the single-forward attention-probe
optimization, a 50-sample full-scope MI run takes ~12-16 hours wall clock.

A single A100 finishes the same workload in ~1-2 hours. This document
covers the minimal recipe to run only the MI step on a cloud GPU and merge
the results back with the local oMLX outputs.

## What stays local

Three already-completed phases live on your Mac and don't need cloud:

- Wide neutral_strict baseline (`outputs/<run_id>_omlx_baseline-neutral-strict-1024/`)
- Deep canonical/paraphrase/perturbation oMLX baselines
  (`outputs/<run_id>_deep-table/{baseline,paraphrase,perturbation}/`)
- Variant fixtures (`data/variants/<run_id>_deep-table/{paraphrase,perturbation}/`)

These provide accuracy, latency, output_length, paraphrase consistency, and
perturbation Δ accuracy directly. Only `token_entropy` and
`mechanistic_intervention_delta_accuracy` need cloud compute.

## RunPod recipe (cheapest reliable option)

Pricing as of late 2025: A100 80GB on RunPod community cloud is ~$1.20/hr
on-demand, ~$0.50-0.70/hr spot. A 50-sample MI run is ~1-2 hours on A100
spot, so the total cost is roughly $1-3.

1. Sign in to runpod.io, create an account, add $10 of credit.
2. Templates -> Deploy -> pick "RunPod PyTorch 2.4" (or any image with
   torch 2.x and python 3.10+) on an A100 80GB pod.
3. SSH into the pod once it's ready:
   ```bash
   ssh root@<pod-ip> -p <pod-port>
   ```
4. Bootstrap the workspace:
   ```bash
   cd /workspace
   git clone https://github.com/edward-lcl/AJAR.git
   cd AJAR
   pip install -r requirements.txt
   ```
5. Pre-pull both Qwen weights so they hit the local SSD before the run:
   ```bash
   huggingface-cli download Qwen/Qwen3-4B-Instruct-2507
   huggingface-cli download Qwen/Qwen3-4B-Thinking-2507
   ```
6. Run only the torch MI slice. The runner's resume logic skips any
   sample dirs already completed, so this command is safe to re-run if
   the pod restarts.
   ```bash
   AJAR_BACKEND=torch \
   AJAR_MODELS=instruct,thinking \
   AJAR_PROMPTS=explicit_cot,explicit_no_cot,neutral_strict \
   AJAR_NUM_SAMPLES=50 \
   AJAR_MAX_NEW_TOKENS=1536 \
   AJAR_INTERVENTION_MAX_NEW_TOKENS=384 \
   AJAR_INTERVENTIONS=residual_zero,residual_scale_0.25,attention_zero,attention_scale_0.25 \
   AJAR_TOP_ANCHOR_STEPS=3 \
   AJAR_NUM_CONTROL_STEPS=2 \
   AJAR_TOP_ANCHOR_LAYERS=4 \
   AJAR_RUN_MI=1 \
   AJAR_DTYPE=auto \
   AJAR_SAVE_FULL_PROBE_ATTENTION=0 \
   AJAR_ANALYSIS_MAX_SEQ_LEN=4096 \
   AJAR_OUTPUT_DIR=outputs/cloud_mech \
       python3 scripts/run_qwen3_gsm8k_mi.py
   ```
7. When it finishes, scp the results back:
   ```bash
   scp -P <pod-port> -r root@<pod-ip>:/workspace/AJAR/outputs/cloud_mech \
       outputs/cloud_mech
   ```
8. Locally, aggregate everything:
   ```bash
   python3 scripts/build_task6_table.py \
       --baseline-dir outputs/<run_id>_deep-table/baseline \
       --paraphrase-dir outputs/<run_id>_deep-table/paraphrase \
       --paraphrase-index data/variants/<run_id>_deep-table/paraphrase/index.csv \
       --perturbation-dir outputs/<run_id>_deep-table/perturbation \
       --perturbation-index data/variants/<run_id>_deep-table/perturbation/index.csv \
       --mi-dir outputs/cloud_mech \
       --out results/runs/<run_id>_deep-table/task6_table.csv
   python3 scripts/aggregate_anchor_sensitivity.py \
       --mi-dir outputs/cloud_mech \
       --out results/runs/<run_id>_deep-table/task10_anchor_sensitivity.csv
   ```
9. Stop the pod once you've confirmed the scp completed.

## Alternative venues

- **Lambda Labs**: A10 24GB at ~$0.75/hr. A10 has 24GB VRAM which is
  enough for Qwen3-4B at fp16; expect ~2x slower than A100. Use this if
  RunPod is out.
- **Modal**: serverless per-second billing. Higher overhead per request,
  but you only pay while compute is running. Worth it if you're
  iterating on the runner code itself.
- **vast.ai**: cheapest spot but the reliability is variable; expect
  to hit pre-emptions on long runs. Combine with the runner's resume
  logic if you go this route.

Do **not** use mlx-lm in place of torch for the MI step: it's faster on
Apple Silicon for unconditional generation but doesn't support PyTorch
forward hooks, which the intervention path requires.
