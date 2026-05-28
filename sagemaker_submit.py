"""Submit a SageMaker training job for the AJAR Qwen3 MI slice.

Wraps ``scripts/run_qwen3_gsm8k_mi.py`` (the same entry point ``run_deep_table.sh``
calls locally). Defaults match a single-GPU L4 (ml.g6.xlarge) and the deep-table
shape from README / docs/cloud_gpu_setup.md (~1-2 hr on A100; expect ~3-5 hr on L4).

Usage:
    python sagemaker_submit.py --samples 5  --max-runtime-hours 1 --spot   # smoke
    python sagemaker_submit.py --samples 50 --max-runtime-hours 6 --spot   # full deep table
    python sagemaker_submit.py --samples 50 --models thinking \\
        --max-new-tokens 2048 --intervention-max-new-tokens 768 --spot     # Thinking-only re-run
"""

import argparse
import json
import os
import shutil
import tempfile
import time

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch


# Paths copied directly into the staging dir.
SOURCE_WHITELIST = (
    "scripts",
    "tests",
    "data",         # paraphrase / perturbation fixtures
    "third_party",  # mlx_interp shim
    "pyproject.toml",
    "README.md",
)
# Special-case: requirements-sagemaker.txt → requirements.txt in staging.
# SageMaker's PyTorch container auto-installs requirements.txt at the source-dir
# root before invoking the entry point. We use a SageMaker-specific version that
# skips the torch pin (DLC ships 2.4, AJAR pins 2.8 — reinstalling adds ~3 min
# cold-start cost we don't need for inference).
SAGEMAKER_REQUIREMENTS_SRC = "requirements-sagemaker.txt"


def stage_source_dir(repo_root: str) -> str:
    """Copy whitelisted paths into a temp dir; cwd itself is 2.4GB+ from outputs/."""
    staging = tempfile.mkdtemp(prefix="ajar-sm-")
    for name in SOURCE_WHITELIST:
        src = os.path.join(repo_root, name)
        if not os.path.exists(src):
            continue
        dst = os.path.join(staging, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", ".pytest_cache", "*.npz", "*.pt", "*.bin", "*.safetensors"))
        else:
            shutil.copy2(src, dst)
    sm_req = os.path.join(repo_root, SAGEMAKER_REQUIREMENTS_SRC)
    if not os.path.exists(sm_req):
        raise FileNotFoundError(f"missing {SAGEMAKER_REQUIREMENTS_SRC} at {repo_root}")
    shutil.copy2(sm_req, os.path.join(staging, "requirements.txt"))
    return staging


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instance", default=os.environ.get("INSTANCE_TYPE", "ml.g6.xlarge"))
    p.add_argument("--max-runtime-hours", type=int, default=6,
                   help="Hard cap; spot jobs auto-resume if preempted under max_wait")
    p.add_argument("--spot", action="store_true",
                   help="Use spot instances (60-70%% cheaper; safe — AJAR runner is resume-friendly)")
    p.add_argument("--no-wait", action="store_true",
                   help="Submit and return without streaming logs (poll with describe-training-job)")
    p.add_argument("--job-name", default=None)

    p.add_argument("--samples", type=int, default=50)
    p.add_argument("--models", default="instruct,thinking")
    p.add_argument("--prompts", default="explicit_cot,explicit_no_cot,neutral_strict")
    p.add_argument("--max-new-tokens", type=int, default=1536)
    p.add_argument("--intervention-max-new-tokens", type=int, default=384)
    p.add_argument("--interventions",
                   default="residual_zero,residual_scale_0.25,attention_zero,attention_scale_0.25")
    p.add_argument("--top-anchor-steps", type=int, default=3)
    p.add_argument("--num-control-steps", type=int, default=2)
    p.add_argument("--top-anchor-layers", type=int, default=4)
    p.add_argument("--max-answer-probes", type=int, default=None,
                   help="Override AJAR_MAX_ANSWER_PROBES. Default 64 fits the per-sample attention "
                        "array on 24 GB. On L40S/A100/H100 (>=40 GB) pass 99999 to disable the cap "
                        "and resolve docs/run_postmortem.md validity caveat #3.")
    p.add_argument("--no-mi", action="store_true",
                   help="Skip mechanistic-interpretability slice; baseline-only generations")
    p.add_argument("--parallel-gpus", type=int, default=1,
                   help="Number of GPUs on the chosen instance. >1 sets AJAR_PARALLEL=1 and a "
                        "shared GPU allocation across all models (worker pool steals across "
                        "GPUs). Must match the instance type's GPU count (g6.12xlarge=4).")
    args = p.parse_args()

    region = os.environ["AWS_REGION"]
    bucket = os.environ["S3_BUCKET"]
    handle = os.environ["STUDENT_HANDLE"]
    project = os.environ["PROJECT_SLUG"]
    account = boto3.client("sts").get_caller_identity()["Account"]
    role_arn = f"arn:aws:iam::{account}:role/AmazonSageMaker-{handle}"

    # AWS Deep Learning Container — PyTorch 2.4 / py311 / CUDA 12.4.
    # AJAR's pyproject pins torch==2.8.0; the container's requirements path will
    # upgrade torch at job start (~2-3 min cold-start cost). Either accept the
    # cost, relax the AJAR pin to 'torch>=2.4', or build a custom ECR image.
    # Catalog: https://github.com/aws/deep-learning-containers/blob/master/available_images.md
    image_uri = f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.4.0-gpu-py311-cu124-ubuntu22.04-sagemaker"

    job_name = args.job_name or f"{project}-mi-{int(time.time())}"
    staging_dir = stage_source_dir(os.getcwd())

    # AJAR_* are consumed by scripts/run_qwen3_gsm8k_mi.py. Without
    # AJAR_BACKEND=torch the runner defaults to oMLX, which has no server
    # in this container and will fail at startup.
    ajar_env = {
        "AJAR_BACKEND": "torch",
        "AJAR_MODELS": args.models,
        "AJAR_PROMPTS": args.prompts,
        "AJAR_NUM_SAMPLES": str(args.samples),
        "AJAR_MAX_NEW_TOKENS": str(args.max_new_tokens),
        "AJAR_INTERVENTION_MAX_NEW_TOKENS": str(args.intervention_max_new_tokens),
        "AJAR_INTERVENTIONS": args.interventions,
        "AJAR_TOP_ANCHOR_STEPS": str(args.top_anchor_steps),
        "AJAR_NUM_CONTROL_STEPS": str(args.num_control_steps),
        "AJAR_TOP_ANCHOR_LAYERS": str(args.top_anchor_layers),
        "AJAR_RUN_MI": "0" if args.no_mi else "1",
        "AJAR_DTYPE": "auto",
        # Qwen3-4B-Thinking-2507 needs remote-code resolution on clean Linux
        # containers; without this the torch backend fails to instantiate the
        # Thinking class after the model weights download.
        "AJAR_TRUST_REMOTE_CODE": "1",
        "AJAR_SAVE_FULL_PROBE_ATTENTION": "0",
        "AJAR_ANALYSIS_MAX_SEQ_LEN": "4096",
        "AJAR_OUTPUT_DIR": "/opt/ml/output/data/ajar_run",
        "HF_TOKEN": os.environ["HF_TOKEN"],
        "S3_BUCKET": bucket,
        # HF cache on the EBS volume, not /tmp (small + tmpfs OOM-prone)
        "HF_HOME": "/opt/ml/input/data/hf_cache",
        "TRANSFORMERS_CACHE": "/opt/ml/input/data/hf_cache",
    }
    if args.parallel_gpus > 1:
        gpu_ids = list(range(args.parallel_gpus))
        model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
        ajar_env["AJAR_PARALLEL"] = "1"
        ajar_env["AJAR_MODEL_GPU_ALLOCATIONS"] = json.dumps(
            {model_key: gpu_ids for model_key in model_keys}
        )
    if args.max_answer_probes is not None:
        ajar_env["AJAR_MAX_ANSWER_PROBES"] = str(args.max_answer_probes)

    # PyTorch estimator (vs generic Estimator) is what triggers requirements.txt
    # auto-install at the source-dir root. With generic Estimator the file is
    # uploaded but never installed — silent no-op that costs you a full job.
    estimator = PyTorch(
        image_uri=image_uri,
        role=role_arn,
        instance_type=args.instance,
        instance_count=1,
        volume_size=100,
        max_run=args.max_runtime_hours * 3600,
        use_spot_instances=args.spot,
        max_wait=(args.max_runtime_hours * 3600 + 3600) if args.spot else None,
        checkpoint_s3_uri=f"s3://{bucket}/checkpoints/{job_name}/" if args.spot else None,
        sagemaker_session=sagemaker.Session(boto_session=boto3.Session(region_name=region)),
        output_path=f"s3://{bucket}/runs/{job_name}/",
        base_job_name=project,
        entry_point="scripts/run_qwen3_gsm8k_mi.py",
        source_dir=staging_dir,
        environment=ajar_env,
        tags=[
            {"Key": "participant", "Value": handle},
            {"Key": "project", "Value": project},
            {"Key": "compute_path", "Value": "A"},
        ],
    )

    print(f"Submitting: {job_name}")
    print(f"  instance:  {args.instance}{' (spot)' if args.spot else ' (on-demand)'}")
    print(f"  samples:   {args.samples}  models: {args.models}  prompts: {args.prompts}")
    print(f"  parallel:  {args.parallel_gpus} GPU(s)"
          + (f"  allocations: {ajar_env['AJAR_MODEL_GPU_ALLOCATIONS']}" if args.parallel_gpus > 1 else ""))
    print(f"  max run:   {args.max_runtime_hours} hr")
    print(f"  source:    {staging_dir} (whitelisted paths only)")
    print(f"  output:    s3://{bucket}/runs/{job_name}/")
    estimator.fit(job_name=job_name, wait=not args.no_wait, logs="All" if not args.no_wait else None)
    if args.no_wait:
        print(f"\nSubmitted (async). Poll: aws sagemaker describe-training-job "
              f"--training-job-name {job_name} --region {region} --query TrainingJobStatus")
    else:
        print(f"\nDone. Pull: aws s3 sync s3://{bucket}/runs/{job_name}/ ./outputs/{job_name}/")


if __name__ == "__main__":
    main()
