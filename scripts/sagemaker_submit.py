"""Submit an AJAR torch MI slice as a SageMaker training job.

Usage:
    python scripts/sagemaker_submit.py                        # 50-sample slice, ml.g6.xlarge
    python scripts/sagemaker_submit.py --samples 100
    python scripts/sagemaker_submit.py --instance ml.g6.12xlarge --samples 200

Requirements:
    pip install sagemaker boto3
    Source ~/.sagemaker.env so HF_TOKEN, S3_BUCKET, AWS_REGION are set.
    Training-job quota for the chosen instance must be > 0 (request via Service Quotas).

Outputs land at s3://$S3_BUCKET/ajar-runs/<job-name>/output/model.tar.gz.
Logs stream to CloudWatch and to stdout while the job runs.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import sagemaker
from sagemaker.pytorch import PyTorch

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=50)
    p.add_argument("--instance", default="ml.g6.xlarge")
    p.add_argument("--max-runtime-hours", type=int, default=4)
    p.add_argument("--job-name", default=None)
    args = p.parse_args()

    bucket = os.environ["S3_BUCKET"]
    region = os.environ["AWS_REGION"]
    hf_token = os.environ["HF_TOKEN"]
    handle = os.environ["STUDENT_HANDLE"]
    role_arn = f"arn:aws:iam::{boto3_account_id()}:role/AmazonSageMaker-{handle}"

    session = sagemaker.Session(boto_session=_boto_session(region))
    job_name = args.job_name or f"ajar-mi-slice-{int(time.time())}"

    estimator = PyTorch(
        entry_point="sagemaker_entrypoint.sh",
        source_dir=str(REPO_ROOT),
        role=role_arn,
        instance_type=args.instance,
        instance_count=1,
        framework_version="2.3.0",
        py_version="py311",
        max_run=args.max_runtime_hours * 3600,
        sagemaker_session=session,
        output_path=f"s3://{bucket}/ajar-runs",
        base_job_name="ajar-mi-slice",
        environment={
            "HF_TOKEN": hf_token,
            "AJAR_SAMPLES": str(args.samples),
            "AJAR_BACKEND": "torch",
            "S3_BUCKET": bucket,
        },
        keep_alive_period_in_seconds=0,
    )
    estimator.fit(job_name=job_name, wait=True, logs="All")
    print(f"\nDone. Outputs: s3://{bucket}/ajar-runs/{job_name}/output/")


def _boto_session(region: str):
    import boto3
    return boto3.Session(region_name=region)


def boto3_account_id() -> str:
    import boto3
    return boto3.client("sts").get_caller_identity()["Account"]


if __name__ == "__main__":
    main()
