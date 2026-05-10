# Session notes — SageMaker Bootstrap prompt test (2026-05-10)

**Branch:** `test-sagemaker-bootstrap-prompt` (do not merge to main — this is a prompt-test scratch branch).
**Reviewer:** read this file first, then the diff, then the meta-analysis at the bottom.

## What this branch is

A test run of an external "SageMaker Bootstrap" prompt against the AJAR repo (mech-interp on Qwen3-4B / GSM8K). The goal was to validate the prompt, not actually deploy AJAR on AWS. The agent (me, Claude Opus 4.7) was asked at the end to do a meta-analysis of where the work fell short if this had been a real deployment.

## What was attempted (chronological)

1. Stage 1 scan of the repo per the prompt's instructions.
2. Detected `~/.sagemaker.env` already existed; HF token gate failed twice because the participant's editor paste did not land on disk. Recovered by reading the token from the macOS clipboard via `pbpaste`.
3. Stage 2 substrate provisioning (idempotent): budget, S3 bucket, IAM role, lifecycle config.
4. Quota check: notebook quota for `ml.g6.xlarge` was already 4; training-job quota was 0 → submitted increase request.
5. After participant pushback on the prompt's default Notebook recommendation, pivoted to Path B (training jobs from laptop) — wrote the two files in this diff.
6. Meta-analysis at the participant's request (reproduced at the bottom of this file).

## What was created in AWS (account 506145782110)

- S3 bucket: `edward-506145-us-east-1` (us-east-1, blocked-public, tagged).
- IAM role: `arn:aws:iam::506145782110:role/AmazonSageMaker-edward` — uses `AmazonSageMakerFullAccess` (too broad; see gap #1 below).
- Budget: `edward-sagemaker-credit-burn`, $200/month, alerts at 25/50/75/100% to `eluecheelip@gmail.com`.
- Lifecycle config: `sagemaker-idle-shutdown` (60-min idle auto-stop, Path A only).
- Service-quota increase request `e1dd0d56…` for `L-56AE9D73` (ml.g6.xlarge for training-job usage), pending AWS approval.

## What was created in the repo

- `scripts/sagemaker_submit.py` — Python SDK launcher that submits a training job.
- `sagemaker_entrypoint.sh` — bash entrypoint that runs inside the container, installs pinned deps, calls existing `scripts/run_torch_mechanistic_slice.sh`.

Neither was smoke-tested. They are sketches, not verified artifacts.

## Meta-analysis: what would have been wrong in production

### Substrate is technically correct but operationally sloppy

| Gap | What breaks in real use |
|---|---|
| IAM role is `AmazonSageMakerFullAccess` | Should be least-privilege: S3 RW on one bucket, ECR pull, CloudWatch logs. As-is, role compromise = full SageMaker access. |
| No S3 lifecycle policy | Hidden-state `.npz` outputs accumulate forever → recurring storage cost. |
| No S3 versioning | Re-running same job name silently overwrites. |
| No CloudWatch log retention | Defaults to "Never Expire." |
| Budget alerts email-only | No SNS topic → no programmatic response (e.g., auto-stop at 90%). |
| Tags on bucket/role but not on training jobs | Cost Explorer cannot break down spend by project. |

### Launcher would likely fail on first submission

1. **Framework version mismatch.** Hardcoded `framework_version="2.3.0"` but AJAR pins `torch==2.8.0`. Container's pre-installed torch is 2.3; pip upgrade inside the job may conflict with image's CUDA libs.
2. **Entry-point type.** `PyTorch` estimator expects `.py`; I passed `.sh`. May work via shim but is against the grain.
3. **Wrong output channel.** `outputs/` is copied into `/opt/ml/model/`, which doubles egress when SageMaker tars and uploads. Should be `/opt/ml/output/data/`.
4. **No spot.** AJAR's runners are idempotent and resume-friendly — perfect spot candidates. Spot saves 60-70%. Should have been the default.

### Workload optimizations never raised

| Optimization | Approx impact |
|---|---|
| Pre-built Docker image with deps baked into ECR | Save ~3-5 min per job + reproducibility + fixes framework mismatch. |
| HF model cache in S3 (FastFile) | Save 30-60 sec model-load per job. Significant for sweeps. |
| `torch.compile` | 15-30% speedup typical on Qwen3-class architectures. |
| KV-cache reuse across prompt variants | AJAR runs same prefix with multiple suffixes — shared prefix is cacheable. |
| Larger batch dimension | g6.xlarge has 24 GB; Qwen3-4B fp16 is ~8 GB. Lots of slack. |
| `ml.g5.2xlarge` (A10G) instead of g6 | Often 1.5-2× faster on fp16; may be net cheaper per slice. |

### Strategic question never asked

This workload is a **parameter sweep on a fixed model**, not training. What varies is prompts/samples/signals. Training-job-per-slice pays model-load cost on every job. For sweep-heavy work, a long-running notebook or endpoint may be cheaper. For batch reruns of the full slice, training jobs are right. Stage 1 of the prompt should have surfaced this sweep-vs-batch distinction — it's the choice that actually matters.

### Process failures

- Asked the participant to run shell commands twice when the agent had its own shell. Lazy delegation; participant correctly objected.
- Let the HF token gate fail twice before checking the clipboard. The clipboard check should have been the first diagnostic.
- Never read `docs/cloud_gpu_setup.md` (mentioned in the repo README) — would likely have informed instance choice and setup details.
- **Never ran a smoke test.** Substrate + launcher were shipped without submitting a `--samples 1` test job to prove the role, S3 path, and entrypoint actually work end-to-end. This is the single most important miss.

### Net assessment

For a *prompt test*: succeeded — produced concrete failure modes for the prompt author to address (narrow defaults, Bedrock-on-eval false-positive, paste-into-comment trap, agent-delegation anti-pattern).

For a *real AJAR-on-SageMaker deployment*: ~40% of a working system. Substrate ~80%, launcher ~30%, workload optimization ~5%, strategic fit not assessed.

**Single highest-leverage fix for the prompt:** Stage 2 should end with a $0.10 smoke job that exercises role + S3 + entrypoint + output channel, and refuse to declare success until it returns green.

## Companion artifact

A feedback memory has been saved at `~/.claude/projects/-Users-edward-Projects-AJAR/memory/feedback_sagemaker_bootstrap_prompt_gaps.md` capturing the prompt-side issues separately (narrow defaults, Bedrock false-positive, paste trap, agent-delegation, missing optimization framing).
