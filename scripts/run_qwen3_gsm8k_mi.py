#!/usr/bin/env python
"""
Mechanistic-interpretability runner for:
  - Qwen/Qwen3-4B-Thinking-2507
  - Qwen/Qwen3-4B-Instruct-2507

It evaluates the first N GSM8K examples with two prompts, extracts detailed
generation/activation/attention artifacts, scores sentence-level "thought
anchors", and reruns counterfactual continuations with targeted suppression.
"""

from __future__ import annotations

import csv
import dataclasses
import importlib
import json
import math
import multiprocessing as mp
import os
import platform
import random
import queue
import re
import threading
import time
import urllib.error
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

# Apple's system Python ships with LibreSSL, which urllib3>=2 warns about on
# every import. The warning is informational and out of our control here, so
# we silence it before urllib3 gets imported transitively.
warnings.filterwarnings("ignore", message=r".*LibreSSL.*", module="urllib3.*")
warnings.filterwarnings("ignore", category=Warning, module=r"urllib3\..*")

# Hard-imported deps. These are required by every code path; if any of them
# is missing the right answer is to fail at startup with a single clear
# message, not to limp along and crash later inside a worker.
import numpy as np
from tqdm.auto import tqdm

# Backend-specific deps (torch, transformers, datasets) are imported through
# require_backend_dependencies() so that an oMLX-only user does not need a
# torch install. The names are pre-declared here for type-checkers.
if TYPE_CHECKING:  # pragma: no cover
    import torch  # noqa: F401
    import torch.nn.functional as F  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
torch: Any = None
F: Any = None
load_dataset: Any = None
AutoModelForCausalLM: Any = None
AutoTokenizer: Any = None


_BACKEND_DEPS: Dict[str, List[str]] = {
    "omlx": [],
    "torch": ["torch", "transformers", "datasets"],
}


def require_backend_dependencies(backend: str) -> None:
    """Hard-import the dependencies this backend actually needs.

    Called once at startup. Missing deps produce a single multi-line error
    explaining how to recover, rather than a cryptic AttributeError later.
    """
    global torch, F, load_dataset, AutoModelForCausalLM, AutoTokenizer
    needed = _BACKEND_DEPS.get(backend)
    if needed is None:
        raise ValueError(
            f"Unknown AJAR_BACKEND={backend!r}. Valid backends: {sorted(_BACKEND_DEPS)}."
        )

    # `datasets` is needed in both backends because GSM8K is loaded through it
    # whenever a custom GSM8K_JSONL fixture is not supplied. We only flag it
    # as required when no fixture is present; that check happens in
    # load_gsm8k_examples. We optimistically import it here for the torch
    # backend so type-check assertions hold; for omlx we import lazily.
    missing: List[str] = []
    for module_name in needed:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)
    if missing:
        raise SystemExit(
            "Missing dependencies for AJAR_BACKEND={backend}: {missing}.\n"
            "Recover with one of:\n"
            "  pip install -r requirements.txt\n"
            "  pip install {missing_args}\n"
            "If you are on Apple Silicon and want oMLX-only runs, set "
            "AJAR_BACKEND=omlx and the torch deps are not needed.".format(
                backend=backend,
                missing=", ".join(missing),
                missing_args=" ".join(missing),
            )
        )

    if backend == "torch":
        import torch as _torch
        import torch.nn.functional as _F
        from datasets import load_dataset as _load_dataset
        from transformers import (
            AutoModelForCausalLM as _AutoModelForCausalLM,
            AutoTokenizer as _AutoTokenizer,
        )
        torch = _torch
        F = _F
        load_dataset = _load_dataset
        AutoModelForCausalLM = _AutoModelForCausalLM
        AutoTokenizer = _AutoTokenizer
    else:
        # oMLX path: try to import datasets lazily so HF GSM8K loading still
        # works, but do not hard-fail; the runner falls back to GSM8K_JSONL.
        try:
            from datasets import load_dataset as _load_dataset
            load_dataset = _load_dataset
        except Exception:
            load_dataset = None


PROMPTS = {
    "explicit_cot": lambda q: [
        {
            "role": "system",
            "content": (
                "Think through this step by step showing your reasoning process "
                "then provide your final answer. Put the final numeric answer in "
                "\\boxed{}."
            ),
        },
        {"role": "user", "content": f"Question: {q}"},
    ],
    "explicit_no_cot": lambda q: [
        {
            "role": "system",
            "content": (
                "Answer-only mode. /no_think\n"
                "Output exactly one line: \\boxed{number}. Do not include reasoning, "
                "explanations, equations, restatements, or units."
            ),
        },
        {"role": "user", "content": f"Question: {q}\n\n/no_think\nRespond only with \\boxed{{number}}."},
    ],
    "neutral": lambda q: [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the question directly, with "
                "your final numerical answer in \\boxed{}."
            ),
        },
        {"role": "user", "content": f"Question: {q}"},
    ],
    # Matches the neutral prompt as written in the AJAR proposal verbatim. Use this
    # for HCDS so the Neutral condition is not contaminated by directives that bias
    # toward CoT or no-CoT. Answer extraction falls back to "last number in text"
    # when no \\boxed{} is present.
    "neutral_strict": lambda q: [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Question: {q}"},
    ],
}


INFERENCE_BACKEND = os.environ.get("AJAR_BACKEND", "omlx").strip().lower()


def parse_csv_env(name: str, default: Sequence[str]) -> List[str]:
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    return [part.strip() for part in raw.split(",") if part.strip()]


HF_MODEL_SPECS = {
    "thinking": os.environ.get("AJAR_HF_MODEL_THINKING", "Qwen/Qwen3-4B-Thinking-2507"),
    "instruct": os.environ.get("AJAR_HF_MODEL_INSTRUCT", "Qwen/Qwen3-4B-Instruct-2507"),
    # Cross-family torch path: needed only to capture token entropy (no logprobs
    # over oMLX). Run baseline-only (AJAR_RUN_MI unset); the Qwen-specific anchor
    # intervention is not used for this model.
    # unsloth mirror is an ungated re-upload of google/gemma-3-4b-it (the google
    # repo is license-gated); weights are identical for entropy purposes.
    "gemma": os.environ.get("AJAR_HF_MODEL_GEMMA", "unsloth/gemma-3-4b-it"),
}


OMLX_MODEL_SPECS = {
    "thinking": "Qwen3-4B-Thinking-2507-MLX-8bit",
    "instruct": "Qwen3-4B-Instruct-2507-MLX-8bit",
    # Cross-family replication (PI/reviewer ask). Gemma is instruction-tuned
    # with no reasoning variant, so it replicates the validated Instruct case.
    # oMLX is baseline-only, so this yields the 5 behavioral/linguistic
    # features; the torch anchor intervention (6th feature) is not run.
    "gemma": os.environ.get("AJAR_OMLX_MODEL_GEMMA", "gemma-4-E4B-it-MLX-4bit"),
}


MODEL_SPECS = OMLX_MODEL_SPECS if INFERENCE_BACKEND == "omlx" else HF_MODEL_SPECS


INTERVENTION_SPECS = {
    "residual_zero": ("residual", 0.0),
    "residual_scale_0.25": ("residual", 0.25),
    "attention_zero": ("attention", 0.0),
    "attention_scale_0.25": ("attention", 0.25),
}


# Where all outputs and artifacts are written.
OUTPUT_DIR = Path(os.environ.get("AJAR_OUTPUT_DIR", "outputs/qwen3_gsm8k_mi"))
# GSM8K split to evaluate.
DATASET_SPLIT = "test"
# Number of GSM8K examples to process from the start of the split.
NUM_SAMPLES = int(os.environ.get("AJAR_NUM_SAMPLES", "3" if INFERENCE_BACKEND == "omlx" else "500"))
# Random seed for control-step selection and reproducibility.
SEED = 17
# Which model variants to run. Valid values: "thinking", "instruct".
MODELS_TO_RUN = parse_csv_env("AJAR_MODELS", ["thinking", "instruct"])
# Which prompt variants to run. Valid values: "explicit_cot", "explicit_no_cot", "neutral".
PROMPTS_TO_RUN = parse_csv_env("AJAR_PROMPTS", ["explicit_cot", "explicit_no_cot", "neutral"])
# Maximum number of new tokens generated per completion.
MAX_NEW_TOKENS = int(os.environ.get("AJAR_MAX_NEW_TOKENS", "256" if INFERENCE_BACKEND == "omlx" else "512"))
# Interventions only need to run long enough to emit a final answer after the
# intervention point. The full baseline budget is overkill: with a typical
# anchor step around mid-reasoning, the model usually reaches the boxed answer
# within 256-512 new tokens. Setting this lower than MAX_NEW_TOKENS cuts
# intervention wall time roughly proportionally on Thinking-class outputs.
INTERVENTION_MAX_NEW_TOKENS = int(
    os.environ.get("AJAR_INTERVENTION_MAX_NEW_TOKENS", "384")
)
# Whether to sample during generation. Keep False for deterministic runs.
DO_SAMPLE = False
# Sampling temperature when DO_SAMPLE is True.
TEMPERATURE = 0.0
# Nucleus sampling threshold when DO_SAMPLE is True.
TOP_P = 1.0
# Top-k sampling cutoff when DO_SAMPLE is True.
TOP_K = 0
# Beam count for generation. Keep at 1 for standard decoding.
NUM_BEAMS = 1
# Precision mode. "auto" picks fp16 on MPS (bf16 on MPS is unreliable in
# torch 2.x), bf16 on bf16-capable CUDA GPUs, fp16 on older CUDA, fp32 on CPU.
# Override with AJAR_DTYPE={float32,float16,bfloat16,auto}.
DTYPE = os.environ.get("AJAR_DTYPE", "float32" if INFERENCE_BACKEND == "omlx" else "auto")
# Attention backend. "eager" is best for attention probing reliability.
ATTN_IMPLEMENTATION = "eager"
# Device placement passed into Hugging Face model loading for single-process fallback.
DEVICE_MAP = "auto"
# Whether to use one worker process per configured GPU. The default is False
# even on torch because most contributors run on a single Mac/CUDA GPU.
# Multi-GPU clusters opt in via AJAR_PARALLEL=1.
PARALLELIZE_ACROSS_GPUS = os.environ.get("AJAR_PARALLEL", "0") == "1"
# Which GPU ids each model is allowed to use; the global worker pool steals work across them.
MODEL_GPU_ALLOCATIONS = {
    "thinking": [0, 1, 2, 3],
    "instruct": [4, 5, 6, 7],
    "gemma": [0],
}
# CPU thread cap per worker so 8 workers do not oversubscribe the host.
OMP_NUM_THREADS_PER_WORKER = 4
# Enable this only if the target model requires remote code.
TRUST_REMOTE_CODE = False
# Maximum number of answer-token probes per sample. Step-end probes are
# always retained exhaustively (one per reasoning step). Answer probes are
# sampled when the answer span is long; the downstream metrics average
# across answer probes, so a 64-probe sample is statistically equivalent
# to probing every answer token while keeping the per-sample numpy array
# under ~1 GB instead of ~12 GB on long Thinking outputs.
MAX_ANSWER_PROBES = int(os.environ.get("AJAR_MAX_ANSWER_PROBES", "64"))
# Number of highest-scoring anchor steps to intervene on per sample.
TOP_ANCHOR_STEPS = int(os.environ.get("AJAR_TOP_ANCHOR_STEPS", "3"))
# Number of random non-anchor control steps to intervene on per sample.
NUM_CONTROL_STEPS = int(os.environ.get("AJAR_NUM_CONTROL_STEPS", "2"))
# Number of top-ranked layers to target for each intervention.
TOP_ANCHOR_LAYERS = int(os.environ.get("AJAR_TOP_ANCHOR_LAYERS", "4"))
# Which intervention types to run for each chosen step.
INTERVENTIONS_TO_RUN = parse_csv_env("AJAR_INTERVENTIONS", list(INTERVENTION_SPECS.keys()))
# Save the full hidden state tensors for every baseline sample.
SAVE_FULL_HIDDEN_STATES = False
# Save the full probe-attention tensors for every baseline sample. Off by
# default because at 50+ samples this directory can balloon past 50GB; flip
# AJAR_SAVE_FULL_PROBE_ATTENTION=1 to archive raw tensors. Per-layer summary
# attention is still saved in step_scores either way.
SAVE_FULL_PROBE_ATTENTION = os.environ.get("AJAR_SAVE_FULL_PROBE_ATTENTION", "0") == "1"
# Local oMLX exposes an OpenAI-compatible chat completions API. It supports
# text generation, but not PyTorch hidden-state/attention probing or hooks.
OMLX_BASE_URL = os.environ.get("OMLX_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
OMLX_CONCURRENCY = max(1, int(os.environ.get("AJAR_OMLX_CONCURRENCY", "1")))
RUN_MECHANISTIC_ANALYSIS = os.environ.get(
    "AJAR_RUN_MI",
    "0" if INFERENCE_BACKEND == "omlx" else "1",
) == "1"
ALLOW_SMOKE_DATASET = os.environ.get("AJAR_ALLOW_SMOKE_DATASET", "0") == "1"
# Skip a sample if its baseline artifacts already exist.
RESUME = True
# Skip hidden-state and attention analysis above this total token length.
ANALYSIS_MAX_SEQ_LEN = 2048
# Emit timestamped logs throughout the run.
VERBOSE_LOGGING = True
# Print a log line before and after each baseline generation.
LOG_BASELINE_GENERATION = True
# Print a log line before and after each hidden-state and attention analysis stage.
LOG_ANALYSIS_STAGES = True
# Print a log line before and after each intervention generation.
LOG_INTERVENTIONS = True
# Update progress-bar postfixes with live counters.
SHOW_PROGRESS_POSTFIX = True
# Emit periodic "still working" heartbeats during long blocking stages.
ENABLE_STAGE_HEARTBEAT = True
# Seconds between heartbeat messages for long stages.
HEARTBEAT_INTERVAL_SECONDS = 15


NUMERIC_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def resolve_omlx_api_key() -> str:
    env_key = os.environ.get("OMLX_API_KEY")
    if env_key:
        return env_key
    settings_path = Path.home() / ".omlx" / "settings.json"
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        return str(settings.get("auth", {}).get("api_key", ""))
    except Exception:
        return ""


LOCAL_SMOKE_GSM8K = [
    {
        "question": (
            "Natalia sold clips to 48 of her friends in April, and then she sold "
            "half as many clips in May. How many clips did Natalia sell altogether "
            "in April and May?"
        ),
        "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
        "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
    },
    {
        "question": (
            "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 "
            "minutes of babysitting. How much did she earn?"
        ),
        "answer": "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\n"
        "Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10",
    },
    {
        "question": (
            "Betty is saving money for a new wallet which costs $100. Betty has "
            "only half of the money she needs. Her parents decided to give her $15 "
            "for that purpose, and her grandparents twice as much as her parents. "
            "How much more money does Betty need to buy the wallet?"
        ),
        "answer": "Betty has 100 / 2 = $<<100/2=50>>50.\n"
        "Her grandparents gave 15 * 2 = $<<15*2=30>>30.\n"
        "She has 50 + 15 + 30 = $<<50+15+30=95>>95.\n"
        "She needs 100 - 95 = $<<100-95=5>>5 more.\n#### 5",
    },
]


@dataclass
class ExperimentConfig:
    inference_backend: str
    output_dir: Path
    split: str
    num_samples: int
    seed: int
    models: List[str]
    prompts: List[str]
    max_new_tokens: int
    intervention_max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int
    num_beams: int
    dtype: str
    attn_implementation: str
    device_map: str
    parallelize_across_gpus: bool
    model_gpu_allocations: Dict[str, List[int]]
    omp_num_threads_per_worker: int
    trust_remote_code: bool
    max_answer_probes: int
    top_anchor_steps: int
    num_control_steps: int
    top_anchor_layers: int
    interventions: List[str]
    save_full_hidden_states: bool
    save_full_probe_attention: bool
    resume: bool
    analysis_max_seq_len: int
    verbose_logging: bool
    log_baseline_generation: bool
    log_analysis_stages: bool
    log_interventions: bool
    show_progress_postfix: bool
    enable_stage_heartbeat: bool
    heartbeat_interval_seconds: int
    omlx_base_url: str
    omlx_api_key: str
    omlx_concurrency: int
    run_mechanistic_analysis: bool
    allow_smoke_dataset: bool


@dataclass
class StepSpan:
    step_index: int
    text: str
    char_start_generated: int
    char_end_generated: int
    full_token_positions: List[int]
    generated_token_positions: List[int]


def get_config() -> ExperimentConfig:
    unknown_models = [model_key for model_key in MODELS_TO_RUN if model_key not in MODEL_SPECS]
    if unknown_models:
        raise ValueError(f"Unknown AJAR_MODELS entries: {unknown_models}. Valid models: {sorted(MODEL_SPECS)}")
    unknown_prompts = [prompt_name for prompt_name in PROMPTS_TO_RUN if prompt_name not in PROMPTS]
    if unknown_prompts:
        raise ValueError(f"Unknown AJAR_PROMPTS entries: {unknown_prompts}. Valid prompts: {sorted(PROMPTS)}")
    return ExperimentConfig(
        inference_backend=INFERENCE_BACKEND,
        output_dir=OUTPUT_DIR,
        split=DATASET_SPLIT,
        num_samples=NUM_SAMPLES,
        seed=SEED,
        models=MODELS_TO_RUN,
        prompts=PROMPTS_TO_RUN,
        max_new_tokens=MAX_NEW_TOKENS,
        intervention_max_new_tokens=INTERVENTION_MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        num_beams=NUM_BEAMS,
        dtype=DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION,
        device_map=DEVICE_MAP,
        parallelize_across_gpus=PARALLELIZE_ACROSS_GPUS,
        model_gpu_allocations=MODEL_GPU_ALLOCATIONS,
        omp_num_threads_per_worker=OMP_NUM_THREADS_PER_WORKER,
        trust_remote_code=TRUST_REMOTE_CODE,
        max_answer_probes=MAX_ANSWER_PROBES,
        top_anchor_steps=TOP_ANCHOR_STEPS,
        num_control_steps=NUM_CONTROL_STEPS,
        top_anchor_layers=TOP_ANCHOR_LAYERS,
        interventions=INTERVENTIONS_TO_RUN,
        save_full_hidden_states=SAVE_FULL_HIDDEN_STATES,
        save_full_probe_attention=SAVE_FULL_PROBE_ATTENTION,
        resume=RESUME,
        analysis_max_seq_len=ANALYSIS_MAX_SEQ_LEN,
        verbose_logging=VERBOSE_LOGGING,
        log_baseline_generation=LOG_BASELINE_GENERATION,
        log_analysis_stages=LOG_ANALYSIS_STAGES,
        log_interventions=LOG_INTERVENTIONS,
        show_progress_postfix=SHOW_PROGRESS_POSTFIX,
        enable_stage_heartbeat=ENABLE_STAGE_HEARTBEAT,
        heartbeat_interval_seconds=HEARTBEAT_INTERVAL_SECONDS,
        omlx_base_url=OMLX_BASE_URL,
        omlx_api_key=resolve_omlx_api_key(),
        omlx_concurrency=OMLX_CONCURRENCY,
        run_mechanistic_analysis=RUN_MECHANISTIC_ANALYSIS,
        allow_smoke_dataset=ALLOW_SMOKE_DATASET,
    )


class RunLogger:
    def __init__(self, enabled: bool, log_path: Optional[Path] = None) -> None:
        self.enabled = enabled
        self.log_path = log_path
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        if not self.enabled:
            return
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        if hasattr(tqdm, "write"):
            tqdm.write(line)
        else:  # pragma: no cover
            print(line)
        if self.log_path is not None:
            try:
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                # Never let logging take down a run.
                pass


class QueueLogger(RunLogger):
    def __init__(self, enabled: bool, event_queue: mp.queues.Queue, worker_name: str) -> None:
        super().__init__(enabled)
        self.event_queue = event_queue
        self.worker_name = worker_name

    def log(self, message: str) -> None:
        if not self.enabled:
            return
        self.event_queue.put(
            {
                "type": "log",
                "worker": self.worker_name,
                "message": message,
            }
        )


def log_environment_banner(args: "ExperimentConfig", logger: "RunLogger") -> None:
    logger.log(
        "AJAR runner starting | python "
        f"{platform.python_version()} | platform {platform.platform()} | "
        f"backend={args.inference_backend} | output_dir={args.output_dir}"
    )
    logger.log(
        f"Config: models={args.models} prompts={args.prompts} "
        f"num_samples={args.num_samples} max_new_tokens={args.max_new_tokens} "
        f"dtype={args.dtype} resume={args.resume}"
    )
    if args.inference_backend == "torch":
        logger.log(
            f"Torch flags: top_anchor_steps={args.top_anchor_steps} "
            f"num_control_steps={args.num_control_steps} "
            f"top_anchor_layers={args.top_anchor_layers} "
            f"interventions={args.interventions} "
            f"save_full_probe_attention={args.save_full_probe_attention}"
        )
    elif args.inference_backend == "omlx":
        logger.log(f"oMLX flags: base_url={args.omlx_base_url} concurrency={args.omlx_concurrency}")


class StageHeartbeat:
    def __init__(self, logger: RunLogger, enabled: bool, label: str, interval_seconds: int) -> None:
        self.logger = logger
        self.enabled = enabled
        self.label = label
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time = 0.0

    def __enter__(self) -> "StageHeartbeat":
        if not self.enabled:
            return self
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            elapsed = time.time() - self._start_time
            self.logger.log(f"Still working on {self.label} after {elapsed:.1f}s.")

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_dtype(dtype_name: str) -> "torch.dtype":
    if torch is None:
        raise RuntimeError("PyTorch is required for AJAR_BACKEND=torch.")
    name = (dtype_name or "auto").lower()
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if name != "auto":
        raise ValueError(
            f"Unknown AJAR_DTYPE={dtype_name!r}. Valid: float32, float16, bfloat16, auto."
        )
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        # MPS bf16 support is partial in torch 2.x; fp16 is the safe pick.
        return torch.float16
    return torch.float32


def make_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if torch is not None and isinstance(value, torch.dtype):
        return str(value)
    if np is not None and isinstance(value, (np.floating, np.integer)):
        return value.item()
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return [make_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: make_jsonable(v) for k, v in value.items()}
    return value


def redacted_config(args: ExperimentConfig) -> Dict[str, Any]:
    payload = dict(args.__dict__)
    if payload.get("omlx_api_key"):
        payload["omlx_api_key"] = "***"
    return payload


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(make_jsonable(payload), f, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(make_jsonable(payload), ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(make_jsonable(row), ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: make_jsonable(row.get(k)) for k in fieldnames})


def normalize_number_str(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    raw = raw.strip().replace(",", "").replace("$", "").replace("%", "")
    raw = raw.rstrip(".")
    if not raw:
        return None
    try:
        value = float(raw)
    except Exception:
        return None
    if math.isfinite(value) and abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.10f}".rstrip("0").rstrip(".")


def extract_boxed_content(text: str) -> Optional[str]:
    anchor = text.find("\\boxed{")
    if anchor < 0:
        return None
    i = anchor + len("\\boxed{")
    depth = 1
    chars: List[str] = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        chars.append(ch)
        i += 1
    return None


_ANSWER_LABEL_RE = re.compile(
    r"(?:final\s+answer|the\s+answer\s+is|answer)\s*[:=]\s*([-+]?\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)


# StrategyQA mode: when AJAR_DATASET_KIND=strategyqa is set, the runner
# treats answers as yes/no rather than numeric. Gold-answer rationales
# end in `#### yes` / `#### no`; predictions are extracted from the
# generated text by looking for the strongest yes/no signal.
_DATASET_KIND = os.environ.get("AJAR_DATASET_KIND", "gsm8k").strip().lower()
_YESNO_LABEL_RE = re.compile(
    r"(?:final\s+answer|the\s+answer\s+is|answer)\s*[:=]?\s*"
    r"(yes|no|true|false)\b",
    re.IGNORECASE,
)
_YESNO_BOXED_RE = re.compile(r"^\s*(yes|no|true|false)\s*\.?\s*$", re.IGNORECASE)
# Some prompts (e.g. explicit_no_cot which says "put final numeric
# answer in \boxed{}") force the model to express yes/no as 0/1.
# Across all StrategyQA explicit_no_cot outputs in our 2026-05-06 run
# the convention was perfectly consistent: 0=no, 1=yes.
_YESNO_BOXED_BINARY_RE = re.compile(r"^\s*([01])\s*\.?\s*$")


def _normalize_yesno(token: str | None) -> Optional[str]:
    if token is None:
        return None
    t = token.strip().lower().rstrip(".")
    if t in ("yes", "true", "y", "1"):
        return "yes"
    if t in ("no", "false", "n", "0"):
        return "no"
    return None


def extract_predicted_yesno(text: str) -> Tuple[Optional[str], Optional[str]]:
    """StrategyQA analog of extract_predicted_number.

    Returns (prediction, boxed_content). prediction is "yes" / "no" /
    None. boxed_content is the literal contents of \\boxed{} if any.

    Recognised forms (in priority order):
      * ``\\boxed{yes}`` / ``\\boxed{no}`` (literal verdict)
      * ``\\boxed{0}`` / ``\\boxed{1}`` — the model's binary mapping
        when the prompt forces a numeric ``\\boxed{}`` answer (0=no, 1=yes)
      * ``Final answer: yes`` / ``The answer is no`` (explicit label)
      * Last yes/no token in the post-think answer text (fallback)
    """
    boxed = extract_boxed_content(text)
    if boxed is not None:
        bm = _YESNO_BOXED_RE.match(boxed)
        if bm:
            return _normalize_yesno(bm.group(1)), boxed
        bm01 = _YESNO_BOXED_BINARY_RE.match(boxed)
        if bm01:
            return ("yes" if bm01.group(1) == "1" else "no"), boxed

    answer_text = _post_think_text(text)

    # Prefer an explicit "answer: yes" template if present (last match).
    label_match = None
    for label_match in _YESNO_LABEL_RE.finditer(answer_text):
        pass
    if label_match is not None:
        return _normalize_yesno(label_match.group(1)), boxed

    # Last resort: scan for the LAST yes/no token in the post-think
    # answer text. StrategyQA chains often discuss yes/no scenarios
    # within the rationale; we want the model's terminal verdict.
    last_match = None
    for m in re.finditer(r"\b(yes|no)\b", answer_text, re.IGNORECASE):
        last_match = m
    if last_match is not None:
        return _normalize_yesno(last_match.group(1)), boxed

    return None, boxed


def parse_strategyqa_answer(answer: str) -> Optional[str]:
    """Parse '#### yes' / '#### no' from a StrategyQA-shaped gold
    answer string. Falls back to scanning for yes/no anywhere if the
    canonical line is missing."""
    match = re.search(r"####\s*([^\n]+)", answer)
    if match:
        return _normalize_yesno(match.group(1))
    last_match = None
    for m in re.finditer(r"\b(yes|no)\b", answer, re.IGNORECASE):
        last_match = m
    return _normalize_yesno(last_match.group(1)) if last_match else None


def _post_think_text(text: str) -> str:
    """Return the text after </think> if a thinking block is present.

    Qwen3-Thinking models emit reasoning inside <think>...</think> followed by
    the actual answer. Numeric extraction must operate on the answer portion
    or the fallback "last number" rule will pick up internal scratch work.
    """
    close_idx = text.rfind("</think>")
    if close_idx < 0:
        return text
    return text[close_idx + len("</think>") :]


def extract_predicted_number(text: str) -> Tuple[Optional[str], Optional[str]]:
    boxed = extract_boxed_content(text)
    if boxed is not None:
        match = NUMERIC_RE.search(boxed)
        if match:
            return normalize_number_str(match.group(0)), boxed

    answer_text = _post_think_text(text)

    # Prefer an explicit "answer: X" / "final answer = X" template; this is
    # what unprompted neutral runs often produce and is much more reliable
    # than "last number in text".
    label_match = None
    for label_match in _ANSWER_LABEL_RE.finditer(answer_text):
        pass  # take the last one
    if label_match is not None:
        return normalize_number_str(label_match.group(1)), boxed

    matches = list(NUMERIC_RE.finditer(answer_text))
    if not matches:
        return None, boxed
    return normalize_number_str(matches[-1].group(0)), boxed


def parse_gsm8k_answer(answer: str) -> Optional[str]:
    if _DATASET_KIND == "strategyqa":
        return parse_strategyqa_answer(answer)
    match = re.search(r"####\s*([^\n]+)", answer)
    if not match:
        fallback = NUMERIC_RE.findall(answer)
        return normalize_number_str(fallback[-1]) if fallback else None
    numeric = NUMERIC_RE.search(match.group(1))
    return normalize_number_str(numeric.group(0)) if numeric else None


# Re-bind extract_predicted_number when the dataset is StrategyQA so all
# call sites in the runner pick up yes/no extraction without needing
# per-call dispatch. The two functions return the same (prediction,
# boxed_content) tuple shape.
if _DATASET_KIND == "strategyqa":
    _extract_predicted_number_numeric = extract_predicted_number  # noqa: F841
    extract_predicted_number = extract_predicted_yesno  # type: ignore[assignment]


def get_model_input_device(model: torch.nn.Module) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def get_decoder_layers(model: torch.nn.Module) -> Sequence[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    # Multimodal wrappers (e.g. Gemma3ForConditionalGeneration) nest the text
    # decoder under a language_model submodule. Reach it explicitly so we hook the
    # text stack, NOT the vision tower's own .layers (model.vision_tower.encoder.layers).
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(getattr(inner, "language_model", None), "layers"):
        return inner.language_model.layers
    if hasattr(getattr(model, "language_model", None), "layers"):
        return model.language_model.layers
    raise ValueError("Could not locate decoder layers for hook registration.")


def sanitize_worker_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def get_worker_name(model_key: str, gpu_id: int, slot_index: int) -> str:
    return sanitize_worker_name(f"{model_key}_gpu{gpu_id}_slot{slot_index}")


def get_worker_output_paths(run_root: Path, worker_name: str) -> Dict[str, Path]:
    worker_dir = run_root / "workers"
    return {
        "dir": worker_dir,
        "baseline_jsonl": worker_dir / f"{worker_name}_baselines.jsonl",
        "intervention_jsonl": worker_dir / f"{worker_name}_interventions.jsonl",
        "baseline_csv": worker_dir / f"{worker_name}_baseline_summary.csv",
        "intervention_csv": worker_dir / f"{worker_name}_intervention_summary.csv",
    }


def configure_worker_environment(args: ExperimentConfig, gpu_id: Optional[int]) -> None:
    os.environ["OMP_NUM_THREADS"] = str(args.omp_num_threads_per_worker)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)


def build_jobs(
    dataset: Sequence[Dict[str, Any]],
    args: ExperimentConfig,
    run_root: Path,
) -> Tuple[Dict[str, List[Dict[str, Any]]], int]:
    jobs_by_model: Dict[str, List[Dict[str, Any]]] = {model_key: [] for model_key in args.models}
    skipped_existing = 0
    for sample_idx, example in enumerate(dataset):
        for prompt_name in args.prompts:
            for model_key in args.models:
                if args.resume and is_sample_complete_for_resume(
                    run_root,
                    model_key,
                    prompt_name,
                    sample_idx,
                    expected_question=str(example["question"]),
                ):
                    skipped_existing += 1
                    continue
                job = {
                    "task_type": "baseline",
                    "model_key": model_key,
                    "model_name": MODEL_SPECS[model_key],
                    "sample_idx": sample_idx,
                    "prompt_name": prompt_name,
                    "question": example["question"],
                    "gold_answer_raw": example["answer"],
                }
                jobs_by_model[model_key].append(job)
    return jobs_by_model, skipped_existing


def validate_parallel_config(args: ExperimentConfig) -> None:
    missing = [model_key for model_key in args.models if model_key not in args.model_gpu_allocations]
    if missing:
        raise ValueError(f"Missing MODEL_GPU_ALLOCATIONS entries for: {missing}")
    for model_key in args.models:
        if not args.model_gpu_allocations[model_key]:
            raise ValueError(f"MODEL_GPU_ALLOCATIONS[{model_key!r}] must list at least one GPU id.")


def get_baseline_path(run_root: Path, model_key: str, prompt_name: str, sample_idx: int) -> Path:
    return run_root / model_key / prompt_name / f"sample_{sample_idx:04d}" / "baseline.json"


def get_intervention_manifest_path(run_root: Path, model_key: str, prompt_name: str, sample_idx: int) -> Path:
    return run_root / model_key / prompt_name / f"sample_{sample_idx:04d}" / "intervention_manifest.json"


def get_sample_dir(run_root: Path, model_key: str, prompt_name: str, sample_idx: int) -> Path:
    return run_root / model_key / prompt_name / f"sample_{sample_idx:04d}"


def is_sample_complete_for_resume(
    run_root: Path,
    model_key: str,
    prompt_name: str,
    sample_idx: int,
    expected_question: Optional[str] = None,
) -> bool:
    baseline_path = get_baseline_path(run_root, model_key, prompt_name, sample_idx)
    if not baseline_path.exists():
        return False
    if expected_question is not None:
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if baseline.get("question") != expected_question:
            return False
    manifest_path = get_intervention_manifest_path(run_root, model_key, prompt_name, sample_idx)
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    expected_files = manifest.get("expected_intervention_files", [])
    sample_dir = get_sample_dir(run_root, model_key, prompt_name, sample_idx)
    for rel_path in expected_files:
        if not (sample_dir / rel_path).exists():
            return False
    return True


def resolve_gpu_allocations(
    args: ExperimentConfig,
    detected_gpu_count: int,
) -> Tuple[bool, Dict[str, List[int]]]:
    resolved: Dict[str, List[int]] = {}
    use_parallel = bool(args.parallelize_across_gpus and detected_gpu_count > 1)
    for model_key in args.models:
        configured = args.model_gpu_allocations[model_key]
        if detected_gpu_count <= 1:
            resolved[model_key] = [0]
            continue
        usable = [gpu_id for gpu_id in configured if gpu_id < detected_gpu_count]
        if not usable:
            raise ValueError(
                f"No configured GPU ids for model {model_key!r} are available on this machine. "
                f"Detected GPUs: 0..{detected_gpu_count - 1}, configured: {configured}"
            )
        resolved[model_key] = usable if use_parallel else [usable[0]]
    return use_parallel, resolved


def merge_jsonl_files(source_paths: Sequence[Path], destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("w", encoding="utf-8") as out:
        for source_path in source_paths:
            if not source_path.exists():
                continue
            with source_path.open("r", encoding="utf-8") as inp:
                for line in inp:
                    out.write(line)


def merge_csv_files(source_paths: Sequence[Path], destination_path: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for source_path in source_paths:
        if not source_path.exists():
            continue
        with source_path.open("r", encoding="utf-8", newline="") as f:
            rows.extend(list(csv.DictReader(f)))
    write_csv(destination_path, rows)


def emit_event(event_queue: Any, event_type: str, **payload: Any) -> None:
    event_queue.put({"type": event_type, **payload})


def load_model_and_tokenizer(
    model_name: str,
    dtype: torch.dtype,
    device_map: str,
    attn_implementation: str,
    trust_remote_code: bool,
    device: Optional[str] = None,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    kwargs: Dict[str, Any] = {"torch_dtype": dtype, "trust_remote_code": trust_remote_code}
    if device is None:
        kwargs["device_map"] = device_map
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if device is not None:
        model.to(device)
    model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer, model


def build_messages(prompt_name: str, question: str) -> List[Dict[str, str]]:
    return PROMPTS[prompt_name](question)


def load_gsm8k_examples(args: ExperimentConfig, logger: RunLogger) -> List[Dict[str, Any]]:
    # A custom variant fixture (perturbations, paraphrases, etc.) takes precedence
    # over the canonical HF dataset so the runner can be reused for Task 2 / Task 9
    # without forking. The fixture must contain rows with "question" and "answer"
    # in the same shape as the GSM8K dataset.
    local_path = os.environ.get("GSM8K_JSONL")
    if local_path:
        path = Path(local_path).expanduser()
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "question" not in row or "answer" not in row:
                    raise ValueError(f"{path} rows must contain 'question' and 'answer'.")
                rows.append(row)
                if len(rows) >= args.num_samples:
                    break
        logger.log(f"Loaded {len(rows)} GSM8K row(s) from GSM8K_JSONL={path}.")
        return rows

    if load_dataset is not None:
        dataset = load_dataset("gsm8k", "main", split=f"{args.split}[:{args.num_samples}]")
        return [dict(row) for row in dataset]

    if args.allow_smoke_dataset:
        logger.log(
            "Python package 'datasets' is not installed and GSM8K_JSONL was not set; "
            "using the built-in 3-example GSM8K smoke set because AJAR_ALLOW_SMOKE_DATASET=1."
        )
        return LOCAL_SMOKE_GSM8K[: args.num_samples]

    raise RuntimeError(
        "Real GSM8K is not available in this Python environment. Install the 'datasets' "
        "package, or set GSM8K_JSONL=/path/to/gsm8k.jsonl. For a tiny smoke test only, "
        "set AJAR_ALLOW_SMOKE_DATASET=1."
    )


def build_prompt_text(tokenizer: AutoTokenizer, messages: List[Dict[str, str]]) -> str:
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = build_plain_prompt_text(messages)
    # Phase-1 no-CoT pole repair (NoThinking-style forced closure). The 2507
    # Thinking template unconditionally opens '<think>\n' and dropped the
    # /no_think soft switch, so instruction-level suppression cannot produce a
    # clean answer-only pole. When AJAR_FORCE_THINK_CLOSE=1 we close the block
    # in the prefix so generation begins after '</think>'.
    if os.environ.get("AJAR_FORCE_THINK_CLOSE") == "1" and text.rstrip().endswith("<think>"):
        # Faithful NoThinking (Ma et al. 2504.09858): a fabricated *completed*
        # thought, not an empty close. The empty close makes the model treat the
        # block as truncated and re-reason in the answer body; the dummy thought
        # makes it proceed straight to the answer (clean pole, median ~8 tokens).
        text = text.rstrip() + "\nOkay, I think I have finished thinking.\n</think>\n\n"
    return text


def build_plain_prompt_text(messages: Sequence[Dict[str, str]]) -> str:
    return "\n\n".join(f"{message['role'].upper()}: {message['content']}" for message in messages)


def generate_omlx_completion(
    model_name: str,
    messages: List[Dict[str, str]],
    args: ExperimentConfig,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature if args.do_sample else 0,
    }
    if args.do_sample:
        payload["top_p"] = args.top_p

    request = urllib.request.Request(
        f"{args.omlx_base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {args.omlx_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    start = time.time()
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach local oMLX at {args.omlx_base_url}. "
            "Start oMLX or set OMLX_BASE_URL/OMLX_API_KEY."
        ) from exc
    elapsed = time.time() - start
    data = json.loads(raw)
    choice = data["choices"][0]
    message = choice.get("message", {})
    generated_text = message.get("content") or ""
    return {
        "generated_text": generated_text,
        "generation_seconds": elapsed,
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage", {}),
        "raw_response_id": data.get("id"),
    }


def tokenize_to_device(tokenizer: AutoTokenizer, text: str, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(text, return_tensors="pt")
    device = get_model_input_device(model)
    return {k: v.to(device) for k, v in encoded.items()}


def generate_completion(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    args: ExperimentConfig,
) -> Dict[str, Any]:
    model_inputs = tokenize_to_device(tokenizer, prompt_text, model)
    generate_kwargs: Dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "num_beams": args.num_beams,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if args.do_sample:
        generate_kwargs.update(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
        )
    start = time.time()
    gen = model.generate(**model_inputs, **generate_kwargs)
    elapsed = time.time() - start
    prompt_len = int(model_inputs["input_ids"].shape[1])
    generated_ids = gen[0, prompt_len:].detach().cpu()
    full_ids = gen[0].detach().cpu()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {
        "prompt_len": prompt_len,
        "prompt_input_ids": model_inputs["input_ids"][0].detach().cpu(),
        "generated_ids": generated_ids,
        "full_ids": full_ids,
        "generated_text": generated_text,
        "generation_seconds": elapsed,
    }


def identify_reasoning_and_answer_spans(generated_text: str) -> Dict[str, Any]:
    reasoning_start = 0
    reasoning_end = len(generated_text)
    answer_start = 0
    answer_end = len(generated_text)

    think_open = generated_text.find("<think>")
    think_close = generated_text.find("</think>")
    if think_open >= 0 and think_close > think_open:
        reasoning_start = think_open + len("<think>")
        reasoning_end = think_close
        answer_start = think_close + len("</think>")
        while answer_start < len(generated_text) and generated_text[answer_start].isspace():
            answer_start += 1
        answer_end = len(generated_text)
    else:
        boxed_anchor = generated_text.find("\\boxed{")
        if boxed_anchor >= 0:
            reasoning_end = boxed_anchor
            answer_start = boxed_anchor

    return {
        "reasoning_char_span_generated": [reasoning_start, reasoning_end],
        "answer_char_span_generated": [answer_start, answer_end],
        "reasoning_text": generated_text[reasoning_start:reasoning_end].strip(),
        "answer_text": generated_text[answer_start:answer_end].strip(),
    }


def split_text_into_steps(reasoning_text: str, base_offset: int) -> List[Tuple[int, int, str]]:
    steps: List[Tuple[int, int, str]] = []
    cursor = 0
    for raw_line in reasoning_text.splitlines(keepends=True):
        line_start = cursor
        line_end = cursor + len(raw_line)
        cursor = line_end
        stripped = raw_line.strip()
        if not stripped:
            continue
        content_start = line_start + raw_line.find(stripped)
        sentence_spans = list(re.finditer(r".+?(?:[.!?](?=\s|$)|$)", stripped))
        if not sentence_spans:
            steps.append((base_offset + content_start, base_offset + content_start + len(stripped), stripped))
            continue
        for match in sentence_spans:
            frag = match.group(0).strip()
            if not frag:
                continue
            frag_start = content_start + match.start()
            frag_end = content_start + match.end()
            steps.append((base_offset + frag_start, base_offset + frag_end, frag))
    if not steps and reasoning_text.strip():
        stripped = reasoning_text.strip()
        start = reasoning_text.find(stripped)
        steps.append((base_offset + start, base_offset + start + len(stripped), stripped))
    return steps


def tokenize_full_text(tokenizer: AutoTokenizer, full_text: str) -> Dict[str, Any]:
    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    return {
        "input_ids": encoded["input_ids"],
        "offset_mapping": encoded["offset_mapping"],
        "special_tokens_mask": encoded["special_tokens_mask"],
        "tokens": tokenizer.convert_ids_to_tokens(encoded["input_ids"]),
    }


def positions_overlapping_char_span(
    offset_mapping: Sequence[Tuple[int, int]],
    special_tokens_mask: Sequence[int],
    char_start: int,
    char_end: int,
) -> List[int]:
    positions: List[int] = []
    for idx, ((start, end), is_special) in enumerate(zip(offset_mapping, special_tokens_mask)):
        if is_special or end <= start:
            continue
        if end <= char_start or start >= char_end:
            continue
        positions.append(idx)
    return positions


def build_step_spans(
    tokenizer: AutoTokenizer,
    prompt_text: str,
    generated_text: str,
    reasoning_char_span_generated: Sequence[int],
) -> Tuple[List[StepSpan], Dict[str, Any]]:
    full_text = prompt_text + generated_text
    prompt_char_len = len(prompt_text)
    tokenization = tokenize_full_text(tokenizer, full_text)
    step_specs = split_text_into_steps(
        generated_text[reasoning_char_span_generated[0] : reasoning_char_span_generated[1]],
        base_offset=reasoning_char_span_generated[0],
    )
    steps: List[StepSpan] = []
    for step_index, (gen_start, gen_end, step_text) in enumerate(step_specs):
        full_positions = positions_overlapping_char_span(
            tokenization["offset_mapping"],
            tokenization["special_tokens_mask"],
            prompt_char_len + gen_start,
            prompt_char_len + gen_end,
        )
        generated_positions = [
            pos
            for pos in full_positions
            if tokenization["offset_mapping"][pos][0] >= prompt_char_len
        ]
        steps.append(
            StepSpan(
                step_index=step_index,
                text=step_text,
                char_start_generated=gen_start,
                char_end_generated=gen_end,
                full_token_positions=full_positions,
                generated_token_positions=generated_positions,
            )
        )
    return steps, tokenization


def get_answer_token_positions(
    offset_mapping: Sequence[Tuple[int, int]],
    special_tokens_mask: Sequence[int],
    prompt_text: str,
    answer_char_span_generated: Sequence[int],
) -> List[int]:
    return positions_overlapping_char_span(
        offset_mapping,
        special_tokens_mask,
        len(prompt_text) + answer_char_span_generated[0],
        len(prompt_text) + answer_char_span_generated[1],
    )


def run_full_forward_hidden_analysis(
    model: AutoModelForCausalLM,
    full_ids: torch.Tensor,
    save_full_hidden_states: bool,
    sample_dir: Path,
    collect_attentions: bool = True,
) -> Dict[str, Any]:
    device = get_model_input_device(model)
    full_ids_device = full_ids.unsqueeze(0).to(device)
    with torch.inference_mode():
        outputs = model(
            input_ids=full_ids_device,
            attention_mask=torch.ones_like(full_ids_device),
            output_hidden_states=True,
            output_attentions=collect_attentions,
            use_cache=False,
        )
    logits = outputs.logits[0].detach().float().cpu()
    hidden_states = [h[0].detach().float().cpu() for h in outputs.hidden_states]
    stacked = torch.stack(hidden_states, dim=0)
    norm_by_layer_token = torch.linalg.norm(stacked, dim=-1).numpy()
    delta_norm_by_layer_token = torch.zeros_like(stacked[:, :, 0]).numpy()
    if stacked.shape[0] > 1:
        deltas = stacked[1:] - stacked[:-1]
        delta_norm_by_layer_token[1:] = torch.linalg.norm(deltas, dim=-1).numpy()

    shift_logits = logits[:-1]
    shift_labels = full_ids[1:].long()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logprobs = torch.gather(log_probs, 1, shift_labels.unsqueeze(-1)).squeeze(-1).numpy()
    aligned_logprobs = np.empty((full_ids.shape[0],), dtype=np.float32)
    aligned_logprobs[:] = np.nan
    aligned_logprobs[1:] = token_logprobs

    # Predictive-distribution entropy in nats per position. shift_logits[i] is the
    # distribution that predicts token i+1, so token_entropy[i+1] is the entropy of
    # the model's distribution over its prediction for token i+1. Position 0 has no
    # predictor and is left as NaN.
    probs = log_probs.exp()
    entropy_per_pred = -(probs * log_probs).sum(dim=-1).numpy()
    aligned_entropy = np.empty((full_ids.shape[0],), dtype=np.float32)
    aligned_entropy[:] = np.nan
    aligned_entropy[1:] = entropy_per_pred

    np.savez_compressed(
        sample_dir / "hidden_summary.npz",
        norm_by_layer_token=norm_by_layer_token.astype(np.float32),
        delta_norm_by_layer_token=delta_norm_by_layer_token.astype(np.float32),
        token_logprobs=aligned_logprobs.astype(np.float32),
        token_entropy=aligned_entropy.astype(np.float32),
    )
    if save_full_hidden_states:
        torch.save(
            {"hidden_states_fp16": stacked.to(torch.float16), "full_ids": full_ids},
            sample_dir / "hidden_states.pt",
        )

    # Hold the device-side attentions tuple by reference so the caller can
    # index into it for probing without paying for another forward pass. We
    # do not serialise the attentions to disk by default; the per-probe
    # slices written by probe_attention_vectors are already enough for the
    # downstream anchor scoring.
    raw_attentions = outputs.attentions if collect_attentions else None
    del outputs

    return {
        "num_layers_plus_embed": int(stacked.shape[0]),
        "seq_len": int(stacked.shape[1]),
        "hidden_dim": int(stacked.shape[2]),
        "norm_by_layer_token": norm_by_layer_token.astype(np.float32),
        "delta_norm_by_layer_token": delta_norm_by_layer_token.astype(np.float32),
        "token_logprobs": aligned_logprobs.astype(np.float32),
        "token_entropy": aligned_entropy.astype(np.float32),
        "raw_attentions": raw_attentions,
    }


def build_probe_plan(
    steps: Sequence[StepSpan],
    answer_token_positions: Sequence[int],
    max_answer_probes: int = 64,
) -> List[Dict[str, Any]]:
    """Build the probe plan for attention probing.

    Step-end probes are kept exhaustively (one per reasoning step) because
    each step is identified individually downstream. Answer probes are
    sampled when the answer span is long: a 1500-token answer span produces
    1500 answer probes which then materialise as a 12+ GB numpy array
    in probe_attention_vectors. The downstream consumer averages over all
    answer probes anyway, so a sampled subset of size max_answer_probes
    yields a statistically equivalent mean while keeping memory bounded.
    """
    probes: List[Dict[str, Any]] = []
    for step in steps:
        if step.full_token_positions:
            probes.append(
                {
                    "probe_type": "step_end",
                    "step_index": step.step_index,
                    "full_token_position": step.full_token_positions[-1],
                }
            )
    if len(answer_token_positions) > max_answer_probes:
        # Evenly-spaced sample across the answer span. Linspace then cast to
        # int catches both ends and avoids stride aliasing on edge sizes.
        sample_indices = np.linspace(
            0, len(answer_token_positions) - 1, max_answer_probes, dtype=np.int64
        ).tolist()
        sampled_positions = [int(answer_token_positions[i]) for i in sample_indices]
    else:
        sampled_positions = [int(p) for p in answer_token_positions]
    for pos in sampled_positions:
        probes.append(
            {
                "probe_type": "answer",
                "step_index": None,
                "full_token_position": pos,
            }
        )
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for probe in probes:
        key = (probe["probe_type"], probe["step_index"], probe["full_token_position"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(probe)
    return deduped


def probe_attention_vectors(
    model: AutoModelForCausalLM,
    full_ids: torch.Tensor,
    probes: Sequence[Dict[str, Any]],
    save_full_probe_attention: bool,
    sample_dir: Path,
    raw_attentions: Optional[Sequence["torch.Tensor"]] = None,
) -> Dict[str, Any]:
    """Build the (num_probes, num_layers, num_heads, full_len) attention array.

    raw_attentions, when supplied, is the tuple of per-layer attention tensors
    from a single full-sequence forward pass with output_attentions=True.
    Causal masking guarantees that the slice attentions[layer][0, :, pos, :]
    is bit-identical to running a fresh forward over the prefix full_ids[:pos+1]
    and taking attentions[layer][0, :, -1, :], so this path produces the same
    numerical result as the prefix-forward path with one model call instead of
    one per probe. On a 1500-token sequence this is the difference between
    ~150s of attention probing and ~2s.
    """
    device = get_model_input_device(model)
    full_len = int(full_ids.shape[0])

    if raw_attentions is None:
        # Legacy fallback: do per-probe forward passes. Kept for the rare case
        # where a caller wants attentions without paying for the full
        # output_attentions forward (e.g. very-long sequence + tight memory).
        collected = []
        meta = []
        for probe in probes:
            pos = int(probe["full_token_position"])
            prefix = full_ids[: pos + 1].unsqueeze(0).to(device)
            with torch.inference_mode():
                outputs = model(
                    input_ids=prefix,
                    attention_mask=torch.ones_like(prefix),
                    output_attentions=True,
                    use_cache=False,
                )
            if outputs.attentions is None:
                raise RuntimeError(
                    "Model backend did not return attentions. Re-run with "
                    "--attn-implementation eager."
                )
            layer_vectors = []
            for attn in outputs.attentions:
                vec = attn[0, :, -1, :].detach().float().cpu()
                padded = torch.zeros((vec.shape[0], full_len), dtype=torch.float32)
                padded[:, : vec.shape[1]] = vec
                layer_vectors.append(padded)
            collected.append(torch.stack(layer_vectors, dim=0))
            meta.append(
                {
                    "probe_type": probe["probe_type"],
                    "step_index": probe["step_index"],
                    "full_token_position": pos,
                }
            )
        attention_array = (
            torch.stack(collected, dim=0).numpy().astype(np.float32)
            if collected
            else np.zeros((0, 0, 0, full_len), dtype=np.float32)
        )
    else:
        # Fast path: one forward already gave us all per-layer attentions.
        # Causal masking guarantees equivalence to the prefix-forward path,
        # and we vectorise the per-layer slice so we do one device->host
        # transfer per layer instead of one per (probe, layer) pair.
        meta = []
        if not probes:
            attention_array = np.zeros((0, 0, 0, full_len), dtype=np.float32)
        else:
            num_layers = len(raw_attentions)
            sample_layer = raw_attentions[0]
            num_heads = int(sample_layer.shape[1])
            seq_len = int(sample_layer.shape[2])
            probe_positions = [int(p["full_token_position"]) for p in probes]
            for pos in probe_positions:
                if pos >= seq_len:
                    raise RuntimeError(
                        f"Probe position {pos} exceeds attention seq_len {seq_len}; "
                        f"the attention tensor was computed on a shorter input."
                    )
            attention_array = np.zeros(
                (len(probes), num_layers, num_heads, full_len), dtype=np.float32
            )
            position_index = torch.tensor(probe_positions, device=sample_layer.device)
            for layer_idx, layer_attn in enumerate(raw_attentions):
                # layer_attn: [1, num_heads, seq_len, seq_len]
                # index_select on dim=2 gives [1, num_heads, num_probes, seq_len];
                # permute brings probes to the leading axis so the host array
                # write line up below is contiguous.
                sliced = (
                    layer_attn.index_select(2, position_index)
                    .squeeze(0)
                    .permute(1, 0, 2)
                    .detach()
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )
                attention_array[:, layer_idx, :, :seq_len] = sliced
            for probe in probes:
                meta.append(
                    {
                        "probe_type": probe["probe_type"],
                        "step_index": probe["step_index"],
                        "full_token_position": int(probe["full_token_position"]),
                    }
                )

    if save_full_probe_attention and attention_array.size > 0:
        np.savez_compressed(
            sample_dir / "probe_attention.npz",
            probe_attention=attention_array.astype(np.float16),
        )
    return {"probe_metadata": meta, "probe_attention": attention_array}


def build_token_rows(
    tokenization: Dict[str, Any],
    prompt_text: str,
    token_logprobs: np.ndarray,
    token_entropy: np.ndarray,
    reasoning_span_generated: Sequence[int],
    answer_span_generated: Sequence[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    prompt_char_len = len(prompt_text)
    token_logprob_len = len(token_logprobs)
    token_entropy_len = len(token_entropy)
    for idx, token_id in enumerate(tokenization["input_ids"]):
        char_start, char_end = tokenization["offset_mapping"][idx]
        is_generated = char_start >= prompt_char_len and char_end > char_start
        gen_start = char_start - prompt_char_len if is_generated else None
        gen_end = char_end - prompt_char_len if is_generated else None
        is_reasoning = (
            is_generated
            and gen_end is not None
            and gen_end > reasoning_span_generated[0]
            and gen_start < reasoning_span_generated[1]
        )
        is_answer = (
            is_generated
            and gen_end is not None
            and gen_end > answer_span_generated[0]
            and gen_start < answer_span_generated[1]
        )
        rows.append(
            {
                "token_index": idx,
                "token_id": int(token_id),
                "token_str": tokenization["tokens"][idx],
                "char_start_full": int(char_start),
                "char_end_full": int(char_end),
                "char_start_generated": gen_start,
                "char_end_generated": gen_end,
                "is_special": bool(tokenization["special_tokens_mask"][idx]),
                "is_generated": bool(is_generated),
                "is_reasoning": bool(is_reasoning),
                "is_answer": bool(is_answer),
                "token_logprob": (
                    None
                    if idx >= token_logprob_len or np.isnan(token_logprobs[idx])
                    else float(token_logprobs[idx])
                ),
                "token_entropy": (
                    None
                    if idx >= token_entropy_len or np.isnan(token_entropy[idx])
                    else float(token_entropy[idx])
                ),
            }
        )
    return rows


def summarize_entropy_over_generated(
    token_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    generated = [
        (row["token_index"], row["token_entropy"])
        for row in token_rows
        if row["is_generated"] and not row["is_special"] and row["token_entropy"] is not None
    ]
    if not generated:
        return {
            "mean_entropy_generated": None,
            "entropy_slope_generated": None,
            "mean_entropy_reasoning": None,
            "mean_entropy_answer": None,
            "num_entropy_tokens": 0,
        }
    indices = np.array([row[0] for row in generated], dtype=np.float64)
    values = np.array([row[1] for row in generated], dtype=np.float64)
    mean_entropy = float(values.mean())
    if indices.size >= 2 and float(indices.std()) > 0.0:
        slope = float(np.polyfit(indices - indices.min(), values, 1)[0])
    else:
        slope = 0.0

    def mean_for(predicate) -> Optional[float]:
        vals = [
            row["token_entropy"]
            for row in token_rows
            if row["is_generated"]
            and not row["is_special"]
            and row["token_entropy"] is not None
            and predicate(row)
        ]
        return float(np.mean(vals)) if vals else None

    return {
        "mean_entropy_generated": mean_entropy,
        "entropy_slope_generated": slope,
        "mean_entropy_reasoning": mean_for(lambda row: row["is_reasoning"]),
        "mean_entropy_answer": mean_for(lambda row: row["is_answer"]),
        "num_entropy_tokens": int(values.size),
    }


def summarize_step_scores(
    steps: Sequence[StepSpan],
    probe_metadata: Sequence[Dict[str, Any]],
    probe_attention: np.ndarray,
    delta_norm_by_layer_token: np.ndarray,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    if not steps:
        return [], []

    num_layers = int(probe_attention.shape[1]) if probe_attention.ndim == 4 else 0
    answer_probe_indices = [i for i, meta in enumerate(probe_metadata) if meta["probe_type"] == "answer"]
    step_probe_indices = [i for i, meta in enumerate(probe_metadata) if meta["probe_type"] == "step_end"]

    def zscore(values: List[float]) -> List[float]:
        arr = np.array(values, dtype=np.float32)
        if arr.size == 0:
            return []
        std = float(arr.std())
        if std < 1e-8:
            return [0.0 for _ in values]
        return ((arr - float(arr.mean())) / std).tolist()

    rows: List[Dict[str, Any]] = []
    future_vals: List[float] = []
    answer_vals: List[float] = []
    activation_vals: List[float] = []

    for step in steps:
        positions = step.full_token_positions
        if positions and probe_attention.size > 0:
            answer_mean = (
                float(probe_attention[answer_probe_indices, :, :, :][:, :, :, positions].sum(axis=-1).mean())
                if answer_probe_indices
                else 0.0
            )
            future_probe_indices = [
                i
                for i in step_probe_indices
                if probe_metadata[i]["step_index"] is not None
                and int(probe_metadata[i]["step_index"]) > step.step_index
            ]
            future_mean = (
                float(probe_attention[future_probe_indices, :, :, :][:, :, :, positions].sum(axis=-1).mean())
                if future_probe_indices
                else 0.0
            )
            answer_by_layer = (
                probe_attention[answer_probe_indices, :, :, :][:, :, :, positions].sum(axis=-1).mean(axis=(0, 2)).tolist()
                if answer_probe_indices
                else [0.0] * num_layers
            )
            future_by_layer = (
                probe_attention[future_probe_indices, :, :, :][:, :, :, positions].sum(axis=-1).mean(axis=(0, 2)).tolist()
                if future_probe_indices
                else [0.0] * num_layers
            )
        else:
            answer_mean = 0.0
            future_mean = 0.0
            answer_by_layer = [0.0] * num_layers
            future_by_layer = [0.0] * num_layers

        activation_mean = (
            float(delta_norm_by_layer_token[1:, positions].mean())
            if positions and delta_norm_by_layer_token.shape[0] > 1
            else 0.0
        )
        rows.append(
            {
                "step_index": step.step_index,
                "text": step.text,
                "num_full_tokens": len(step.full_token_positions),
                "num_generated_tokens": len(step.generated_token_positions),
                "future_attention_mean": future_mean,
                "answer_attention_mean": answer_mean,
                "activation_delta_mean": activation_mean,
                "answer_attention_by_layer": answer_by_layer,
                "future_attention_by_layer": future_by_layer,
            }
        )
        future_vals.append(future_mean)
        answer_vals.append(answer_mean)
        activation_vals.append(activation_mean)

    future_z = zscore(future_vals)
    answer_z = zscore(answer_vals)
    activation_z = zscore(activation_vals)

    # Optional late-trace position penalty for the anchor score. The
    # raw attention-based score is biased toward late-trace summary
    # steps on long Thinking traces (which are by-construction "what
    # the rest of the trace looks at" — and the final summary IS what
    # the trace looks at). When AJAR_ANCHOR_LATE_PENALTY > 0 we
    # multiply each step's combined score by (1 - alpha * rel_pos)
    # where rel_pos is 0 at the first step and 1 at the last step.
    # Default 0.0 preserves committed runs exactly. Recommended
    # values for the methodology-fix run: 0.5 (mild) to 1.0
    # (aggressive — last step gets zero).
    try:
        penalty_alpha = float(os.environ.get("AJAR_ANCHOR_LATE_PENALTY", "0.0"))
    except ValueError:
        penalty_alpha = 0.0
    n_steps = max(len(rows), 1)

    for idx, row in enumerate(rows):
        base_score = float(future_z[idx] + answer_z[idx] + activation_z[idx])
        if penalty_alpha != 0.0 and n_steps > 1:
            rel_pos = idx / (n_steps - 1)
            penalty = 1.0 - penalty_alpha * rel_pos
        else:
            penalty = 1.0
        row["combined_anchor_score"] = base_score * penalty
        row["combined_anchor_score_unpenalised"] = base_score
        row["late_penalty_alpha"] = penalty_alpha
        layer_scores = np.array(row["answer_attention_by_layer"]) + np.array(row["future_attention_by_layer"])
        row["top_layers"] = np.argsort(-layer_scores).astype(int).tolist()

    ranked_indices = [r["step_index"] for r in sorted(rows, key=lambda x: x["combined_anchor_score"], reverse=True)]
    return rows, ranked_indices


def choose_control_steps(
    steps: Sequence[StepSpan],
    anchor_indices: Sequence[int],
    desired_count: int,
    rng: random.Random,
) -> List[int]:
    available = [step.step_index for step in steps if step.step_index not in set(anchor_indices)]
    rng.shuffle(available)
    return available[:desired_count]


def hook_scale_positions(output: Any, positions: Sequence[int], factor: float) -> Any:
    if not positions:
        return output

    def scale_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor) or tensor.ndim < 3:
            return tensor
        seq_len = tensor.shape[1]
        valid_positions = [p for p in positions if 0 <= p < seq_len]
        if seq_len <= 1 or not valid_positions:
            return tensor
        result = tensor.clone()
        result[:, valid_positions, :] = result[:, valid_positions, :] * factor
        return result

    if isinstance(output, torch.Tensor):
        return scale_tensor(output)
    if isinstance(output, tuple) and output:
        return (scale_tensor(output[0]),) + output[1:]
    return output


def generate_with_intervention(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix_text: str,
    target_token_positions: Sequence[int],
    target_layers: Sequence[int],
    intervention_mode: str,
    args: ExperimentConfig,
) -> Dict[str, Any]:
    stream_kind, factor = INTERVENTION_SPECS[intervention_mode]
    decoder_layers = get_decoder_layers(model)
    handles = []
    for layer_idx in target_layers:
        if layer_idx < 0 or layer_idx >= len(decoder_layers):
            continue
        layer = decoder_layers[layer_idx]
        module = layer if stream_kind == "residual" else getattr(layer, "self_attn", None)
        if module is None:
            continue
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, pos=list(target_token_positions), scale=factor: hook_scale_positions(
                    output, pos, scale
                )
            )
        )
    try:
        # Intervention continuations use a tighter token budget than baselines:
        # we only need to reach the answer, not regenerate the whole reasoning
        # trace. This is the dominant wall-time saving on long-output models.
        intervention_args = dataclasses.replace(
            args, max_new_tokens=args.intervention_max_new_tokens
        )
        return generate_completion(model, tokenizer, prefix_text, intervention_args)
    finally:
        for handle in handles:
            handle.remove()


def build_prefix_text(prompt_text: str, generated_text: str, step: StepSpan) -> str:
    return prompt_text + generated_text[: step.char_end_generated]


def maybe_free_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # MPS keeps allocations cached too, and unlike CUDA we depend on it
    # actually shrinking when we del a model in load_or_switch_model. Calling
    # empty_cache between major phases keeps peak memory in check, especially
    # when swapping between Instruct and Thinking weights.
    mps_module = getattr(torch, "mps", None)
    if mps_module is not None and hasattr(mps_module, "empty_cache"):
        try:
            mps_module.empty_cache()
        except Exception:
            pass


def build_intervention_tasks(
    args: ExperimentConfig,
    model_key: str,
    model_name: str,
    prompt_name: str,
    sample_idx: int,
    sample_dir: Path,
    prompt_text: str,
    generated_text: str,
    gold_numeric: Optional[str],
    step_scores: Sequence[Dict[str, Any]],
    steps: Sequence[StepSpan],
    anchor_indices: Sequence[int],
    control_indices: Sequence[int],
    tokenizer: AutoTokenizer,
) -> List[Dict[str, Any]]:
    step_by_index = {step.step_index: step for step in steps}
    targets = [("anchor", idx) for idx in anchor_indices[: args.top_anchor_steps]]
    targets += [("control", idx) for idx in control_indices]
    valid_targets = [
        (step_kind, step_index)
        for step_kind, step_index in targets
        if step_by_index.get(step_index) is not None
        and step_by_index[step_index].generated_token_positions
    ]
    tasks: List[Dict[str, Any]] = []
    for step_kind, step_index in valid_targets:
        step = step_by_index[step_index]
        layer_row = next((row for row in step_scores if row["step_index"] == step_index), None)
        target_layers = (
            layer_row["top_layers"][: args.top_anchor_layers]
            if layer_row is not None
            else list(range(args.top_anchor_layers))
        )
        prefix_text = build_prefix_text(prompt_text, generated_text, step)
        prefix_token_count = len(tokenizer(prefix_text, add_special_tokens=True)["input_ids"])
        for intervention_mode in args.interventions:
            output_rel_path = f"interventions/{step_kind}_step{step_index:02d}_{intervention_mode}.json"
            tasks.append(
                {
                    "task_type": "intervention",
                    "model_key": model_key,
                    "model_name": model_name,
                    "prompt_name": prompt_name,
                    "sample_idx": sample_idx,
                    "sample_dir": str(sample_dir),
                    "gold_numeric": gold_numeric,
                    "step_kind": step_kind,
                    "step_index": step_index,
                    "step_text": step.text,
                    "step_num_tokens": len(step.generated_token_positions),
                    "step_char_end_generated": step.char_end_generated,
                    "target_layers": target_layers,
                    "target_token_positions": step.full_token_positions,
                    "intervention_mode": intervention_mode,
                    "prefix_text": prefix_text,
                    "prefix_full_token_count": prefix_token_count,
                    "baseline_generated_text": generated_text,
                    "output_rel_path": output_rel_path,
                }
            )
    return tasks


def run_baseline_task(
    task: Dict[str, Any],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    args: ExperimentConfig,
    run_root: Path,
    worker_name: str,
    worker_paths: Dict[str, Path],
    logger: QueueLogger,
    event_queue: Any,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    model_key = str(task["model_key"])
    model_name = str(task["model_name"])
    sample_idx = int(task["sample_idx"])
    prompt_name = str(task["prompt_name"])
    question = str(task["question"])
    gold_answer_raw = str(task["gold_answer_raw"])
    gold_numeric = parse_gsm8k_answer(gold_answer_raw)
    prompt_tag = f"{worker_name} {model_key} sample={sample_idx:04d} prompt={prompt_name}"

    sample_dir = get_sample_dir(run_root, model_key, prompt_name, sample_idx)
    baseline_path = sample_dir / "baseline.json"
    if args.resume and is_sample_complete_for_resume(
        run_root,
        model_key,
        prompt_name,
        sample_idx,
        expected_question=question,
    ):
        logger.log(f"Skipping fully complete sample for {prompt_tag}.")
        emit_event(event_queue, "baseline_skipped")
        return None, []
    sample_dir.mkdir(parents=True, exist_ok=True)

    messages = build_messages(prompt_name, question)
    prompt_text = build_prompt_text(tokenizer, messages)
    if args.log_baseline_generation:
        logger.log(f"Starting baseline generation for {prompt_tag}.")
    with StageHeartbeat(
        logger,
        args.enable_stage_heartbeat and args.log_baseline_generation,
        f"baseline generation for {prompt_tag}",
        args.heartbeat_interval_seconds,
    ):
        generation = generate_completion(model, tokenizer, prompt_text, args)
    if args.log_baseline_generation:
        logger.log(
            f"Finished baseline generation for {prompt_tag} in "
            f"{generation['generation_seconds']:.2f}s with "
            f"{int(generation['generated_ids'].shape[0])} generated tokens."
        )
    generated_text = generation["generated_text"]
    full_text = prompt_text + generated_text
    prediction_numeric, boxed_content = extract_predicted_number(generated_text)
    correct = prediction_numeric == gold_numeric and prediction_numeric is not None

    span_info = identify_reasoning_and_answer_spans(generated_text)
    steps, tokenization = build_step_spans(
        tokenizer,
        prompt_text,
        generated_text,
        span_info["reasoning_char_span_generated"],
    )
    answer_token_positions = get_answer_token_positions(
        tokenization["offset_mapping"],
        tokenization["special_tokens_mask"],
        prompt_text,
        span_info["answer_char_span_generated"],
    )
    if args.log_analysis_stages:
        logger.log(
            f"Segmented {prompt_tag} into {len(steps)} reasoning step(s); "
            f"{len(answer_token_positions)} answer token(s) detected."
        )

    if generation["full_ids"].shape[0] <= args.analysis_max_seq_len:
        if args.log_analysis_stages:
            logger.log(
                f"Running hidden-state analysis for {prompt_tag} "
                f"at sequence length {int(generation['full_ids'].shape[0])}."
            )
        with StageHeartbeat(
            logger,
            args.enable_stage_heartbeat and args.log_analysis_stages,
            f"hidden-state analysis for {prompt_tag}",
            args.heartbeat_interval_seconds,
        ):
            hidden_summary = run_full_forward_hidden_analysis(
                model,
                generation["full_ids"],
                save_full_hidden_states=args.save_full_hidden_states,
                sample_dir=sample_dir,
            )
        if args.log_analysis_stages:
            logger.log(f"Hidden-state analysis complete for {prompt_tag}.")
        probe_plan = build_probe_plan(
            steps, answer_token_positions, max_answer_probes=args.max_answer_probes
        )
        if args.log_analysis_stages:
            logger.log(
                f"Running attention probing for {prompt_tag} with "
                f"{len(probe_plan)} probe position(s)."
            )
        with StageHeartbeat(
            logger,
            args.enable_stage_heartbeat and args.log_analysis_stages,
            f"attention probing for {prompt_tag}",
            args.heartbeat_interval_seconds,
        ):
            probe_summary = probe_attention_vectors(
                model,
                generation["full_ids"],
                probe_plan,
                save_full_probe_attention=args.save_full_probe_attention,
                sample_dir=sample_dir,
                raw_attentions=hidden_summary.get("raw_attentions"),
            )
        # Free the device-side attention tensors as soon as we are done
        # slicing them; they are 5+ GB on long Thinking outputs.
        hidden_summary.pop("raw_attentions", None)
        if args.log_analysis_stages:
            logger.log(f"Attention probing complete for {prompt_tag}.")
        step_scores, anchor_indices = summarize_step_scores(
            steps,
            probe_summary["probe_metadata"],
            probe_summary["probe_attention"],
            hidden_summary["delta_norm_by_layer_token"],
        )
        if args.log_analysis_stages:
            logger.log(
                f"Scored anchor candidates for {prompt_tag}; "
                f"top anchors: {anchor_indices[: args.top_anchor_steps]}."
            )
    else:
        logger.log(
            f"Skipping deep analysis for {prompt_tag} because sequence length "
            f"{int(generation['full_ids'].shape[0])} exceeds "
            f"ANALYSIS_MAX_SEQ_LEN={args.analysis_max_seq_len}."
        )
        hidden_summary = {
            "token_logprobs": np.full((int(generation["full_ids"].shape[0]),), np.nan, dtype=np.float32),
            "token_entropy": np.full((int(generation["full_ids"].shape[0]),), np.nan, dtype=np.float32),
            "delta_norm_by_layer_token": np.zeros((0, int(generation["full_ids"].shape[0])), dtype=np.float32),
        }
        step_scores, anchor_indices = [], []

    control_indices = choose_control_steps(
        steps=steps,
        anchor_indices=anchor_indices[: args.top_anchor_steps],
        desired_count=args.num_control_steps,
        rng=random.Random(args.seed + sample_idx),
    )
    token_rows = build_token_rows(
        tokenization,
        prompt_text,
        hidden_summary["token_logprobs"],
        hidden_summary["token_entropy"],
        span_info["reasoning_char_span_generated"],
        span_info["answer_char_span_generated"],
    )
    entropy_summary = summarize_entropy_over_generated(token_rows)
    write_csv(sample_dir / "tokens.csv", token_rows)
    if args.log_analysis_stages:
        logger.log(f"Wrote token table for {prompt_tag} to {sample_dir / 'tokens.csv'}.")

    baseline_record = {
        "sample_idx": sample_idx,
        "question": question,
        "gold_numeric": gold_numeric,
        "gold_answer_raw": gold_answer_raw,
        "model_key": model_key,
        "model_name": model_name,
        "prompt_name": prompt_name,
        "prompt_text": prompt_text,
        "generated_text": generated_text,
        "full_text": full_text,
        "prompt_len_tokens": int(generation["prompt_len"]),
        "generated_len_tokens": int(generation["generated_ids"].shape[0]),
        "full_len_tokens": int(generation["full_ids"].shape[0]),
        "generation_seconds": float(generation["generation_seconds"]),
        "prediction_numeric": prediction_numeric,
        "prediction_boxed_content": boxed_content,
        "correct": bool(correct),
        "reasoning_char_span_generated": span_info["reasoning_char_span_generated"],
        "answer_char_span_generated": span_info["answer_char_span_generated"],
        "reasoning_text": span_info["reasoning_text"],
        "answer_text": span_info["answer_text"],
        "num_steps": len(steps),
        "step_scores": step_scores,
        "anchor_indices_ranked": anchor_indices,
        "anchor_indices_selected": anchor_indices[: args.top_anchor_steps],
        "control_indices_selected": control_indices,
        "analysis_skipped": bool(generation["full_ids"].shape[0] > args.analysis_max_seq_len),
        "artifact_dir": str(sample_dir),
        "worker_name": worker_name,
        **entropy_summary,
    }
    save_json(baseline_path, baseline_record)
    append_jsonl(worker_paths["baseline_jsonl"], baseline_record)
    emit_event(event_queue, "baseline_done")
    logger.log(
        f"Baseline complete for {prompt_tag}: correct={correct}, "
        f"prediction={prediction_numeric}, gold={gold_numeric}."
    )

    baseline_row = {
        "sample_idx": sample_idx,
        "model_key": model_key,
        "prompt_name": prompt_name,
        "correct": int(correct),
        "gold_numeric": gold_numeric,
        "prediction_numeric": prediction_numeric,
        "generated_len_tokens": int(generation["generated_ids"].shape[0]),
        "generation_seconds": float(generation["generation_seconds"]),
        "num_steps": len(steps),
        "analysis_skipped": bool(generation["full_ids"].shape[0] > args.analysis_max_seq_len),
        "worker_name": worker_name,
        "mean_entropy_generated": entropy_summary["mean_entropy_generated"],
        "entropy_slope_generated": entropy_summary["entropy_slope_generated"],
    }
    intervention_tasks = build_intervention_tasks(
        args=args,
        model_key=model_key,
        model_name=model_name,
        prompt_name=prompt_name,
        sample_idx=sample_idx,
        sample_dir=sample_dir,
        prompt_text=prompt_text,
        generated_text=generated_text,
        gold_numeric=gold_numeric,
        step_scores=step_scores,
        steps=steps,
        anchor_indices=anchor_indices,
        control_indices=control_indices,
        tokenizer=tokenizer,
    )
    save_json(
        sample_dir / "intervention_manifest.json",
        {
            "sample_idx": sample_idx,
            "model_key": model_key,
            "prompt_name": prompt_name,
            "expected_intervention_files": [task["output_rel_path"] for task in intervention_tasks],
        },
    )
    logger.log(f"Prepared {len(intervention_tasks)} intervention task(s) for {prompt_tag}.")
    maybe_free_cuda()
    return baseline_row, intervention_tasks


def run_intervention_task(
    task: Dict[str, Any],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    args: ExperimentConfig,
    worker_name: str,
    worker_paths: Dict[str, Path],
    logger: QueueLogger,
    event_queue: Any,
) -> Optional[Dict[str, Any]]:
    model_key = str(task["model_key"])
    model_name = str(task["model_name"])
    prompt_name = str(task["prompt_name"])
    sample_idx = int(task["sample_idx"])
    sample_dir = Path(str(task["sample_dir"]))
    gold_numeric = task["gold_numeric"]
    step_kind = str(task["step_kind"])
    step_index = int(task["step_index"])
    step_text = str(task["step_text"])
    step_num_tokens = int(task["step_num_tokens"])
    step_char_end_generated = int(task["step_char_end_generated"])
    target_layers = list(task["target_layers"])
    target_token_positions = list(task["target_token_positions"])
    intervention_mode = str(task["intervention_mode"])
    prefix_text = str(task["prefix_text"])
    prefix_token_count = int(task["prefix_full_token_count"])
    baseline_generated_text = str(task["baseline_generated_text"])
    output_rel_path = str(task["output_rel_path"])
    intervention_tag = (
        f"{worker_name} {model_key} sample={sample_idx:04d} prompt={prompt_name} "
        f"{step_kind}_step={step_index} mode={intervention_mode}"
    )
    if args.log_interventions:
        logger.log(f"Starting intervention for {intervention_tag} on layers={target_layers}.")
    with StageHeartbeat(
        logger,
        args.enable_stage_heartbeat and args.log_interventions,
        f"intervention generation for {intervention_tag}",
        args.heartbeat_interval_seconds,
    ):
        intervention = generate_with_intervention(
            model=model,
            tokenizer=tokenizer,
            prefix_text=prefix_text,
            target_token_positions=target_token_positions,
            target_layers=target_layers,
            intervention_mode=intervention_mode,
            args=args,
        )
    continuation_text = intervention["generated_text"]
    counterfactual_text = baseline_generated_text[:step_char_end_generated] + continuation_text
    cf_numeric, cf_boxed = extract_predicted_number(counterfactual_text)
    cf_correct = cf_numeric == gold_numeric and cf_numeric is not None
    intervention_record = {
        "sample_idx": sample_idx,
        "model_key": model_key,
        "model_name": model_name,
        "prompt_name": prompt_name,
        "step_kind": step_kind,
        "step_index": step_index,
        "step_text": step_text,
        "step_num_tokens": step_num_tokens,
        "target_layers": target_layers,
        "intervention_mode": intervention_mode,
        "prefix_text": prefix_text,
        "prefix_full_token_count": prefix_token_count,
        "continuation_text": continuation_text,
        "counterfactual_text": counterfactual_text,
        "prediction_numeric": cf_numeric,
        "prediction_boxed_content": cf_boxed,
        "correct": bool(cf_correct),
        "changed_from_baseline": bool(counterfactual_text != baseline_generated_text),
        "generation_seconds": float(intervention["generation_seconds"]),
        "worker_name": worker_name,
    }
    save_json(sample_dir / output_rel_path, intervention_record)
    append_jsonl(worker_paths["intervention_jsonl"], intervention_record)
    emit_event(event_queue, "intervention_done")
    if args.log_interventions:
        logger.log(
            f"Finished intervention for {intervention_tag} in "
            f"{intervention['generation_seconds']:.2f}s; correct={cf_correct}, "
            f"prediction={cf_numeric}."
        )
    maybe_free_cuda()
    return {
        "sample_idx": sample_idx,
        "model_key": model_key,
        "prompt_name": prompt_name,
        "step_kind": step_kind,
        "step_index": step_index,
        "intervention_mode": intervention_mode,
        "correct": int(cf_correct),
        "prediction_numeric": cf_numeric,
        "generation_seconds": float(intervention["generation_seconds"]),
        "worker_name": worker_name,
    }


def run_omlx_baseline_task(
    example: Dict[str, Any],
    args: ExperimentConfig,
    run_root: Path,
    model_key: str,
    model_name: str,
    prompt_name: str,
    sample_idx: int,
    logger: RunLogger,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    sample_dir = get_sample_dir(run_root, model_key, prompt_name, sample_idx)
    question = str(example["question"])
    if args.resume and is_sample_complete_for_resume(
        run_root,
        model_key,
        prompt_name,
        sample_idx,
        expected_question=question,
    ):
        logger.log(f"Skipping complete local sample: {model_key} sample={sample_idx:04d} prompt={prompt_name}.")
        return None, None

    sample_dir.mkdir(parents=True, exist_ok=True)
    gold_answer_raw = str(example["answer"])
    gold_numeric = parse_gsm8k_answer(gold_answer_raw)
    messages = build_messages(prompt_name, question)
    prompt_text = build_plain_prompt_text(messages)
    tag = f"{model_key} sample={sample_idx:04d} prompt={prompt_name}"

    logger.log(f"Starting local oMLX baseline generation for {tag}.")
    with StageHeartbeat(
        logger,
        args.enable_stage_heartbeat and args.log_baseline_generation,
        f"local oMLX baseline generation for {tag}",
        args.heartbeat_interval_seconds,
    ):
        generation = generate_omlx_completion(model_name, messages, args)
    logger.log(
        f"Finished local oMLX baseline generation for {tag} in "
        f"{generation['generation_seconds']:.2f}s."
    )

    generated_text = generation["generated_text"]
    prediction_numeric, boxed_content = extract_predicted_number(generated_text)
    correct = prediction_numeric == gold_numeric and prediction_numeric is not None
    span_info = identify_reasoning_and_answer_spans(generated_text)
    reasoning_steps = split_text_into_steps(
        span_info["reasoning_text"],
        base_offset=span_info["reasoning_char_span_generated"][0],
    )

    baseline_record = {
        "sample_idx": sample_idx,
        "question": question,
        "gold_numeric": gold_numeric,
        "gold_answer_raw": gold_answer_raw,
        "model_key": model_key,
        "model_name": model_name,
        "prompt_name": prompt_name,
        "prompt_text": prompt_text,
        "generated_text": generated_text,
        "generation_seconds": float(generation["generation_seconds"]),
        "finish_reason": generation["finish_reason"],
        "usage": generation["usage"],
        "raw_response_id": generation["raw_response_id"],
        "prediction_numeric": prediction_numeric,
        "prediction_boxed_content": boxed_content,
        "correct": bool(correct),
        "reasoning_char_span_generated": span_info["reasoning_char_span_generated"],
        "answer_char_span_generated": span_info["answer_char_span_generated"],
        "reasoning_text": span_info["reasoning_text"],
        "answer_text": span_info["answer_text"],
        "num_steps": len(reasoning_steps),
        "step_scores": [],
        "anchor_indices_ranked": [],
        "anchor_indices_selected": [],
        "control_indices_selected": [],
        "analysis_skipped": True,
        "analysis_skip_reason": "local oMLX API returns generated text only; PyTorch hidden states, attentions, and hooks are unavailable.",
        "artifact_dir": str(sample_dir),
        "worker_name": "local_omlx",
    }
    save_json(sample_dir / "baseline.json", baseline_record)
    save_json(
        sample_dir / "intervention_manifest.json",
        {
            "sample_idx": sample_idx,
            "model_key": model_key,
            "prompt_name": prompt_name,
            "expected_intervention_files": [],
            "skip_reason": "Interventions require PyTorch forward hooks; local oMLX mode is baseline-only.",
        },
    )
    baseline_row = {
        "sample_idx": sample_idx,
        "model_key": model_key,
        "prompt_name": prompt_name,
        "correct": int(correct),
        "gold_numeric": gold_numeric,
        "prediction_numeric": prediction_numeric,
        "generation_seconds": float(generation["generation_seconds"]),
        "num_steps": len(reasoning_steps),
        "analysis_skipped": True,
        "worker_name": "local_omlx",
    }
    logger.log(f"Local baseline complete for {tag}: correct={correct}, prediction={prediction_numeric}, gold={gold_numeric}.")
    return baseline_record, baseline_row


def load_or_switch_model(
    current_model_key: Optional[str],
    target_model_key: str,
    current_tokenizer: Optional[AutoTokenizer],
    current_model: Optional[AutoModelForCausalLM],
    args: ExperimentConfig,
    device: str,
    dtype: torch.dtype,
    logger: QueueLogger,
) -> Tuple[str, AutoTokenizer, AutoModelForCausalLM]:
    if current_model_key == target_model_key and current_tokenizer is not None and current_model is not None:
        return current_model_key, current_tokenizer, current_model
    if current_model is not None:
        logger.log(f"Unloading model '{current_model_key}' before switching to '{target_model_key}'.")
        del current_model
        maybe_free_cuda()
    model_name = MODEL_SPECS[target_model_key]
    logger.log(f"Loading model '{target_model_key}' on {device}.")
    tokenizer, model = load_model_and_tokenizer(
        model_name=model_name,
        dtype=dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        device=device,
    )
    return target_model_key, tokenizer, model


def worker_main(
    gpu_id: int,
    slot_index: int,
    task_queue: Any,
    event_queue: Any,
    pending_task_counter: Any,
    pending_task_lock: Any,
    args: ExperimentConfig,
) -> None:
    worker_name = get_worker_name("pool", gpu_id, slot_index)
    logger = QueueLogger(args.verbose_logging, event_queue, worker_name)
    worker_paths = get_worker_output_paths(args.output_dir, worker_name)
    worker_paths["dir"].mkdir(parents=True, exist_ok=True)
    baseline_rows: List[Dict[str, Any]] = []
    intervention_rows: List[Dict[str, Any]] = []

    # On spawn-based multiprocessing the child re-imports the module from
    # scratch, so torch/transformers/datasets are unbound until we ask for
    # them again. Doing this before any other work means a missing dep
    # surfaces here as a clean SystemExit, not as a downstream NameError.
    require_backend_dependencies(args.inference_backend)

    try:
        configure_worker_environment(args, gpu_id)
        set_seed(args.seed + gpu_id + (slot_index * 1000))
        if torch.cuda.is_available():
            device = f"cuda:{gpu_id}"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        dtype = choose_dtype(args.dtype)
        logger.log(f"Worker starting on device={device} with dtype={dtype}.")
        current_model_key: Optional[str] = None
        tokenizer: Optional[AutoTokenizer] = None
        model: Optional[AutoModelForCausalLM] = None

        while True:
            try:
                task = task_queue.get(timeout=2.0)
            except queue.Empty:
                with pending_task_lock:
                    if pending_task_counter.value == 0:
                        break
                continue
            try:
                task_type = str(task["task_type"])
                target_model_key = str(task["model_key"])
                current_model_key, tokenizer, model = load_or_switch_model(
                    current_model_key=current_model_key,
                    target_model_key=target_model_key,
                    current_tokenizer=tokenizer,
                    current_model=model,
                    args=args,
                    device=device,
                    dtype=dtype,
                    logger=logger,
                )
                logger.log(f"Starting {task_type} task for model '{target_model_key}'.")
                if task_type == "baseline":
                    baseline_row, intervention_tasks = run_baseline_task(
                        task=task,
                        tokenizer=tokenizer,
                        model=model,
                        args=args,
                        run_root=args.output_dir,
                        worker_name=worker_name,
                        worker_paths=worker_paths,
                        logger=logger,
                        event_queue=event_queue,
                    )
                    if baseline_row is not None:
                        baseline_rows.append(baseline_row)
                    if intervention_tasks:
                        for intervention_task in intervention_tasks:
                            task_queue.put(intervention_task)
                        with pending_task_lock:
                            pending_task_counter.value += len(intervention_tasks)
                        emit_event(event_queue, "intervention_total_add", count=len(intervention_tasks))
                elif task_type == "intervention":
                    intervention_row = run_intervention_task(
                        task=task,
                        tokenizer=tokenizer,
                        model=model,
                        args=args,
                        worker_name=worker_name,
                        worker_paths=worker_paths,
                        logger=logger,
                        event_queue=event_queue,
                    )
                    if intervention_row is not None:
                        intervention_rows.append(intervention_row)
                else:
                    raise ValueError(f"Unknown task_type: {task_type}")
            except Exception as exc:
                model_key = str(task["model_key"])
                sample_dir = (
                    args.output_dir
                    / model_key
                    / str(task["prompt_name"])
                    / f"sample_{int(task['sample_idx']):04d}"
                )
                save_json(
                    sample_dir / "worker_error.json",
                    {
                        "worker_name": worker_name,
                        "model_key": model_key,
                        "gpu_id": gpu_id,
                        "task": task,
                        "error": repr(exc),
                    },
                )
                logger.log(
                    f"Task failed for sample={task['sample_idx']} prompt={task['prompt_name']} "
                    f"type={task.get('task_type')}: {exc!r}"
                )
                if str(task.get("task_type")) == "baseline":
                    emit_event(event_queue, "baseline_failed")
                else:
                    emit_event(event_queue, "intervention_failed")
            finally:
                with pending_task_lock:
                    pending_task_counter.value -= 1
                maybe_free_cuda()

        write_csv(worker_paths["baseline_csv"], baseline_rows)
        write_csv(worker_paths["intervention_csv"], intervention_rows)
        logger.log(
            f"Worker finished with {len(baseline_rows)} baseline row(s) and "
            f"{len(intervention_rows)} intervention row(s)."
        )
        emit_event(event_queue, "worker_done", worker=worker_name)
    except Exception as exc:
        logger.log(f"Worker crashed: {exc!r}")
        emit_event(event_queue, "worker_crashed", worker=worker_name, error=repr(exc))
        raise


def rebuild_jsonl_from_sample_dirs(run_root: Path) -> Tuple[int, int]:
    """Rebuild baselines.jsonl and interventions.jsonl from per-sample files.

    The per-sample `baseline.json` and `interventions/*.json` files are the
    authoritative source. The top-level merged JSONLs are convenience
    aggregates and must NOT be written from in-memory state alone, because
    a resume re-run produces empty in-memory records (everything skipped)
    and would silently overwrite a complete prior aggregate with empty
    content. Globbing per-sample files makes this idempotent.
    """
    baseline_rows: List[Dict[str, Any]] = []
    intervention_rows: List[Dict[str, Any]] = []
    # run_root/<model_key>/<prompt_name>/sample_NNNN/{baseline.json, interventions/*.json}
    for baseline_path in sorted(run_root.glob("*/*/sample_*/baseline.json")):
        try:
            baseline_rows.append(json.loads(baseline_path.read_text(encoding="utf-8")))
        except Exception:
            continue
    for intervention_path in sorted(run_root.glob("*/*/sample_*/interventions/*.json")):
        try:
            intervention_rows.append(json.loads(intervention_path.read_text(encoding="utf-8")))
        except Exception:
            continue
    write_jsonl(run_root / "baselines.jsonl", baseline_rows)
    write_jsonl(run_root / "interventions.jsonl", intervention_rows)
    return len(baseline_rows), len(intervention_rows)


def merge_worker_outputs(run_root: Path) -> None:
    worker_dir = run_root / "workers"
    baseline_csvs = sorted(worker_dir.glob("*_baseline_summary.csv"))
    intervention_csvs = sorted(worker_dir.glob("*_intervention_summary.csv"))
    merge_csv_files(baseline_csvs, run_root / "baseline_summary.csv")
    merge_csv_files(intervention_csvs, run_root / "intervention_summary.csv")
    # Reconstruct the JSONL aggregates from per-sample files instead of from
    # per-worker JSONLs. Per-worker JSONLs only contain rows the *current*
    # invocation produced, so a resume re-run with all samples already done
    # would emit empty per-worker JSONLs and overwrite the prior aggregate
    # with nothing.
    rebuild_jsonl_from_sample_dirs(run_root)


def main_omlx(args: ExperimentConfig, logger: RunLogger) -> None:
    require_backend_dependencies("omlx")
    if not args.omlx_api_key:
        raise SystemExit(
            "Could not resolve oMLX API key. Either set OMLX_API_KEY in the "
            "environment, or ensure ~/.omlx/settings.json exists with an "
            "auth.api_key field."
        )
    # Quick connectivity check up-front so we fail in the first second of the
    # run, not after fixture loading.
    try:
        request = urllib.request.Request(
            f"{args.omlx_base_url.rstrip('/')}/v1/models",
            headers={"Authorization": f"Bearer {args.omlx_api_key}"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            response.read(1)
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Could not reach the local oMLX server at {args.omlx_base_url}. "
            f"Start the server (or set OMLX_BASE_URL/OMLX_API_KEY) before "
            f"launching the runner. Underlying error: {exc!r}"
        ) from exc
    if args.run_mechanistic_analysis:
        raise RuntimeError(
            "AJAR_BACKEND=omlx cannot run mechanistic analysis. "
            "Use AJAR_BACKEND=torch with local PyTorch model weights for hidden states, attentions, and interventions."
        )

    run_root = args.output_dir
    output_dir_preexisted = run_root.exists()
    run_root.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "config": redacted_config(args),
        "model_specs": MODEL_SPECS,
        "output_dir_preexisted": output_dir_preexisted,
        "note": "Local oMLX mode runs baseline text generation only.",
    }
    save_json(run_root / "run_config.json", config_payload)
    logger.log(f"Wrote local run config to {run_root / 'run_config.json'}.")
    if output_dir_preexisted and args.resume:
        logger.log(f"Existing output directory detected at {run_root}; resuming completed local baselines.")

    logger.log(f"Loading GSM8K split '{args.split}' with first {args.num_samples} examples.")
    dataset = load_gsm8k_examples(args, logger)
    total_baselines = len(dataset) * len(args.models) * len(args.prompts)
    logger.log(
        f"Prepared up to {total_baselines} local baseline job(s) across "
        f"{len(args.models)} model(s) and {len(args.prompts)} prompt(s)."
    )
    logger.log(f"Using local oMLX request concurrency={args.omlx_concurrency}.")

    jobs: List[Dict[str, Any]] = []
    for sample_idx, example in enumerate(dataset):
        for prompt_name in args.prompts:
            for model_key in args.models:
                jobs.append(
                    {
                        "example": example,
                        "model_key": model_key,
                        "model_name": MODEL_SPECS[model_key],
                        "prompt_name": prompt_name,
                        "sample_idx": sample_idx,
                    }
                )

    baseline_records: List[Dict[str, Any]] = []
    baseline_rows: List[Dict[str, Any]] = []
    completed = 0
    skipped = 0
    failed = 0
    progress = tqdm(total=total_baselines, desc="local omlx", dynamic_ncols=True)

    def execute_job(job: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        return run_omlx_baseline_task(
            example=job["example"],
            args=args,
            run_root=run_root,
            model_key=str(job["model_key"]),
            model_name=str(job["model_name"]),
            prompt_name=str(job["prompt_name"]),
            sample_idx=int(job["sample_idx"]),
            logger=logger,
        )

    def record_success(record: Optional[Dict[str, Any]], row: Optional[Dict[str, Any]]) -> None:
        nonlocal completed, skipped
        if record is None:
            skipped += 1
            return
        baseline_records.append(record)
        if row is not None:
            baseline_rows.append(row)
        completed += 1

    def record_failure(job: Dict[str, Any], exc: Exception) -> None:
        nonlocal failed
        failed += 1
        sample_dir = get_sample_dir(
            run_root,
            str(job["model_key"]),
            str(job["prompt_name"]),
            int(job["sample_idx"]),
        )
        save_json(
            sample_dir / "worker_error.json",
            {
                "worker_name": "local_omlx",
                "model_key": job["model_key"],
                "prompt_name": job["prompt_name"],
                "sample_idx": job["sample_idx"],
                "error": repr(exc),
            },
        )
        logger.log(
            f"Local baseline failed for {job['model_key']} sample={int(job['sample_idx']):04d} "
            f"prompt={job['prompt_name']}: {exc!r}"
        )

    try:
        if args.omlx_concurrency == 1:
            for job in jobs:
                try:
                    record, row = execute_job(job)
                    record_success(record, row)
                except Exception as exc:
                    record_failure(job, exc)
                finally:
                    progress.update(1)
                    if args.show_progress_postfix:
                        progress.set_postfix(
                            {"done": completed, "skipped": skipped, "failed": failed},
                            refresh=False,
                        )
        else:
            with ThreadPoolExecutor(max_workers=args.omlx_concurrency) as executor:
                future_to_job = {executor.submit(execute_job, job): job for job in jobs}
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        record, row = future.result()
                        record_success(record, row)
                    except Exception as exc:
                        record_failure(job, exc)
                    finally:
                        progress.update(1)
                        if args.show_progress_postfix:
                            progress.set_postfix(
                                {"done": completed, "skipped": skipped, "failed": failed},
                                refresh=False,
                            )
    finally:
        progress.close()

    prompt_rank = {prompt_name: index for index, prompt_name in enumerate(args.prompts)}
    model_rank = {model_key: index for index, model_key in enumerate(args.models)}
    baseline_records.sort(
        key=lambda row: (
            int(row["sample_idx"]),
            prompt_rank.get(str(row["prompt_name"]), 999),
            model_rank.get(str(row["model_key"]), 999),
        )
    )
    baseline_rows.sort(
        key=lambda row: (
            int(row["sample_idx"]),
            prompt_rank.get(str(row["prompt_name"]), 999),
            model_rank.get(str(row["model_key"]), 999),
        )
    )
    # Rebuild the aggregate JSONLs from per-sample files instead of from
    # the in-memory baseline_records list. baseline_records only contains
    # rows the *current* invocation produced; on a resume re-run where
    # everything is skipped, it is empty and would silently overwrite a
    # complete prior aggregate with nothing.
    n_base, _ = rebuild_jsonl_from_sample_dirs(run_root)
    write_csv(run_root / "baseline_summary.csv", baseline_rows)
    write_csv(run_root / "intervention_summary.csv", [])
    logger.log(
        f"Local oMLX run complete. Baselines={completed}, skipped={skipped}, "
        f"failed={failed}; aggregate baselines.jsonl now contains {n_base} row(s)."
    )


def main() -> None:
    args = get_config()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(args.verbose_logging, log_path=args.output_dir / "run.log")
    log_environment_banner(args, logger)
    if args.inference_backend == "omlx":
        main_omlx(args, logger)
        return
    require_backend_dependencies("torch")
    set_seed(args.seed)
    validate_parallel_config(args)
    detected_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_parallel, resolved_allocations = resolve_gpu_allocations(args, detected_gpu_count)
    dtype = choose_dtype(args.dtype)
    logger.log(f"Loaded config for {len(args.models)} model(s), {len(args.prompts)} prompt(s), {args.num_samples} sample(s).")
    logger.log(
        f"Resolved torch dtype: {dtype}; CUDA available: {torch.cuda.is_available()}; "
        f"detected_gpus={detected_gpu_count}; parallel={use_parallel}."
    )

    run_root = args.output_dir
    output_dir_preexisted = run_root.exists()
    run_root.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "config": redacted_config(args),
        "resolved_dtype": str(dtype),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "output_dir_preexisted": output_dir_preexisted,
        "effective_parallelize_across_gpus": use_parallel,
        "resolved_model_gpu_allocations": resolved_allocations,
    }
    if torch.cuda.is_available():
        config_payload["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    save_json(run_root / "run_config.json", config_payload)
    logger.log(f"Wrote run config to {run_root / 'run_config.json'}.")
    if output_dir_preexisted and args.resume:
        logger.log(f"Existing output directory detected at {run_root}; resuming from completed baselines.")

    logger.log(f"Loading GSM8K split '{args.split}' with first {args.num_samples} examples.")
    dataset = load_gsm8k_examples(args, logger)
    jobs_by_model, skipped_existing_jobs = build_jobs(dataset, args, run_root)
    total_baselines = sum(len(jobs) for jobs in jobs_by_model.values())
    logger.log(
        f"Prepared {total_baselines} remaining baseline job(s) across all models; "
        f"skipped {skipped_existing_jobs} already-completed baseline job(s)."
    )

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    task_queue = ctx.Queue()
    pending_task_counter = ctx.Value("i", total_baselines)
    pending_task_lock = ctx.Lock()
    processes: List[mp.Process] = []
    worker_gpu_ids = sorted({gpu_id for gpu_ids in resolved_allocations.values() for gpu_id in gpu_ids})
    for model_key in args.models:
        logger.log(f"Model '{model_key}' contributes jobs and can use GPU ids {resolved_allocations[model_key]}.")
        for job in jobs_by_model[model_key]:
            task_queue.put(job)
    logger.log(f"Launching {len(worker_gpu_ids)} generic GPU worker(s) across ids {worker_gpu_ids}.")
    for slot_index, gpu_id in enumerate(worker_gpu_ids):
        worker_name = get_worker_name("pool", gpu_id, slot_index)
        proc = ctx.Process(
            target=worker_main,
            args=(gpu_id, slot_index, task_queue, event_queue, pending_task_counter, pending_task_lock, args),
            name=worker_name,
        )
        processes.append(proc)

    progress = tqdm(total=total_baselines, desc="overall", dynamic_ncols=True)
    completed_baselines = 0
    skipped_baselines = skipped_existing_jobs
    failed_baselines = 0
    completed_interventions = 0
    failed_interventions = 0
    worker_done_count = 0

    def refresh_progress() -> None:
        if not args.show_progress_postfix:
            return
        progress.set_postfix(
            {
                "baselines": completed_baselines,
                "skipped": skipped_baselines,
                "failed": failed_baselines,
                "interventions": completed_interventions,
                "ifail": failed_interventions,
                "workers": f"{worker_done_count}/{len(processes)}",
                "total": progress.total,
            },
            refresh=False,
        )

    for proc in processes:
        logger.log(f"Starting worker process {proc.name}.")
        proc.start()

    refresh_progress()
    try:
        while True:
            alive = any(proc.is_alive() for proc in processes)
            try:
                event = event_queue.get(timeout=1.0)
            except queue.Empty:
                if not alive:
                    break
                continue

            event_type = event["type"]
            if event_type == "log":
                logger.log(f"[{event['worker']}] {event['message']}")
            elif event_type == "baseline_done":
                completed_baselines += 1
                progress.update(1)
            elif event_type == "baseline_skipped":
                skipped_baselines += 1
                progress.update(1)
            elif event_type == "baseline_failed":
                failed_baselines += 1
                progress.update(1)
            elif event_type == "intervention_total_add":
                progress.total = int(progress.total or 0) + int(event["count"])
                progress.refresh()
            elif event_type == "intervention_done":
                completed_interventions += 1
                progress.update(1)
            elif event_type == "intervention_failed":
                failed_interventions += 1
                progress.update(1)
            elif event_type == "worker_done":
                worker_done_count += 1
            elif event_type == "worker_crashed":
                logger.log(f"Worker crash reported from {event['worker']}: {event['error']}")
            refresh_progress()

        while True:
            try:
                event = event_queue.get_nowait()
            except queue.Empty:
                break
            event_type = event["type"]
            if event_type == "log":
                logger.log(f"[{event['worker']}] {event['message']}")
            elif event_type == "baseline_done":
                completed_baselines += 1
                progress.update(1)
            elif event_type == "baseline_skipped":
                skipped_baselines += 1
                progress.update(1)
            elif event_type == "baseline_failed":
                failed_baselines += 1
                progress.update(1)
            elif event_type == "intervention_total_add":
                progress.total = int(progress.total or 0) + int(event["count"])
                progress.refresh()
            elif event_type == "intervention_done":
                completed_interventions += 1
                progress.update(1)
            elif event_type == "intervention_failed":
                failed_interventions += 1
                progress.update(1)
            elif event_type == "worker_done":
                worker_done_count += 1
            elif event_type == "worker_crashed":
                logger.log(f"Worker crash reported from {event['worker']}: {event['error']}")
            refresh_progress()
    finally:
        progress.close()

    bad_exit_codes = []
    for proc in processes:
        proc.join()
        if proc.exitcode not in (0, None):
            bad_exit_codes.append((proc.name, proc.exitcode))
    if bad_exit_codes:
        raise RuntimeError(f"One or more workers exited non-zero: {bad_exit_codes}")

    logger.log("Merging per-worker JSONL and CSV outputs.")
    merge_worker_outputs(run_root)
    logger.log(
        f"Run complete. Baselines={completed_baselines}, skipped={skipped_baselines}, "
        f"failed={failed_baselines}, interventions={completed_interventions}, "
        f"intervention_failures={failed_interventions}."
    )


if __name__ == "__main__":
    main()
