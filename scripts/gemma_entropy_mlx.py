#!/usr/bin/env python3
"""Fast, restart-survivable Gemma entropy via mlx_lm on the local 4-bit model.

oMLX returns no logprobs and the torch fp32 path is too slow to survive this
machine's frequent restarts. This loads gemma-3-4b-it-4bit directly, generates
greedily on the SAME 3 prompts x 50 GSM8K questions as the behavioral run, and
computes per-generated-token Shannon entropy from stream_generate logprobs.

Writes rows {model_key, prompt_name, sample_idx, mean_entropy_generated,
entropy_slope_generated} to <out>/baselines.jsonl in the mi-dir format that
build_task6_table.py reads. Appends per (sample,prompt) and skips completed
keys on resume.
"""
import json, os, sys
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

MODEL = os.path.expanduser("~/.omlx/models/gemma-3-4b-it-4bit")
OUT = sys.argv[1] if len(sys.argv) > 1 else "outputs/gemma_mlx_entropy"
N = int(os.environ.get("N_SAMPLES", "50"))
MAXTOK = int(os.environ.get("MAXTOK", "1024"))
os.makedirs(OUT, exist_ok=True)
JSONL = os.path.join(OUT, "baselines.jsonl")

PROMPTS = {
    "explicit_cot": lambda q: [
        {"role": "system", "content": "Think through this step by step showing your reasoning process then provide your final answer. Put the final numeric answer in \\boxed{}."},
        {"role": "user", "content": f"Question: {q}"},
    ],
    "explicit_no_cot": lambda q: [
        {"role": "system", "content": "Answer-only mode. /no_think\nOutput exactly one line: \\boxed{number}. Do not include reasoning, explanations, equations, restatements, or units."},
        {"role": "user", "content": f"Question: {q}\n\n/no_think\nRespond only with \\boxed{{number}}."},
    ],
    "neutral_strict": lambda q: [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Question: {q}"},
    ],
}

def done_keys():
    keys = set()
    if os.path.exists(JSONL):
        for line in open(JSONL):
            try:
                r = json.loads(line)
                keys.add((r["prompt_name"], int(r["sample_idx"])))
            except Exception:
                pass
    return keys

def main():
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    questions = [ds[i]["question"] for i in range(N)]
    print(f"loaded {len(questions)} GSM8K questions", flush=True)

    model, tokenizer = load(MODEL)
    sampler = make_sampler(temp=0.0)  # greedy, matches temperature=0 protocol
    skip = done_keys()
    print(f"resuming; {len(skip)} (prompt,sample) cells already done", flush=True)

    for cond, builder in PROMPTS.items():
        for idx, q in enumerate(questions):
            if (cond, idx) in skip:
                continue
            prompt = tokenizer.apply_chat_template(
                builder(q), tokenize=False, add_generation_prompt=True
            )
            ents = []
            for resp in stream_generate(model, tokenizer, prompt,
                                        max_tokens=MAXTOK, sampler=sampler):
                lp = resp.logprobs              # log-softmax over vocab for this step
                H = float(-mx.sum(mx.exp(lp) * lp))
                ents.append(H)
            if ents:
                n = len(ents)
                mean = sum(ents) / n
                # least-squares slope of entropy vs token index
                xs = list(range(n)); xbar = (n - 1) / 2; ybar = mean
                denom = sum((x - xbar) ** 2 for x in xs)
                slope = (sum((xs[i] - xbar) * (ents[i] - ybar) for i in range(n)) / denom) if denom else 0.0
            else:
                mean, slope = None, None
            with open(JSONL, "a") as f:
                f.write(json.dumps({
                    "model_key": "gemma", "model_name": "gemma-3-4b-it-4bit",
                    "prompt_name": cond, "sample_idx": idx,
                    "mean_entropy_generated": mean,
                    "entropy_slope_generated": slope,
                    "num_entropy_tokens": len(ents),
                }) + "\n")
            print(f"  {cond} sample={idx:02d} tokens={len(ents)} entropy={mean}", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
