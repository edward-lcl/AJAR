#!/usr/bin/env python3
"""Phase 1 / Fable gate, fast + restart-safe via mlx_lm on the local 8-bit Thinking model.

Demonstrates the no-CoT pole repair: the 2507-Thinking template force-opens <think>
and the /no_think soft switch is gone, so instruction-only suppression leaks reasoning.
Forcing </think> in the prompt prefix (NoThinking) yields a clean answer-only pole.

For each GSM8K question, generates the explicit_no_cot prompt two ways and records
generated-token counts:
  - baseline (template as-is): reproduces the contaminated pole (~hundreds of tokens)
  - forced </think> prefill:   should collapse to a short answer (gate target <=16)
Writes per-question rows + prints median for each. Resumable.
"""
import json, os, sys, statistics
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

MODEL = os.path.expanduser("~/.omlx/models/Qwen3-4B-Thinking-2507-MLX-8bit")
OUT = sys.argv[1] if len(sys.argv) > 1 else "outputs/thinking_nocot_gate"
N = int(os.environ.get("N_SAMPLES", "50"))
MAXTOK = int(os.environ.get("MAXTOK", "1024"))
os.makedirs(OUT, exist_ok=True)
JSONL = os.path.join(OUT, "gate.jsonl")

def nocot_messages(q):
    return [
        {"role": "system", "content": "Answer-only mode. /no_think\nOutput exactly one line: \\boxed{number}. Do not include reasoning, explanations, equations, restatements, or units."},
        {"role": "user", "content": f"Question: {q}\n\n/no_think\nRespond only with \\boxed{{number}}."},
    ]

def done_idx():
    s = set()
    if os.path.exists(JSONL):
        for line in open(JSONL):
            try: s.add(int(json.loads(line)["sample_idx"]))
            except Exception: pass
    return s

def gen_len(model, tok, prompt, sampler):
    n = 0
    for _ in stream_generate(model, tok, prompt, max_tokens=MAXTOK, sampler=sampler):
        n += 1
    return n

def main():
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    qs = [ds[i]["question"] for i in range(N)]
    model, tok = load(MODEL)
    sampler = make_sampler(temp=0.0)
    skip = done_idx()
    print(f"loaded {len(qs)} questions; {len(skip)} done", flush=True)
    for idx, q in enumerate(qs):
        if idx in skip: continue
        base = tok.apply_chat_template(nocot_messages(q), tokenize=False, add_generation_prompt=True)
        forced = base.rstrip() + "\n</think>\n\n" if base.rstrip().endswith("<think>") else base + "\n</think>\n\n"
        nb = gen_len(model, tok, base, sampler)
        nf = gen_len(model, tok, forced, sampler)
        with open(JSONL, "a") as f:
            f.write(json.dumps({"sample_idx": idx, "tokens_baseline": nb, "tokens_forced": nf}) + "\n")
        print(f"  sample={idx:02d} baseline={nb} forced={nf}", flush=True)
    rows = [json.loads(l) for l in open(JSONL)]
    mb = statistics.median(r["tokens_baseline"] for r in rows)
    mf = statistics.median(r["tokens_forced"] for r in rows)
    print(f"\nGATE  n={len(rows)}", flush=True)
    print(f"  median no-CoT tokens, template as-is (contaminated) = {mb}", flush=True)
    print(f"  median no-CoT tokens, forced </think> (repaired)     = {mf}", flush=True)
    print(f"  gate (target <=16): {'PASS — clean pole' if mf <= 16 else 'leakage past </think> -> stronger prompt-invariance evidence'}", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
