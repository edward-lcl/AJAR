#!/usr/bin/env python3
"""Full n=50 gate with the FAITHFUL NoThinking prefill (Ma et al. 2504.09858).

Corrects the earlier empty-</think> gate. The NoThinking method prefills a
fabricated completed-thought inside the block:  <think>\nOkay, I think I have
finished thinking.\n</think>\n\n  -> the model proceeds straight to the answer.

Per question records generated-token count, whether the output is a clean
answer-only pole (short + boxed), and the text. Resumable.
"""
import json, os, sys, statistics
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

M = os.path.expanduser("~/.omlx/models/Qwen3-4B-Thinking-2507-MLX-8bit")
OUT = sys.argv[1] if len(sys.argv) > 1 else "outputs/nothinking_gate_full"
N = int(os.environ.get("N_SAMPLES", "50"))
MAXTOK = int(os.environ.get("MAXTOK", "768"))
DUMMY = "\nOkay, I think I have finished thinking.\n</think>\n\n"
os.makedirs(OUT, exist_ok=True)
JL = os.path.join(OUT, "gate.jsonl")

def done():
    s = set()
    if os.path.exists(JL):
        for l in open(JL):
            try: s.add(int(json.loads(l)["sample_idx"]))
            except Exception: pass
    return s

def msgs(q):
    return [{"role": "system", "content": "Answer-only mode. /no_think\nOutput exactly one line: \\boxed{number}. Do not include reasoning, explanations, equations, restatements, or units."},
            {"role": "user", "content": f"Question: {q}\n\n/no_think\nRespond only with \\boxed{{number}}."}]

def main():
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    model, tok = load(M); sampler = make_sampler(temp=0.0)
    skip = done()
    print(f"N={N}; {len(skip)} done", flush=True)
    for i in range(N):
        if i in skip: continue
        base = tok.apply_chat_template(msgs(ds[i]["question"]), tokenize=False, add_generation_prompt=True)
        p = base.rstrip() + DUMMY if base.rstrip().endswith("<think>") else base + DUMMY
        out = ""; n = 0
        for r in stream_generate(model, tok, p, max_tokens=MAXTOK, sampler=sampler):
            out += r.text; n += 1
        clean = (n <= 24) and ("\\boxed" in out)
        with open(JL, "a") as f:
            f.write(json.dumps({"sample_idx": i, "tokens": n, "clean": clean, "text": out[:200]}) + "\n")
        print(f"  s={i:02d} tok={n} clean={clean}", flush=True)
    rows = {int(json.loads(l)["sample_idx"]): json.loads(l) for l in open(JL)}.values()
    rows = list(rows)
    toks = [r["tokens"] for r in rows]; nclean = sum(1 for r in rows if r["clean"])
    print(f"\nNOTHINKING GATE  n={len(rows)}", flush=True)
    print(f"  median tokens = {statistics.median(toks):.0f}  mean = {statistics.mean(toks):.0f}", flush=True)
    print(f"  clean answer-only pole = {nclean}/{len(rows)} ({100*nclean/len(rows):.0f}%)", flush=True)
    print(f"  gate (>=90% clean): {'PASS — clean Thinking no-CoT pole achievable' if nclean/len(rows)>=0.9 else 'PARTIAL'}", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
