#!/usr/bin/env python3
"""Thinking HCDS recompute against a CLEAN no-CoT pole (NoThinking prefill).

The headline weakness: Thinking-2507's /no_think pole is contaminated (~491 tok of
reasoning). This recomputes the Thinking HCDS over latency + entropy features with
the no-CoT pole built two ways, so we can report what the clean pole changes:
  - contaminated : explicit_no_cot as-is (template force-opens <think>)
  - clean        : NoThinking prefill (Ma et al. 2504.09858), median ~7 tokens

HCDS = D(neutral, nocot) - D(neutral, cot) per question, features z-scored across
questions within each (condition) cell (matches compute_hcds.py), Euclidean
distance in standardized space; 1000x bootstrap 95% CI (seed 17).
Generation is resumable (per-cell jsonl).
"""
import json, os, sys, math, random
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx

M = os.path.expanduser("~/.omlx/models/Qwen3-4B-Thinking-2507-MLX-8bit")
OUT = "outputs/thinking_clean_pole_hcds"
N = int(os.environ.get("N_SAMPLES", "50"))
os.makedirs(OUT, exist_ok=True)
JL = os.path.join(OUT, "features.jsonl")
DUMMY = "\nOkay, I think I have finished thinking.\n</think>\n\n"

def sys_user(q, mode):
    if mode == "explicit_cot":
        return [{"role": "system", "content": "Think through this step by step showing your reasoning process then provide your final answer. Put the final numeric answer in \\boxed{}."},
                {"role": "user", "content": f"Question: {q}"}]
    if mode == "neutral_strict":
        return [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Question: {q}"}]
    # both no_cot variants share the answer-only instruction
    return [{"role": "system", "content": "Answer-only mode. /no_think\nOutput exactly one line: \\boxed{number}. Do not include reasoning, explanations, equations, restatements, or units."},
            {"role": "user", "content": f"Question: {q}\n\n/no_think\nRespond only with \\boxed{{number}}."}]

CELLS = ["explicit_cot", "neutral_strict", "nocot_contam", "nocot_clean"]

def done():
    s = set()
    if os.path.exists(JL):
        for l in open(JL):
            try:
                r = json.loads(l); s.add((r["cell"], int(r["idx"])))
            except Exception: pass
    return s

def gen_features(model, tok, prompt, sampler, maxtok):
    ents = []; last = None
    for r in stream_generate(model, tok, prompt, max_tokens=maxtok, sampler=sampler):
        lp = r.logprobs
        ents.append(float(-mx.sum(mx.exp(lp) * lp)))
        last = r
    n = len(ents)
    if n == 0:
        return None
    mean = sum(ents) / n
    xs = list(range(n)); xb = (n - 1) / 2
    den = sum((x - xb) ** 2 for x in xs)
    slope = (sum((xs[i] - xb) * (ents[i] - mean) for i in range(n)) / den) if den else 0.0
    tps = getattr(last, "generation_tps", 0) or 0
    lat = (1.0 / tps) if tps > 0 else None
    return {"latency": lat, "ent_mean": mean, "ent_slope": slope, "tokens": n}

def generate():
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    model, tok = load(M); sampler = make_sampler(temp=0.0)
    skip = done()
    for cell in CELLS:
        mode = "explicit_no_cot" if cell.startswith("nocot") else cell
        maxtok = 64 if cell == "nocot_clean" else int(os.environ.get("LONG_MAXTOK", "512"))
        for i in range(N):
            if (cell, i) in skip: continue
            base = tok.apply_chat_template(sys_user(ds[i]["question"], mode), tokenize=False, add_generation_prompt=True)
            prompt = base
            if cell == "nocot_clean":
                prompt = base.rstrip() + DUMMY if base.rstrip().endswith("<think>") else base + DUMMY
            f = gen_features(model, tok, prompt, sampler, maxtok)
            if f is None: continue
            f.update({"cell": cell, "idx": i})
            with open(JL, "a") as fh: fh.write(json.dumps(f) + "\n")
            print(f"  {cell} idx={i:02d} lat={f['latency']} ent={f['ent_mean']:.3f} tok={f['tokens']}", flush=True)

def zscore(vals):
    xs = [v for v in vals if v is not None]
    if len(xs) < 2: return {i: 0.0 for i in range(len(vals))}
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 or 1.0
    return [(v - m) / sd if v is not None else 0.0 for v in vals]

def dist(a, b):
    return math.sqrt(sum((a[k] - b[k]) ** 2 for k in a))

def hcds_for(rows_by_cell, nocot_cell):
    # rows_by_cell[cell] = list indexed by question of {latency,ent_mean,ent_slope}
    feats = ["latency", "ent_mean", "ent_slope"]
    cells = {c: rows_by_cell[c] for c in ["explicit_cot", "neutral_strict", nocot_cell]}
    idxs = sorted(set.intersection(*[set(d.keys()) for d in cells.values()]))
    # z-score each feature within each cell across shared questions
    z = {}
    for c in cells:
        z[c] = {}
        for f in feats:
            zs = zscore([cells[c][i].get(f) for i in idxs])
            for k, i in enumerate(idxs):
                z[c].setdefault(i, {})[f] = zs[k]
    per_q = []
    for i in idxs:
        d_nocot = dist(z["neutral_strict"][i], z[nocot_cell][i])
        d_cot = dist(z["neutral_strict"][i], z["explicit_cot"][i])
        per_q.append(d_nocot - d_cot)
    return per_q

def boot_ci(xs, nb=1000, seed=17):
    rng = random.Random(seed); n = len(xs); means = []
    for _ in range(nb):
        s = [xs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    return means[int(0.025 * nb)], means[int(0.975 * nb)]

def compute():
    rows = [json.loads(l) for l in open(JL)]
    by = {c: {} for c in CELLS}
    for r in rows:
        by[r["cell"]][int(r["idx"])] = r
    print(f"\n=== Thinking HCDS (latency + entropy, n shared) ===")
    for label, cell in [("CONTAMINATED pole (/no_think)", "nocot_contam"), ("CLEAN pole (NoThinking)", "nocot_clean")]:
        pq = hcds_for(by, cell)
        if not pq:
            print(f"{label}: no shared questions"); continue
        m = sum(pq) / len(pq); lo, hi = boot_ci(pq)
        verdict = "neutral aligns with CoT" if lo > 0 else ("aligns with no-CoT" if hi < 0 else "ambiguous (CI crosses 0)")
        print(f"{label}: HCDS={m:+.3f} [{lo:+.3f}, {hi:+.3f}] n={len(pq)} -> {verdict}")
    print("DONE")

if __name__ == "__main__":
    generate()
    compute()
