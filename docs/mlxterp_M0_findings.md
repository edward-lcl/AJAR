# M0 findings — mlxterp contribution kickoff

Date: 2026-05-05/06. Branch: `mlxterp-contrib`.

This is the kickoff milestone for contributing
intervention-during-generation upstream to mlxterp. The plan is in
`docs/mlxterp_contribution_plan.md`. M0 was a 1-2 hour discovery pass
to confirm the basic stack works and to scope M1 accurately. **Major
discovery: M1 is much smaller than the original plan estimated.**

## Setup

- Forked `coairesearch/mlxterp` to `edward-lcl/mlxterp`
- Cloned the fork into `third_party/mlxterp/` (gitignored), with the
  upstream remote configured for fetching future updates
- Created the fork's `feature/generation-with-interventions` branch
  off mlxterp `main` at `2935fbf` (post-`Merge pull request #8`)
- AJAR-side branch: `mlxterp-contrib` (off `mlxterp-eval`), where work
  artefacts (smoke scripts, findings docs, comparison harnesses) live.
  Actual mlxterp source changes live in the fork.
- Installed `pyvene==0.1.8` as the per-token intervention reference.
- Pulled the actual Qwen3-4B-Instruct-2507-MLX-8bit weights (4 GB).
  Earlier eval session had only the 15 MB of configs; that's why the
  smoke test in the eval session hung.

## Smoke test (`scripts/mlxterp_smoke.py`)

**Hardware**: M5 Pro Mac, MLX 0.29.3, mlx-lm 0.29.1, mlxterp 0.1.0.

| step | wall time | notes |
|---|---:|---|
| `mlx_lm.load(MLX-8bit)` | 1.51s | vs ~2 minutes for torch fp16 |
| `InterpretableModel(...)` wrap | 0.00s | discovers + wraps 36 decoder layers |
| `with model.trace(prompt):` (12 tokens) | 0.02s | captures 544 named activations |
| `with model.trace(prompt, interventions={...})` | 0.01s | single-forward intervention works |

The static-analysis path is **dramatically** faster than torch+MPS.
For comparison, our AJAR torch+MPS pipeline takes ~30s for one
analysis pass on a 200-token output. mlxterp does the equivalent in
0.02s on a 12-token input — extrapolating to longer inputs, we'd
expect roughly 100x speedup on this phase.

**Compatibility note**: trace emits a warning per Qwen3 self_attn layer
about `'Attention' object has no attribute 'head_dim'`. mlxterp's
attention-weight computation doesn't handle Qwen3's grouped-query
attention layout (32 query heads vs 8 KV heads). The intervention
mechanism still works; only the *attention-weight* probe in the trace
fails. We should note this in the upstream PR — likely a small fix in
`trace.py` to read `head_dim` from the model config rather than the
attention module.

## Generation probe (`scripts/mlxterp_generation_probe.py`) — major finding

The original M1 plan assumed we'd need to refactor mlxterp's `Trace`
class to allow patches to persist across multiple forward passes. **It
turns out we don't. They already do.**

The trace's `__enter__` patches the model layers and `__exit__`
restores them. Anything that calls a forward pass on the model
*between* enter and exit hits the patched layers. We tested by calling
`mlx_lm.generate(...)` inside a `with model.trace(prompt,
interventions={"layers.5": iv.zero_out}):` block:

```
[step 4] generate INSIDE a trace block where patches are still active
  time: 0.47s
  output: '!!!!!!!!!!!!!!!!!!!!!!!!'
  differs from baseline? YES (intervention is firing during generation!)
```

Output is gibberish because layer 5 is zeroed on every generation
step. **The intervention is firing.** Compare to the same generate
outside the trace context, which gives the clean baseline.

This means M1 (originally estimated 3-4 days for "persisted patches
across forwards") **collapses to API ergonomics**: wrap the existing
patch-persistence behaviour in a clean `model.generate(text,
interventions=...)` call, without forcing the user to construct a
no-op trace and call `mlx_lm.generate` inside it.

Side-effects we still need to handle:

1. The trace's `__enter__` runs one *unwanted* forward pass on the
   prompt input given to `trace(...)`. For a clean
   `model.generate(text, interventions=...)`, we want to skip this.
   We can either pass a 1-token dummy input (cheap) or refactor to
   make the inner forward optional.
2. The trace also captures 544 activations during that wasted forward,
   eating memory we don't need. Same fix.
3. KV-cache: `mlx_lm.generate` already uses cached inference per
   token. Our hack benefits from this for free; the only question is
   whether the cache survives across multiple `model.generate` calls
   in the same trace, which we don't need to support for v1.
4. `intervention_tokens` (from the upstream roadmap) — applying
   interventions only on specific token positions — needs explicit
   logic. The hook fires on every forward pass regardless of which
   token is being generated; selective firing requires inspecting the
   forward call's input shape and gating the intervention accordingly.

## Revised M1-M4 estimate

Original plan (mlxterp_contribution_plan.md):

- M1: persisted patches (3-4 days)
- M2: model.generate + sampling + KV-cache (3-5 days)
- M3: docs (1-2 days)
- M4: PR (1 day)

**Revised**:

- ~~M1: persisted patches~~ → **already works, replace with "wrap the
  hack in a clean API and skip the unwanted prompt-forward"** (1-2 days)
- M2: `intervention_tokens` selectivity + sampling helpers + KV-cache
  validation (2-3 days)
- M3: docs + examples (1-2 days)
- M4: PR (1 day)

**Total ~1 week** (down from ~2 weeks). Plus pyvene-validation work to
ensure our generate output matches their reference behaviour on toy
models.

## What's still ahead

- M0g (this commit): document findings, commit smoke + probe scripts
  on AJAR-side `mlxterp-contrib`.
- Then: write the pyvene-vs-mlxterp parity test on a toy model
  (single-layer linear "model" so both libraries produce identical
  numbers given identical interventions).
- Then: M1 — clean `model.generate(text, interventions=...)` API on
  the fork's `feature/generation-with-interventions` branch.

## Bug filed implicitly

The Qwen3 attention-weight warning is a real upstream issue — when
they next ship, anyone using a Qwen3 family model will see it. We
should fix it as part of our PR (it's in scope for "make this work
properly with current SOTA models"). Will be a small read of the model
config to fetch `num_attention_heads` and `num_key_value_heads`
explicitly rather than via `module.head_dim`.
