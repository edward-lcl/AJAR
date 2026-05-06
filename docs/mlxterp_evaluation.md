# mlxterp evaluation — go/no-go for AJAR

Done on the `mlxterp-eval` branch on 2026-05-05. Verdict: **don't migrate now, revisit in v2 when their roadmap Tier 1 lands.**

## What mlxterp is

`coairesearch/mlxterp` (v0.1.0, 8 stars, last push 2026-04-22) is an
nnsight-style mechanistic interpretability library built on Apple MLX.
It wraps any MLX model with `InterpretableModel` and provides:

- **Single-forward activation capture**: `with model.trace(text) as trace:` runs one forward and captures ~196 named activations (Q/K/V projections, MLP gates, attention outputs, layer norms, etc.).
- **Single-forward interventions**: `interventions={"layers.5": iv.scale(0.5)}` modifies activations during that one forward. Seven types: `zero_out`, `scale`, `add_vector`, `replace_with`, `clamp`, `noise`, `compose`.
- **Higher-level analyses**: `logit_lens`, `activation_patching`, `tuned_lens`, `train_sae`.
- **Apple Silicon native** via MLX. ~2-3x faster than torch+MPS for inference on 4B-class models.

The pitch is real. The API is clean (much cleaner than our forward-hook spaghetti). The activation capture is comprehensive. The intervention library overlaps with what we built.

## Why it can't replace our pipeline today

Our experiment requires **intervention-during-generation**: apply a hook at a specific layer/position, then **generate the continuation autoregressively** with the hook active, and measure how the final answer changes.

mlxterp v0.1 does not support this:

1. `model.trace()` is a context manager that runs **exactly one forward pass** when you `__enter__` the context. The forward pass uses the input you passed to `trace()`. There's no `model.generate(text, interventions=...)` and no per-token intervention application.

2. On `__exit__`, the trace **restores all patched layers**, so any patches applied during the trace are undone. Even if we manually called `model.generate()` inside the `with` block, the patches would be gone the moment we left it.

3. **The maintainers explicitly acknowledge this gap.** From `CAUSAL_INTERP_ROADMAP.md` in their repo:

   > ### 3. Text Generation with Interventions
   > **Why**: Can't study in-context learning, induction heads in practice, or intervention effects on generated text without autoregressive generation. pyvene supports per-token interventions during generation.
   > - [ ] `model.generate()` with basic sampling (greedy, temperature, top-k, top-p)
   > - [ ] KV-cache integration (interventions must work with cached inference)

   This is listed as **Tier 1 — Critical for Causal Interpretability**, with checkboxes still open as of 2026-04-22.

In short: the maintainers know this is missing, are planning to add it, and haven't yet. Until they ship it, our intervention phase can't run on mlxterp.

## What mlxterp could replace

We could partially migrate: keep torch+MPS for intervention generation, switch the **static analysis** phase (one forward + capture hidden states + capture attentions + score anchors) to mlxterp.

That phase is ~30% of our per-baseline wall time at the new optimised config. If mlxterp gives 2x speedup on that phase, we save ~15% of total runtime. Real but not transformative.

The migration cost would be:

- Port `run_full_forward_hidden_analysis` to use `model.trace()` + `trace.activations[...]`
- Port `probe_attention_vectors` to use the captured per-layer attention from the same trace
- Validate numerical equivalence: anchor selection on torch+fp16 vs mlxterp+8bit-MLX should match in top-3 ranks. Quantization (8-bit MLX vs fp16 torch) may shift step_scores at the borderline.
- Maintain two backends because the intervention phase still uses torch.

That's ~1 week of focused work for a 15% wall-clock saving. **Bad ROI** at this stage.

## What's plausibly worth doing later

Three scenarios in which the calculus changes:

1. **mlxterp ships Tier 1 generation.** Their roadmap specifically calls out "model.generate() with KV-cache + intervention support". When that lands and is documented, intervention generation moves from torch+MPS to MLX, which is genuinely 2-3x faster. At that point migrating is high-value. Watch their releases.

2. **We start a new dataset / model family.** For Task 4 (StrategyQA) or any new replication, there's no sunk cost in the existing torch path. Starting fresh on mlxterp would be cheaper than migrating.

3. **SAE training becomes part of the roadmap.** mlxterp's `SAEMixin` (~25KB of code) gives us sparse autoencoder training out of the box. If we want to do SAE-feature-level analysis later (which is a natural Task 7+ extension), mlxterp saves us implementing SAEs ourselves.

## Concrete recommendation

- **Stay on torch+MPS for the current Task 4-6 work.** Our pipeline is finished, validated, and produces statistically significant headline results. No reason to take migration risk.
- **Pin a tracking issue or TODO note** to watch mlxterp's "Generation with Interventions" feature. When it lands, run a 1-day numerical validation (do anchor ranks match between backends?) and benchmark; if both check out, switch for the next round.
- **Consider mlxterp specifically for SAE work** if and when Task 7 expands into SAE features. The static-analysis API is good and the SAE infrastructure is a real gift.

The branch this evaluation lives on (`mlxterp-eval`) can stay open as a marker. If/when we revisit, this doc is the starting point.

## What I actually ran

To be honest about the depth of the evaluation:

- ✅ Cloned `coairesearch/mlxterp` v0.1.0
- ✅ Installed `mlx==0.29.3`, `mlx-lm==0.29.1`, `mlxterp==0.1.0` in our user pip env
- ✅ Read `mlxterp/core/trace.py`, `mlxterp/core/intervention.py`, `examples/basic_usage.py`, `examples/activation_patching_example.py`, `mlxterp/__init__.py`
- ✅ Read `CAUSAL_INTERP_ROADMAP.md` to understand what's planned
- ⚠️ Did **not** run a full numerical benchmark vs torch on Qwen3-4B (the MLX-8bit weights weren't actually cached locally, only the configs were; would need a ~4GB download to run the smoke). The decision verdict is sufficient from code review alone given the API-gap finding.

If the team wants to do the numerical benchmark anyway (e.g. as a baseline for v2 when Tier 1 lands), pull this branch, run `huggingface-cli download lmstudio-community/Qwen3-4B-Instruct-2507-MLX-8bit`, then write a small comparison script that loads the same question through both backends and compares anchor ranks.
