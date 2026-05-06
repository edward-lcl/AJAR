# mlxterp upstream contribution — implementation plan

Companion doc to `mlxterp_evaluation.md`. The evaluation concluded that
mlxterp v0.1 doesn't support intervention-during-generation, which is on
their roadmap as Tier 1 missing functionality. This doc lays out a plan
to implement that feature upstream and contribute it back.

## Why we're doing this

Three reasons in increasing order of selfishness:

1. **It's a real public good.** The mlxterp maintainer (Sigurd Schacht
   at coairesearch) has explicitly listed "Text Generation with
   Interventions" as Tier 1 critical-missing in their public roadmap.
   They want this. They reference pyvene as the standard. A working
   implementation lands an MIT-licensed feature that any Apple Silicon
   researcher doing mech interp will use.
2. **It unblocks our v2 work.** Once this lands upstream, our AJAR
   intervention pipeline can move from torch+MPS to MLX-native. That's
   roughly 2-3x faster generation on Apple Silicon for 4B-class models.
   Future Algoverse projects that hit the same gap also benefit.
3. **It's a portable open-source contribution.** Useful for personal
   profile (Algoverse implementation team work, future research grant
   applications, OpenAI open-source funding which awards API credits
   to active OSS contributors).

We will **not** promise the AJAR team that their work will switch to
mlxterp. The contribution is its own deliverable, not gated on AJAR
deadlines.

## Scope (from mlxterp's roadmap, verbatim)

From `CAUSAL_INTERP_ROADMAP.md` in the mlxterp repo, Tier 1 item 3:

> ### 3. Text Generation with Interventions
> **Why**: Can't study in-context learning, induction heads in
> practice, or intervention effects on generated text without
> autoregressive generation. pyvene supports per-token interventions
> during generation.
> ```python
> # Basic generation
> output = model.generate("The capital of France is", max_tokens=20)
> # Generation with persistent intervention
> output = model.generate(
>     "The Eiffel Tower is in",
>     interventions={"layers.5": add_vector(steering_vec)})
> # Per-token intervention (advanced)
> output = model.generate(
>     "The capital is",
>     interventions={"layers.5": add_vector(steering_vec)},
>     intervention_tokens="all")  # or specific token positions
> ```
> - [ ] `model.generate()` with basic sampling (greedy, temperature, top-k, top-p)
> - [ ] KV-cache integration (interventions must work with cached inference)
> **Reference**: pyvene per-token interventions, mlx-lm generate utilities

Four checkboxes, all open. That's the spec.

## Architectural challenge

The current `Trace` class in `mlxterp/core/trace.py`:

1. Patches all model layers in `__enter__`
2. Runs **one** forward pass with the input given to `trace(...)`
3. Captures activations during that forward
4. Restores all original layers in `__exit__`

For generation, we need:

1. Patches stay active across **many** forward passes (one per generated token)
2. Each forward consumes the previous step's KV cache (so we don't redo
   the prefix on every token)
3. Interventions fire on each token's forward pass — either persistently
   (every token) or selectively (a specified set of token positions)
4. Activations from generated tokens are reachable, not just the prompt

This is a real architectural shift but well-bounded. mlx-lm's existing
`generate_step` and `generate` utilities already handle the loop +
sampling + KV-cache. We just need to ensure mlxterp's intervention
patches survive across calls.

## Milestone plan

**Branch**: `mlxterp-contrib` (separate from `mlxterp-eval` so
evaluation work stays preserved).

### M0 — Reproducible setup (1-2 hours)

- Fork `coairesearch/mlxterp` to our GitHub
- Clone the fork at a pinned SHA
- Set up `mlxterp-contrib` branch in the fork
- Write a smoke script that loads Qwen3-4B-Instruct (8-bit MLX), runs
  one `model.trace(text)` and verifies activation capture works
- Run pyvene's per-token-intervention demo on a toy model so we have a
  reference behaviour to match

### M1 — Persisted patches across forwards (3-4 days)

The smallest meaningful change. Refactor the trace so patches can be
applied without auto-tearing-down:

- Extract patch/restore logic into a `PatchManager` separate from
  `Trace.__enter__`/`__exit__`
- `Trace` keeps current behaviour (one-shot)
- Add a new `model.with_interventions(interventions={...})` context
  manager that uses `PatchManager` directly without running a forward
- Inside that context, the patches stay applied; user can call
  `model.generate(...)` (or any forward) and patches fire on each
  forward
- Validate against a small toy model: same input + same intervention
  → same output as pyvene

### M2 — `model.generate()` with interventions (3-5 days)

Use mlx-lm's `generate_step` as the inner loop:

- Add `InterpretableModel.generate(text, max_tokens, interventions=None,
  intervention_tokens=None, **sampling_kwargs)`
- Implement basic sampling: greedy, temperature, top-k, top-p (mlx-lm
  has these helpers; just wire them through)
- KV-cache integration: ensure each generation step uses
  `mlx_lm.generate_step`'s cache so we don't redo the prefix
- Optional `intervention_tokens` argument: apply interventions only on
  specified token positions (e.g. `[0, 1, 2]` = prompt tokens only,
  `"all"` = every token, `slice(N, None)` = generated tokens only)
- Tests: greedy decoding produces deterministic output; intervention
  applied at first generated token but not after correctly toggles;
  KV-cache hits are observable via mx.eval timing

### M3 — Documentation + examples (1-2 days)

- Update mlxterp's `examples/` directory with a generation+intervention
  demo (replicate pyvene's `add_vector` steering example on Llama)
- Add API docs to mkdocs site
- Write a tutorial-style notebook
- Update `CAUSAL_INTERP_ROADMAP.md` checkboxes

### M4 — Upstream PR (1 day if M0-M3 are clean)

- Open PR to `coairesearch/mlxterp` from our fork
- PR description points at roadmap Tier 1 item 3
- Include before/after benchmarks (generation tok/s with no
  intervention vs with intervention) on Qwen3-4B-Instruct-MLX-8bit
- Tag the maintainer politely; expect review back-and-forth

**Total estimate**: ~2 weeks of focused work, plus review cycle.

## Validation milestones for our own use

Independent of the upstream PR landing, before we'd actually switch
AJAR to mlxterp:

1. **Numerical equivalence on a small slice.** Run our existing AJAR
   intervention experiment on 5 questions through the new mlxterp
   path. Compare anchor-step intervention outcomes against torch+fp16.
   Pass criterion: top-3 anchor selection matches in ≥4/5 cases;
   per-intervention `correct=True/False` matches in ≥80% of cases.
   (Some divergence expected because MLX-8bit ≠ torch-fp16; we just
   need it to not flip the headline result.)
2. **Benchmarks.** Time per (baseline + 6 interventions) on a long
   Thinking trace, mlxterp vs current torch+MPS path. Pass criterion:
   ≥2x faster end-to-end. (If only ~1.2x, the migration cost isn't
   worth it.)
3. **Resume/idempotency.** Confirm the intervention manifests +
   per-sample baseline.json files mlxterp produces are compatible
   with our existing aggregator. If we have to rewrite the aggregator,
   note it as part of the migration cost.

## Funding angles to consider

Since this is genuine OSS work on a research-aligned library, a few
external programs might fund the time:

- **OpenAI open-source program** — awards API credits to active
  open-source contributors. Useful for downstream Algoverse projects
  that need GPT-4 calls for paraphrase generation, evaluation, etc.
- **Apple's Metal / MLX research initiatives** — coairesearch is
  already adjacent to MLX; a PR landing in their library is a natural
  fit for any Apple-funded credits/hardware programs.
- **Algoverse research credits** — if Algoverse has internal compute
  budgets, this work is the kind of "tooling that helps multiple
  teams" that those budgets typically prefer.

None of these are guaranteed; mention them at PR time, not before.

## What we'd leave on the table by NOT doing this

- AJAR v2 stays on torch+MPS. Each MI run takes ~7h on Mac vs the
  ~2-3h it could take on MLX. Multiplied across StrategyQA, paired
  benchmark, Thinking-only deep dives, etc., that's potentially weeks
  of cumulative wall-time we're paying for code we already know is
  slower than necessary.
- Other Algoverse teams hitting the same wall continue hitting it.
- The mlxterp Tier 1 item stays open indefinitely, or someone else
  ships it (which is fine for the field, just less useful for us).

## When to start

This branch (`mlxterp-eval`, soon to become `mlxterp-contrib`) holds
the plan. Actual implementation hasn't started. Come back to this
when:

- AJAR Task 4 (StrategyQA) is either deferred or starting (wherever it
  sits, the question is "how much is mlxterp going to save us on the
  next dataset?")
- ~2 weeks of focused implementation time is genuinely available
- Someone wants to come review the milestone plan before kickoff

The first concrete step at kickoff is M0 — fork the repo, get a
smoke script running, replicate pyvene's reference behaviour on a
toy model. Estimated 1-2 hours; ends with "we have a working
baseline to validate against".
