# M1–M2 findings — mlxterp upstream contribution

Date: 2026-05-05/06. Branch: `mlxterp-contrib`. PR:
https://github.com/coairesearch/mlxterp/pull/10 (draft).

This is the close-out for the milestone phase of the mlxterp
contribution. M0 was 1-2h discovery; M1-M2 was the actual
implementation. Total effort: roughly half a day, dramatically less
than the 1-week revised estimate from M0 and the 2-week original
estimate from `mlxterp_contribution_plan.md`.

## Why it was so much faster than estimated

Three things compounded in our favour:

1. **The patch infrastructure already supported persistence across
   forward passes.** M0's generation probe revealed this — the trace's
   `__enter__` patches layers, `__exit__` restores them, and any
   forward calls between (including `mlx_lm.generate`'s autoregressive
   loop) hit the patched layers. The original M1 milestone budgeted
   3-4 days to architect persistent patches; that work was already
   done by the project's existing context-manager design.
2. **mlx-lm's generate already integrates KV cache.** We didn't have
   to write any cache-aware loop logic — passing through to
   `mlx_lm.generate` with the patches active gives us correct cached
   inference for free. M2c validation confirmed: per-token cost is
   ~constant, no scaling with prefix length.
3. **The per-token gating problem is local.** Each intervention
   function is independent; we wrap each in a position-aware closure.
   No shared state across hooks needed. ~30 lines of code.

## Final scope shipped

PR #10 implements all four checkboxes of `CAUSAL_INTERP_ROADMAP.md`
Tier 1 item 3 (`Text Generation with Interventions`):

- [x] `model.generate()` with basic sampling (greedy / temp / top-p /
  top-k via `sampler=make_sampler(...)` pass-through)
- [x] Intervention support during generation
- [x] KV-cache integration (delegated to `mlx_lm.generate`, validated)
- [x] Per-token intervention selectivity
  (`intervention_tokens="all" | "prompt" | "generated" | int | list/range/slice`)

Plus a bonus fix for an unrelated `head_dim` warning that fired
~36 times per call on every Qwen3-family model. mlxterp's
`_compute_attention_weights` was reading `head_dim` directly off the
attention module, but mlx-lm's Qwen2/Qwen3 attention only computes
it locally in `__init__`. Fall back to inferring from
`q_proj.weight.shape[0] // n_heads`. Affects every Qwen3 user, not
just our flow.

## Validation summary

All scripts in `scripts/mlxterp_M*_validate.py` on the AJAR-side
`mlxterp-contrib` branch.

| Validation | Result |
|---|---|
| M1 / `mlxterp_M1_validate.py` | 4/4 contract checks passed |
| M2a / `mlxterp_M2_validate.py` | 10/10 per-token gating checks passed |
| M2b/c / `mlxterp_M2bc_validate.py` | sampling pass-through correct, KV-cache active |
| Qwen3 fix / re-run M1 | 0 head_dim warnings (was 36 per call) |
| Existing mlxterp tests | 8/8 still passing |

Per-call wall time on M5 Pro for 24-token generation under intervention:
~0.5s (vs ~5-10s on torch+MPS for similar workload — about 10x
speedup at this size).

## Work artifacts

On `coairesearch/mlxterp` (upstream, via PR #10):

- `mlxterp/core/trace.py` — `skip_forward` param on `Trace`, head_dim
  fallback in `_compute_attention_weights`, Qwen3Attention recognised
- `mlxterp/model.py` — `InterpretableModel.generate()` and
  `_make_token_gated_intervention()` helper
- `tests/test_generation_with_interventions.py` — 7 contract tests
- `examples/generation_with_interventions.py` — standalone Llama-3.2-1B demo
- `CAUSAL_INTERP_ROADMAP.md` — Tier 1 #3 checkboxes ticked with notes

On `edward-lcl/AJAR` (this repo, `mlxterp-contrib` branch):

- `scripts/mlxterp_smoke.py` — M0 timing baseline
- `scripts/mlxterp_generation_probe.py` — M0 patch-persistence discovery
- `scripts/mlxterp_M1_validate.py` — M1 contract validation
- `scripts/mlxterp_M2_validate.py` — M2a per-token selectivity validation
- `scripts/mlxterp_M2bc_validate.py` — M2b/c sampling + KV-cache validation
- `docs/mlxterp_contribution_plan.md` — original plan with milestones
- `docs/mlxterp_M0_findings.md` — M0 discovery notes
- `docs/mlxterp_M1_M2_findings.md` — this document

## What's next

**This work is upstream-complete and waiting on review.** PR #10 is
draft; once a maintainer takes a pass and we've addressed any
feedback, mark ready for merge. After it lands:

1. **Do the AJAR-side numerical validation** (the original
   `mlxterp_contribution_plan.md` "Validation milestones for our own
   use" section). Run our actual AJAR intervention experiment on 5
   questions through the new mlxterp path. Compare anchor-step
   intervention outcomes against torch+fp16. If anchor selection
   matches in ≥4/5 cases and intervention `correct=True/False`
   matches in ≥80% of cases, we can use mlxterp for AJAR v2.
2. **Benchmark on long Thinking-class outputs** (1500-token traces).
   The MLX-8bit + new generate API should be ~3x faster than our
   current torch+MPS path. If so, AJAR's StrategyQA replication moves
   to mlxterp.
3. **SAE work.** mlxterp's SAEMixin is ready-to-use if Task 7+ goes
   into sparse-feature analysis.

## Time accounting

| Phase | Estimate (original / revised) | Actual |
|---|---|---|
| M0 | 1-2h / 1-2h | ~1h |
| M1 | 3-4d / 1-2d | ~2h |
| M2a | (part of M2) | ~1h |
| M2b+c | 2-3d / 2-3d | ~30min |
| M3 (docs) | 1-2d / 1-2d | ~30min |
| Qwen3 fix | not in plan | ~15min |
| M4 (PR) | 1d / 1d | ~10min |
| **Total** | **~2 weeks / ~1 week** | **~5h** |

Three reasons this came in well under estimate are listed above.
The biggest is that mlxterp's existing context-manager pattern was
strictly more general than we knew — we just had to discover that
M0 morning and the rest fell out.
