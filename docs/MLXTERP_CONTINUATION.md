# mlxterp continuation plan

State snapshot as of 2026-05-05 night, for the next session.

This is the "what's done, what's next, how to pick up" doc. Open this
first whenever you come back to the mlxterp / AJAR work.

---

## Where things are right now

### 8 PRs open on `coairesearch/mlxterp`, all ready for review

All under `https://github.com/coairesearch/mlxterp/pull/<N>`.

**Tier 1 (causal interpretability core)** — 5 PRs:

| PR | What | Branch on `edward-lcl/mlxterp` | Validation |
|---|---|---|---|
| #10 | Generation with interventions (`model.generate` + per-token gating) | `feature/generation-with-interventions` | 5/5 parity vs torch+fp16, 5x speedup vs oMLX HTTP, +1.69s/q intervention overhead = hook fires per-token |
| #11 | Activation patching with positions + causal metrics + heatmap | `feature/activation-patching-positions` | logit_diff / kl_divergence / cross_entropy_diff metrics, position-resolved |
| #12 | Multi-input `causal_trace(clean, corrupted)` | `feature/multi-input-trace` | clean/corrupted/.patch/.metric ergonomic API |
| #13 | Attribution patching (gradient-based) | `feature/attribution-patching` | Spearman 0.729 vs brute-force, 5x speedup, identity-zero on clean==corrupted |
| #14 | Path patching (single-edge MVP) | `feature/path-patching` | sparse circuit on Llama-1B Paris/London: layers.15.self_attn=+2.21 dominant |

**Tier 2 (competitive feature set)** — 3 PRs:

| PR | What | Branch | Validation |
|---|---|---|---|
| #15 | Direct Logit Attribution (DLA) | `feature/direct-logit-attribution` | sum-check identity gap = -0.0006 on Llama-1B |
| #16 | Residual stream view (`resid_pre`/`mid`/`post` + decompose) | `feature/residual-stream-access` | max identity gap = **0.000000** at every layer |
| #17 | Patching visualisations (heatmap + 3 bar charts) | `feature/patching-visualization` | 7 smoke tests on headless Agg, end-to-end PNG demo |

### AJAR side (no further work pending)

- HCDS finding shipped to the team via HANDOFF.md + GitHub Release
- Speedup demo committed (`5ab518d`) and pushed to `main`, evidence comment posted on PR #10

---

## What changed about how we work (good lessons)

Things we figured out the hard way tonight that future-you should reuse:

1. **`tests/` is gitignored — force-add.** The mlxterp repo's `.gitignore`
   covers `test_*.py`. Use `git add -f tests/test_<thing>.py` so the
   contract tests actually land. Every PR's commit message acknowledges
   this; do the same to keep the maintainer aware.

2. **The adjoint trick is how you get gradients through every layer
   simultaneously.** Naive: replace each layer's output with a free
   variable and differentiate. That decouples the graph because the
   replacement at layer i+1 overwrites whatever depended on the variable
   at layer i, so `dM/d(var_i) = 0` for every i except the last. Right:
   *add* a zero tensor at each layer's output and differentiate w.r.t.
   the zero. The natural forward runs unchanged at zero, and the
   gradient flows via the chain rule. PR #13 uses this; reuse it for
   anything else where you want all-layer gradients in one backward.

3. **"Frozen-norm" is the formulation that makes sum-of-contributions
   identities hold.** When DLA-style decompositions need to sum to the
   actual logit_diff, treat the final-norm scaling as a constant
   captured at the natural forward, then absorb it into the
   logit-direction. PR #15 does this; the identity gap was -0.0006.
   Anywhere you decompose a residual-style activation into linear
   contributions, this is the trick.

4. **Multiple PRs > one big PR** when the maintainer is solo. We talked
   about this — single PR review fatigue is real. Eight focused PRs
   each with one validation table beat one mega-PR.

5. **Validation lives in `examples/<thing>.py` next to a contract test
   in `tests/test_<thing>.py`.** Pattern is now consistent across all 8
   PRs. Tests lock down API shape and one identity / sanity check that
   doesn't take long. Examples produce the validation table you'd put
   in the PR body. Future PRs should follow this pattern.

6. **mlxterp+8bit produces research-grade answers identical to torch+fp16
   on AJAR's task** — proven on PR #10's parity comment. You can drop
   the torch+MPS path entirely if you want; the answers don't change.

---

## What to do first when you come back

Before starting any new work, do this 5-minute check:

1. **Check PR status:**
   ```
   gh pr list --repo coairesearch/mlxterp --author "@me"
   ```
   - Are any of the 8 PRs merged? If yes, note which.
   - Are there review comments? `gh pr view <N> --comments`

2. **If PRs got merged, sync the fork:**
   ```
   cd third_party/mlxterp
   git fetch origin
   git checkout main && git pull origin main
   git push edward-lcl main   # keep your fork's main aligned
   ```

3. **If reviewer requested changes, the branches are still there.**
   Each PR's branch name is in the table above. Check it out, address
   the feedback, push, comment on the PR.

4. **For new Tier 2/3/4/5 work:** branch from the *latest accepted
   `origin/main`*, not from your old branches. Each PR is independent
   so this is straightforward; even if e.g. #11 and #13 both touch
   `analysis.py`, they were authored from `origin/main` so a rebase is
   only needed once #11 lands.

5. **Run the smoke tests to confirm the toolchain still works:**
   ```
   cd /Users/edward/Projects/AJAR
   PYTHONPATH=third_party/mlxterp python3 -m pytest \
       third_party/mlxterp/tests/test_patching_visualization.py \
       -x -q --no-cov
   ```
   That's the lightest-weight test (no model load). If it passes,
   environment is healthy.

---

## What to build next, in priority order

### Tier 2 remaining — 2 pieces

These are the obvious next PRs. Both are sizeable but self-contained.

**#8 SAE Feature Circuits** (the BauLab "sparse feature circuits"
method, attribution-style). Algorithm: for each SAE feature at a
captured site, compute its causal contribution to the metric (using
the attribution-patching adjoint trick from PR #13 but at the feature
level), threshold + prune to get a feature graph. This depends on
mlxterp's existing SAE training pipeline (already ships) — so it's
mostly orchestration on top of:

- existing SAE inference (encode activations into feature space)
- attribution-patching machinery from PR #13 (adapt to features
  instead of layer outputs)
- new: feature-graph construction + prune-by-effect

Estimate: bigger than Tier 1 PRs. Probably one full session.

**#9 Feature Dashboards** (max-activating examples, activation
histograms, logit-weight analysis, HTML dashboards). Mostly an
engineering / viz piece. Can lean heavily on what PR #17 sets up. The
"max-activating examples" loop is the only non-trivial bit — it needs
a dataset to scan. Estimate: half to full session.

If you can only do one, pick **#8** — it's the more research-grade
contribution and sets up Tier 3 ACDC.

### Tier 3 — frontier circuit discovery

In rough order of (impact / difficulty):

**#11 ACDC (Automated Circuit Discovery)** — Conmy et al. 2023
algorithm: greedy edge pruning until removing any edge breaks the
metric. Sits on top of PR #14's path patching. Mostly the loop
(propose edge to prune → test → keep/revert) plus circuit-graph
output. Estimate: one session. **Prerequisites: PR #14 merged or
local.**

**#12 DAS / Boundless DAS** — pyvene's signature contribution.
Learnable rotation of the activation subspace where the causal
intervention happens. Needs `mx.optim` and a training loop on top of
the intervention machinery. Estimate: one to two sessions. Higher
research value than ACDC; lower priority because it's a heavier lift
and pyvene already has it.

**#13 Cross-Layer Transcoders (CLTs)** — Anthropic's circuit-tracing
infrastructure. Open-source weights exist for Gemma-2-2b and
Llama-3.2-1b. The implementation is mostly "load the CLT weights,
expose the same hook-point API as SAEs, plus an attribution-graph
constructor." Estimate: two+ sessions. The biggest research-prestige
win on the list.

**#14 Auto-Interp** — LLM-generated feature descriptions. EleutherAI
and Anthropic both have this. With Claude Sonnet API + your existing
SAE features, this is a small wrapper. Estimate: half session. Low
priority unless you have a specific feature-naming need.

### Tier 4 — conversation-level interpretability (mlxterp's unique niche)

This is where mlxterp could genuinely lead the field — no other library
analyses multi-turn conversations. Roadmap calls for:

**#15 Conversation Trace** — `model.conversation_trace(messages)`
context manager with turn-aware activation slicing. Implementation
plan is already laid out in the roadmap (lines 322-415): chat-template
detection, turn-boundary scanning, `Turn` dataclass, `TurnList` with
role filtering. Estimate: one session for the core, second session for
cross-turn attention aggregation.

**#16 Conversation Patching** — turn-level patching: replace turn 1's
activations with a counterfactual, measure effect on turn 3. Same
machinery as activation patching (PR #11) but operating on turn
position-ranges instead of single positions. Estimate: half session
once #15 lands.

**#17 Conversation Attention Analysis** — cross-turn attention heatmaps,
turn-aggregated attention. Builds on the visualisation module from
PR #17 of mlxterp (this PR). Estimate: half session.

Tier 4 is the most strategic from a "differentiation" standpoint —
it's the gap in the field. If maintainer signals interest in any
particular Tier 4 piece, prioritise it over Tier 3.

### Tier 5 — agentic interpretability (the big bet)

Read the roadmap section starting at line 512 carefully. This is
where mlxterp becomes a *platform*, not just a library. Order:

1. **#20 Structured analysis output** — refactor analysis methods to
   return `AnalysisResult` objects with `.data`, `.summary`,
   `.to_json()`. Without this, every Tier-5 piece downstream is
   awkward. Cross-cuts everything we already shipped — could be a big
   refactor PR or split per-feature.

2. **#19 MCP server** — wraps mlxterp as Claude Code tools via
   `mcp.server.fastmcp`. The MCP SDK (`mcp` package) handles 90% of
   it. Just need to expose our analysis methods as `@mcp.tool()`
   functions. Estimate: one session.

3. **#21 Research workflow primitives** — pre-built workflows like
   `behavior_localization()` that chain multiple analyses. Won't
   matter until #20 is in place. Estimate: one session.

4. **#22 AutoInterp ratchet loop** — Karpathy's autoresearch pattern
   adapted for interpretability. Depends on all of #19/#20/#21.
   Estimate: two sessions; mostly orchestration. **This is the
   marquee feature. Save it for last.**

---

## AJAR-side Tier 2 (separate from mlxterp work)

These are post-run / wider-eval pieces that strengthen the HCDS
finding without touching mlxterp:

- **Multi-seed HCDS sensitivity** — re-run the same 50 questions with
  3-5 different seeds, check whether the headline (instruct + thinking
  show similar HCDS) holds. This is mostly running the existing
  pipeline with different `--seed` flags and re-aggregating. Pipeline
  already supports it; just need a small orchestrator.

- **Dependency-broken perturbations** — current perturbations swap
  numbers; "dependency-broken" means perturbations that violate the
  problem's structure (e.g. asking for a quantity that doesn't depend
  on the prompt). Drop into `scripts/build_perturbations.py`. Half
  session.

- **StrategyQA replication** — run the deep-table pipeline on
  StrategyQA instead of GSM8K, see if the HCDS pattern holds on a
  non-math reasoning benchmark. Bigger lift because StrategyQA needs
  different prompt formatting and a different correctness check.

These are independent from mlxterp upstream — useful momentum if
maintainer review on the 8 PRs is slow.

---

## Open follow-ups across all PRs (the unchecked boxes)

For each PR, these were left explicitly unchecked in the roadmap and
acknowledged in the PR body:

**#11 (activation patching)** — `attn_head` component support is
blocked on splitting `self_attn` into per-head outputs. This is the
**single highest-leverage upstream change** because it unblocks:
- per-head activation patching (#11)
- per-head attribution patching (#13)
- per-head DLA (#15)
- layer × head heatmap (#17)

If you want to land one tactical PR after the current 8 get reviewed,
**this is the one**: a small change to `mlxterp/core/proxy.py` (or
wherever attention is wrapped) to expose per-head outputs as separate
hook points. Probably 20-30 lines plus a test. Then the four
PR-follow-ups above each become small additions.

**#15 (DLA)** — bar-chart visualisation. Subsumed into PR #17 (which
ships `plot_dla`). Mark it done when PR #17 merges.

**#16 (residual stream)** — proxy-style access
(`model.layers[i].resid_pre.save()`) was deferred. Functionally
equivalent to the method-style API we shipped. Worth adding only if
the maintainer asks for it.

**#17 (visualisation)** — Plotly versions of the four renderers.
Same dict / matrix interface, just a different backend. Half-day
follow-up if anyone needs interactivity.

---

## Practical ergonomics

- Working dir: `/Users/edward/Projects/AJAR`
- Fork clone: `third_party/mlxterp/` (gitignored from AJAR)
- Activation: just use `PYTHONPATH=third_party/mlxterp python3 ...` —
  no venv needed for most work; system Python 3.9 has the deps
- Running tests: `cd /Users/edward/Projects/AJAR && PYTHONPATH=third_party/mlxterp python3 -m pytest third_party/mlxterp/tests/test_<thing>.py -x -q --no-cov`
- Running examples: same `PYTHONPATH` prefix; the heavy ones load
  Llama-3.2-1B-Instruct-4bit which is ~1GB and takes ~3s to warm

Model-load reminder: Llama-3.2-1B-Instruct-4bit is the standard test
model used across all 8 PRs. ~3s to warm. If something is slow, that's
why. Don't switch to a bigger model unless you actually need to —
8 PRs of validation evidence are on the 1B model.

---

## TL;DR for future-you

1. Open this doc.
2. Run `gh pr list --repo coairesearch/mlxterp --author "@me"` and
   note merged/changes-requested PRs.
3. If any PRs need rework, address feedback on existing branches and
   push.
4. If clean, pick the next item from the priority list above.
   Default: per-head split for `self_attn` (highest-leverage, smallest
   PR, unblocks four follow-ups across already-shipped PRs). After
   that: SAE feature circuits (Tier 2 #8), then ACDC (Tier 3 #11),
   then conversation trace (Tier 4 #15).
5. Use the `feature/<name>` → branch from `origin/main` → contract
   test in `tests/` (force-added) → validation script in
   `examples/` → markdown PR body → `gh pr create` pattern.
6. Each PR should land with at least one identity / Spearman /
   numerics-vs-baseline table in the body.
