# Outstanding work

Tracked here so nothing slips between launches.

## Definitely planned

- [ ] **Task 4 — StrategyQA replication.** Rerun the same 6-condition matrix
      (2 models x 3 prompts) on StrategyQA. The runner only assumes a
      `{question, answer}` JSONL shape, so this is mostly a fixture loader
      change plus a yes/no answer parser instead of the GSM8K numeric one.
- [ ] **Task 5 — paired diagnostic benchmark.** Same shape, depends on the
      benchmark file landing.
- [ ] **Second perturbation operator.** v1 ships distractor-irrelevant only.
      Dependency-broken variants (preserve surface form, break the logical
      chain) are the next operator and are mentioned explicitly in the
      proposal. Easiest path is a model-generated fixture using the same
      JSONL contract `build_perturbations.py` already produces.

## Statistical follow-ups (Tasks 7-8)

- [ ] **HCDS implementation.** Compute the per-(model, dataset) feature
      vector across the four signals available (latency, entropy,
      consistency, perturbation, mechanistic-drop) and report
      `D(Neutral, NoCoT) - D(Neutral, CoT)`. Decide on standardization
      method up front (z-score per feature across the 3 conditions before
      distance) and document it.
- [ ] **Bootstrap CIs / paired t-tests over per-question HCDS.**
- [ ] **Anchor-vs-control hypothesis test for Task 10.** Currently the
      aggregator emits the difference per condition; a paired test (per
      question, anchor_drop vs control_drop) gives a real p-value.

## Documentation polish

- [ ] In the methods section, document the parsing-difficulty caveat for
      `neutral_strict` (no `\boxed{}` directive => harder parsing => some
      apparent accuracy dip is parser fragility, not reasoning regression).
      The improved parser handles `Final answer:` / `Answer:` / `<think>`
      stripping but is not bulletproof.
