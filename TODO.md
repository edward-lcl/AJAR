# Outstanding work

Tracked here so nothing slips between launches. Reorganised after the
2026-05-04_2232 deep-table run. Each section starts with what's blocking
forward progress.

## Validity follow-ups before claiming significance

These are the open scientific questions on the 2026-05-04_2232 deep-table
run, in priority order. See `docs/run_postmortem.md` for full reasoning
on each.

- [ ] **Bootstrap CIs + paired tests on HCDS (Task 8).** n=50 is too small
      for descriptive HCDS to be a claim. Per-question HCDS, bootstrap
      1000x, paired t-test against zero per (model, dataset).
- [ ] **Validate anchor selection under the 64-probe cap.** Re-run anchor
      scoring on 5-10 samples with `AJAR_MAX_ANSWER_PROBES=99999` (no
      cap), compare top-3 anchor ranks against the capped run, report
      the divergence rate.
- [ ] **Investigate the negative anchor-control delta on Thinking +
      explicit_cot.** Decompose anchor score into its three sub-features
      (future_attn / answer_attn / activation_delta) and re-rank. Re-run
      with `AJAR_INTERVENTION_MAX_NEW_TOKENS=768` to rule out budget
      truncation. Spot-check 5 negative cases by hand.
- [ ] **Multi-seed sensitivity check.** Re-run only the intervention
      phase under seeds 17, 18, 19 (3 runs × ~30 min each at new speed),
      report variance in mech ΔA per condition.
- [ ] **Length-matched analysis (Task 11).** Output token count varies
      from 16 to 912 across conditions. Length-stratified comparison
      to distinguish "neutral_strict reasons like CoT" from "verbose
      like CoT".
- [ ] **HCDS feature-stability check.** Compute HCDS with and without
      paraphrase consistency in the feature vector; if the conclusion
      flips, report both. Likewise without mech ΔA (which we have
      reasons to distrust on Thinking).
- [ ] **Document GSM8K contamination concern.** Both Qwen3 variants
      hit ≥96% on Instruct cot/neutral conditions. Either genuine
      capability or memorisation; HCDS measures something different in
      each case. Mitigation: HCDS-on-paraphrases as a robustness test,
      since paraphrases are unseen even if originals were trained on.
- [ ] **Filter-bias check on paraphrases.** 22% of paraphrases dropped
      by `numbers_preserved=true` filter. Verify drop rate is
      uncorrelated with question difficulty; if not, report unfiltered
      consistency too.

## Immediate next slice (post deep-table)

- [ ] **Analyze the wide neutral_strict baseline.** The 3000-generation
      `2026-05-04_1856_gsm8k500_qwen3-4b_omlx_baseline-neutral-strict-1024`
      run completed but has no `analysis/<run_id>/initial_report.md` yet.
      Run `scripts/analyze_baseline_run.py` against it.
- [ ] **HCDS computation with bootstrap CIs (Task 7 + 8).** The
      `analyze_deep_table.py` script reports a descriptive HCDS at the
      condition-mean level. We still need:
      - per-question HCDS (z-score features, then distance per question)
      - bootstrap CIs over 1000 resamples
      - paired t-test comparing per-question
        `D(neutral, no_cot) − D(neutral, cot)` against zero
      Suggested deliverable: a small `scripts/compute_hcds.py` that takes
      `task6_table.csv` and writes `hcds_per_question.csv` plus a
      summary CSV with mean / 95% CI / p-value per model.
- [ ] **Investigate the negative anchor-control delta on Thinking + explicit_cot.**
      `task10_anchor_sensitivity.csv` shows control_drop > anchor_drop by
      0.275 — opposite of the proposal's prediction. Three sub-experiments:
      decompose the anchor score (`future_attn`, `answer_attn`,
      `activation_delta`) and see which sub-feature actually predicts
      interventionable steps; raise intervention budget from 384 to 768
      tokens to rule out length-cap interactions; spot-check 5 negative
      cases by hand to see if the worker is intervening on truly
      reasoning-irrelevant tokens.

## Definitely planned (multi-day)

- [ ] **Task 4 — StrategyQA replication.** Rerun the same 6-condition matrix
      on StrategyQA. The runner only assumes a `{question, answer}` JSONL
      shape, so this is mostly:
      - a fixture loader: `scripts/build_strategyqa_fixture.py` that fetches
        StrategyQA test, normalizes to GSM8K-shape JSONL with `answer`
        ending in `#### yes` or `#### no`
      - a yes/no answer parser: extend `extract_predicted_number` (or
        sibling `extract_predicted_yesno`) to handle boolean answers
      - rerun `run_deep_table.sh` with `GSM8K_JSONL` pointed at the new
        fixture
- [ ] **Task 5 — paired diagnostic benchmark.** Same shape, depends on the
      benchmark file landing.
- [ ] **Second perturbation operator: dependency-broken variants.** v1
      ships distractor-irrelevant only. The proposal's "dependency-broken"
      variant preserves surface form but breaks the logical chain. Easiest
      path is a model-generated fixture using the same JSONL contract that
      `build_perturbations.py` already produces. New script
      `build_dependency_broken.py` that calls oMLX with a prompt asking
      to alter one numeric value in a way that breaks the math. Gold
      answer needs recomputation, which is the harder part.

## Engineering improvements observed during the run

- [ ] **Resume should not blow away aggregate JSONLs.** Already fixed in
      `6b45840` (rebuild from per-sample files at end-of-run). Worth a
      second look: should we also write a small file at run start that
      explicitly says "this run wrote N samples this invocation, M
      samples were skipped via resume" so future debugging is easier?
- [ ] **Intervention budget should be configurable per-model.** Thinking
      model's intervention continuations occasionally truncate at 384
      tokens before reaching a final answer. A per-model override
      (`AJAR_INTERVENTION_MAX_NEW_TOKENS_THINKING=768`) would be cleaner
      than the current global cap.
- [ ] **Per-model invocation split is robust but inelegant.** The
      orchestrator currently invokes the runner twice. A nicer fix is at
      the worker level: process all queued intervention tasks for the
      currently-loaded model before pulling a baseline that would force a
      swap. That requires a priority queue rather than mp.Queue's FIFO.
      Not urgent; the orchestrator-level split works.
- [ ] **`probe_attention.npz` archival flag is currently unused.** With
      the answer-probe cap of 64, full archival of probe attention would
      now fit on disk for a 50-sample run (~1 GB). If we want to revisit
      attention patterns post-hoc without re-running, flip
      `AJAR_SAVE_FULL_PROBE_ATTENTION=1` for the next big run.

## Statistical follow-ups (Tasks 7-8 detail)

- [ ] **HCDS feature standardization choice.** Per-condition z-score is
      what `analyze_deep_table.py` does. Alternative is to standardize
      across all rows of a model. Document which we use and why in the
      methods section.
- [ ] **Anchor-vs-control hypothesis test for Task 10.** Currently the
      aggregator emits the difference per condition; a paired test (per
      question, anchor_drop vs control_drop) gives a real p-value.
- [ ] **Length-matched analysis (Task 11).** Drop output_tokens correlates
      strongly with accuracy. Need a length-matched comparison so we can
      distinguish "neutral_strict generates as much as explicit_cot
      because it is reasoning" from "neutral_strict generates as much as
      explicit_cot because that's what verbose models do".

## Documentation polish

- [ ] **Methods section on parsing fragility.** The improved parser
      handles `Final answer:` / `Answer:` / `<think>` stripping but
      neutral_strict accuracy is still slightly depressed compared to
      neutral (which has the `\boxed{}` directive) because of harder
      parsing — not solely because of reasoning quality.
- [ ] **Document the cap-at-64 answer-probe choice.** Note in the methods
      that anchor selection used a 64-position even sample of the answer
      span (full answer probes proved memory-prohibitive on Thinking
      outputs). Show that anchor ranks are stable under this sampling.

## Parked

- [ ] **mlxterp evaluation as alternative MI backend.** See
      `docs/cloud_gpu_setup.md` for the cloud option; mlxterp is the
      Apple-native MLX path. Worth a 2-day prototype before committing
      to migration. Would unlock SAE infrastructure, logit_lens, and
      ~3x faster generation if numerical equivalence checks out.
