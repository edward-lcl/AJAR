# Team Handoff — 2026-05-05

This is the "start here" doc for picking up the AJAR experiments after the
2026-05-04 / 2026-05-05 work. Everything described below is committed to
`main`; you don't need to merge anything to see it.

## TL;DR

- **Task 6 deep table is done.** 300 rows × every column, on a 50-question
  GSM8K slice across both Qwen3-4B variants and all three prompts. Live at
  `results/runs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/task6_table.csv`.
- **Headline finding: HCDS is positive for both models.** Neutral_strict
  prompting produces behavioural and mechanistic profiles that look more
  like explicit_cot than like explicit_no_cot. `+1.24` (Instruct), `+1.03`
  (Thinking). Descriptive only at n=50; needs bootstrap CIs before it's a
  claim.
- **The 1.5-week runtime was a real bug.** `probe_attention_vectors` was
  doing one model forward pass per probe position (~1500 forwards per long
  Thinking baseline). The fix is a single full-sequence forward + indexing.
  Bit-validated. Same workload now finishes in ~7h on Mac, ~1-2h on a
  single A100.
- **Pipeline is robust now.** 1050/1050 items completed with 0 failures on
  the most recent run. Resume + caffeinate + per-model-invocation split
  + answer-probe cap means it survives interruption and OOM scenarios.

## What's ready for you to use

| Artifact | Path | Notes |
|---|---|---|
| Task 6 deep table | `results/runs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/task6_table.csv` | 300 rows. All five Task 6 columns populated. |
| Task 10 anchor sensitivity | `.../task10_anchor_sensitivity.csv` | 6 condition rows. Anchor−control accuracy drop. |
| Per-condition summary + analysis | `.../SUMMARY.md` | Headline + four findings to investigate. |
| Wide neutral_strict baseline | `outputs/2026-05-04_1856_*` (locally only — gitignored) | 3000 generations on 500 questions. Replaces the earlier `neutral`-prompt baseline for HCDS purposes. |

To reproduce or extend any of this:

```bash
git pull
pip install -r requirements.txt
pytest                          # 17 tests pass in ~50ms
python3 scripts/analyze_deep_table.py \
    --task6-csv results/runs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/task6_table.csv \
    --task10-csv results/runs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/task10_anchor_sensitivity.csv
```

## Recommended reading order

1. **`README.md`** — setup recipe + the deep-table workflow + env-var reference.
2. **`HANDOFF.md`** (this file) — what's done, what's next.
3. **`results/RUN_INDEX.md`** — every completed run with date/scope/outcome.
4. **`results/runs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/SUMMARY.md`** — the headline findings + four anomalies.
5. **`docs/run_postmortem.md`** — where the week-long runtime came from and 9 validity caveats on the current results.
6. **`TODO.md`** — backlog organised by priority. Validity follow-ups at the top.

## What you can start on right now (Task 6 + 7 + 8)

Listed by what unblocks the most paper-relevant work first.

### 1. Bootstrap CIs and paired tests on HCDS (Task 8) — ~3h

The descriptive HCDS numbers in `analyze_deep_table.py` are condition-mean
distances. To make them a claim, we need:

- Per-question feature vectors (z-score features per question, not per
  condition mean), then per-question HCDS = `D(neutral, no_cot) -
  D(neutral, cot)`.
- Bootstrap 1000 resamples of the question pool; report mean and 95% CI of
  HCDS per (model, dataset).
- Paired t-test of per-question HCDS vs zero per (model, dataset).

Suggested deliverable: `scripts/compute_hcds.py` that takes
`task6_table.csv` and writes `hcds_per_question.csv` plus
`hcds_summary.csv` with mean / 95% CI / p-value.

### 2. Anchor signal validity check on Thinking + explicit_cot — ~2h

`task10_anchor_sensitivity.csv` shows `anchor_drop − control_drop = -0.275`
for Thinking + explicit_cot. The proposal predicts this should be
positive. Three sub-experiments worth running:

- Decompose anchor score (`z(future_attn) + z(answer_attn) +
  z(activation_delta)`) and re-rank using each component alone. See which
  sub-feature actually predicts interventionable steps.
- Re-run the intervention phase with `AJAR_INTERVENTION_MAX_NEW_TOKENS=768`
  to rule out budget truncation.
- Spot-check 5 negative cases by hand; look at the actual
  intervention_record JSONs.

This is gated on whether the anchor algorithm itself is sound. If the
answer is "no, the algorithm misses real anchors," that's a method-level
finding worth reporting.

### 3. Compare HCDS with and without each feature — ~1h

The HCDS feature vector treats six features as equal-weighted after
z-scoring. Two of them are suspect on this dataset:

- Paraphrase consistency tracks accuracy too closely — likely redundant.
- Mechanistic ΔA on Thinking is the wrong sign — likely noisy.

Quick robustness check: report HCDS computed (a) with all features, (b)
without paraphrase consistency, (c) without mech ΔA, (d) with neither.
If the conclusion holds across all four, the result is robust.

### 4. Length-matched analysis (Task 11) — ~3h

Output_tokens varies 16-912 across conditions. Without length-matching,
we can't distinguish "neutral_strict reasons like CoT" from
"neutral_strict is verbose like CoT." Stratify questions into length
bins, recompute HCDS within bins.

## What's slower-burning

These are real but not blocking the paper-prep work above:

- **Task 4 — StrategyQA replication.** Same orchestrator, swap GSM8K
  loader for StrategyQA, swap numeric answer parser for yes/no. Time
  budget: similar ~7h on Mac, ~2h on A100.
- **Task 5 — paired diagnostic benchmark.** Pending the benchmark file.
- **Second perturbation operator** (dependency-broken variants). v1
  ships distractor-irrelevant only.
- **mlxterp evaluation** (the MLX-native MI library Grok suggested) —
  parked for v2. Could give us 3x faster MI on Apple Silicon and free
  SAE infrastructure if numerical equivalence checks out.

## Things to be honest about with reviewers

Pulled from `docs/run_postmortem.md` validity section. Land these in the
methods/limitations section before submission.

- **n=50 per condition is small.** Bootstrap CIs and paired tests are
  non-negotiable before significance claims.
- **GSM8K is plausibly in Qwen3 training data.** Both variants hit
  ≥96% on Instruct cot/neutral conditions. Either capability or
  memorisation; HCDS measures different things in those two worlds.
  Mitigation: HCDS-on-paraphrases (paraphrases are unseen even if
  originals were in training).
- **The Thinking model ignores the explicit_no_cot directive** and emits
  ~600 tokens of reasoning anyway. So Thinking + explicit_no_cot is not
  a "without CoT" condition. Worth reporting as a finding about the
  Thinking variant's training, not as a CoT/no-CoT contrast.
- **Single seed, single trial.** No within-condition variance estimate.
  3-seed control-step rerun would address this for Task 10 cheaply.
- **64-probe answer cap may bias anchor selection.** Mitigation: the
  validation script (`scripts/validate_probe_optimization.py`) confirms
  the optimisation itself doesn't change anchor ranks; needs separate
  check that capping at 64 vs probing all answer tokens doesn't shift
  ranks.

## Engineering state

The repo is clean and self-contained:

- 17 tests pass in <100ms (`pytest`)
- Pinned dependencies in `requirements.txt` and `pyproject.toml`
- One orchestrator script, idempotent, resume-safe (`scripts/run_deep_table.sh`)
- Both backends (oMLX for behavioural, torch for MI) work on Mac MPS
  with the same env-var interface
- Cloud-GPU recipe documented in `docs/cloud_gpu_setup.md` for when MPS
  is too slow
- All optimisations bit-validated against the original code path

If you want to reproduce the deep-table run from scratch on a different
machine (cloud GPU, another Mac, the lab cluster):

```bash
./scripts/run_deep_table.sh             # 50 samples, default config
./scripts/run_deep_table.sh 100         # override sample count
RUN_STAMP=2026-05-04_2232 \
    ./scripts/run_deep_table.sh         # resume into a prior run dir
```

Each step is idempotent via runner-level resume; killed runs pick up
where they left off on relaunch.

## Contact / questions

The full run history with explanations of every config decision is in
the commit log:

```bash
git log --oneline | head -15
```

Post-2026-05-05 commits document the OOM diagnosis, swap-thrash fix,
and resume-overwrite bug fix in their messages. Read those if anything
looks unexpected in the runner.
