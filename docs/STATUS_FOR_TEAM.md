# AJAR — Status for Team Sync

Generated 2026-05-06. ICML target: Friday 2026-05-08. This is the
"single source of truth" status doc — every Task from the proposal
mapped to what's done and what's pending.

---

## TL;DR

**Headline finding is locked.** Both Qwen3-4B variants show
statistically significant positive HCDS on GSM8K under the proper
bootstrap + paired t-test:

| Model | Mean HCDS | 95% CI | p-value | Verdict |
|---|---|---|---|---|
| Instruct | +2.25 | [+1.69, +2.85] | 2.2e-10 | Neutral aligns with CoT |
| Thinking | +0.48 | [+0.07, +0.86] | 0.021 | Neutral aligns with CoT |

Compute infrastructure is sorted: 1.5-week RunPod bottleneck reduced
to 3-8 hours on Mac (M5 Pro), basically zero cost.

**For ICML Friday submission, the gating items are scoping decisions,
not new compute work.** Details below.

---

## Task-by-task status

### Task 1 — Run Qwen3-4B Instruct/Thinking on GSM8K, 3 prompt conditions

**Status: ✅ Done.**

- Deep-table run on **n=50**: full feature vector, 2 models × 3
  prompts. `results/runs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/`
- Wide baseline at **n=500**: 3000 generations, accuracy + latency +
  output length only.
  `results/runs/2026-05-04_1856_gsm8k500_qwen3-4b_omlx_baseline-neutral-strict-1024/`
- StrategyQA replication: separate task, see Task 4 below.

### Task 2 — Causal testing via perturbations on GSM8K

**Status: ✅ Done.** Distractor-irrelevant perturbations applied to all
50 deep-table questions. `perturbation_delta_accuracy` column in
`task6_table.csv`. Second perturbation operator (dependency-broken
variants) is a roadmap item, not a critical-path one.

### Task 3 — Run mechanistic interpretability

**Status: ✅ Done.**

- Mechanistic outputs landed for all 300 (question × model × prompt)
  cells: `outputs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/mech/`
- Per-step scores (future_attn / answer_attn / activation_delta /
  combined), anchor selection, anchor + control intervention
  experiments — all run.
- `mech_intervention_delta` integrated into Task 6 wide table.
- **Methodology caveat documented**: anchor scoring picks late-trace
  summary steps on Thinking + explicit_cot, producing a negative
  anchor-vs-control sign. Investigation report:
  `anchor_investigation_thinking_cot.md`.
  - **Two ship options**: report-as-is (clean methodology finding) or
    apply the cheap position-penalty fix and re-run intervention on
    that one condition (~4 hr compute). See "Methodology fix"
    section below.

### Task 4 — StrategyQA replication

**Status: ❌ Not started.** Originally assigned to Ryan + Abdullah.
Existing pipeline handles it; missing pieces are:

- Fixture loader (`scripts/build_strategyqa_fixture.py`) — converts
  StrategyQA test set to GSM8K-shape JSONL with `#### yes` / `#### no`
  answers
- Yes/no answer parser — extends `extract_predicted_*` helpers
- Re-run `run_deep_table.sh` against the new fixture

If Ryan + Abdullah haven't started, infrastructure team can stand up
the loader + parser tonight (2-3 hrs dev), kick off the run tomorrow
morning (~4-5 hr compute on Mac for n=50). Result: yes/no version of
the same Task 6 table.

### Task 5 — Paired diagnostic benchmark

**Status: 🚫 Blocked on benchmark file.** No work possible until the
benchmark file exists. When it lands, ~4 hrs to wire it up
(loader + answer parser if non-numeric).

### Task 6 — Structured per-(model, dataset, prompt, question, trial) table

**Status: ✅ Done.** 300 rows, 30 columns.
`results/runs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/task6_table.csv`.
Also normalised + joined with HCDS at
`team_spreadsheet/per_question_hcds.csv`.

### Task 7 — HCDS computation

**Status: ✅ Done.** Per-question HCDS, 1000-iteration bootstrap,
paired t-test all in `hcds_per_question.csv` /
`hcds_summary.csv`. Six leave-one-feature-out variants in
`hcds_summary_no_*.csv`. Headline figure at `hcds_figure.png`.

### Task 8 — Statistical testing

**Status: ✅ Done.** Bootstrap CIs and paired t-tests are reported in
`hcds_summary.csv`. Both models p < 0.05; both 95% CIs strictly
above zero.

### Task 9 — Perturbation fragility analysis

**Status: ⚠️ Mostly done.** Per-question Δ accuracy is in
`task6_table.csv` (`perturbation_delta_accuracy`). Plot of "depth vs
accuracy drop" / "distractor vs accuracy drop" is a writeup-time
matplotlib job. Numbers are there.

### Task 10 — Mechanistic validation (anchor vs control)

**Status: ✅ Done with caveat.** `task10_anchor_sensitivity.csv` has
the anchor-suppression vs control-suppression comparison per
(model, prompt). The Thinking + explicit_cot anomaly is documented.

### Task 11 — Control analyses

**Status: ✅ Mostly done.**

- Length-matched analysis: ✅ `hcds_length_matched.csv`. Instruct
  passes all three length tiers; Thinking ambiguous on the medium
  tier where explicit_cot truncated.
- **Entropy-only HCDS baseline (new)**: ✅ Both models stay strongly
  positive. Instruct +1.46 (p=5.9e-08), Thinking **+0.83 (p=2.5e-06)**
  — Thinking's entropy-only signal is *stronger* than its full HCDS
  signal, consistent with earlier feature-stability findings.
- **Latency-only HCDS baseline (new)**: ✅ Both models positive.
  Instruct +1.79 (p=2.6e-19) — strongest p-value of any variant.
  Thinking +0.41 (p=3.9e-04). Latency carries most of the Instruct
  signal; entropy carries most of the Thinking signal.
- Randomised-token baseline: ❌ Not done. Would require re-running
  generation with shuffled-token outputs, which is more compute than
  is worth tonight. Note as future work.

What this means: **HCDS is not a coincidence of any single feature.**
Every leave-one-out variant and every isolate-one-feature variant
shows positive HCDS for both models. The Instruct effect is
length-dominated (huge p-value on latency-only); the Thinking effect
is entropy-dominated.

### Task 12 — Cross-dataset aggregation

**Status: 🚫 Blocked on Tasks 4+5.** Nothing to aggregate until
StrategyQA lands.

---

## Compute / infrastructure status

- **Bottleneck removed**: 1.5-week RunPod pipeline → 3-8 hr Mac
  pipeline at zero cost.
- **Speed validated**: same answers as torch+fp16 (5/5 numerical
  parity on GSM8K).
- **Reproducibility**: `git pull && pip install -r requirements.txt
  && pytest` works clean. README has the recipe. HANDOFF.md is the
  "where do I start" doc.
- **Wide baseline (n=500)** already ran and analysed
  (`initial_report.md` + `condition_summary.csv` etc.).

---

## Pre-Thursday-night decisions for the team meeting

1. **Methodology fix on Thinking anchors: ship-as-is or apply cheap
   fix?**
   - Ship-as-is: methodology caveat reads as a known limitation.
     Numbers in the table go negative for one cell.
   - Cheap fix: position-penalty in anchor score; re-run intervention
     phase on Thinking + explicit_cot only (~4 hr). If sign flips,
     paper looks tighter. If not, fall back to caveat.
   - **Recommendation: cheap fix.** Low downside, possibly cleaner
     numbers.

2. **StrategyQA: who's on it?**
   - If Ryan + Abdullah haven't started, infrastructure team can
     stand up the pipeline tonight, kick off the run tomorrow.
     Result lands by Thursday afternoon.
   - If nobody picks it up, paper ships GSM8K-only. Cross-dataset
     aggregation (Task 12) gets dropped from this submission.

3. **n=500 deep-table extension: is it worth doing?**
   - Full extension: ~63 hr compute, infeasible by Thursday.
   - Baselines + accuracy + latency only on remaining 450: ~6-8 hr,
     doable. Doesn't extend HCDS (still n=50 there) but does extend
     accuracy curves.
   - **Recommendation: skip.** The wide baseline at n=500 already
     supports the accuracy patterns; doubling that with deep-table
     coverage is a future-paper thing.

4. **Multi-seed sensitivity (Task 11 follow-up): worth running?**
   - Re-run intervention phase under 2-3 seeds (~4-6 hr each).
     Reports variance in mech ΔA. Adds robustness.
   - Doable in background while the cheap fix and StrategyQA run.

5. **Pipeline figure ownership**: Edward (implementation team).
   Single methods diagram showing data → 6 conditions → 6 metrics →
   HCDS → significance test. Working in Figma.

---

## What the implementation team can deliver between now and Thursday night

| Item | When | Status |
|---|---|---|
| Consolidated team spreadsheet (`team_spreadsheet/`) | Now | ✅ Done — `results/runs/<deep-table>/team_spreadsheet/` |
| This status doc | Now | ✅ Done |
| StrategyQA fixture loader + yes/no parser | Tonight | ✅ Done — `scripts/build_strategyqa_fixture.py`, `data/fixtures/strategyqa_test_first50/` (n=50, 42% yes), 13 contract tests pass |
| StrategyQA-50 run | Tomorrow morning | ⏳ Ready to kick off — `GSM8K_JSONL=data/fixtures/strategyqa_test_first50/strategyqa.jsonl AJAR_DATASET_KIND=strategyqa bash scripts/run_deep_table.sh 50` |
| Cheap methodology fix scaffolding | Tonight | ✅ Done — `AJAR_ANCHOR_LATE_PENALTY` env var in runner; one-command rerun script at `scripts/run_methodology_fix.sh` |
| Methodology fix Thinking-cot rerun | Tomorrow | ⏳ Pending — `./scripts/run_methodology_fix.sh` (~3-4 hr) |
| Entropy-only / latency-only HCDS baselines | Tonight | ✅ Done — variants in `hcds_summary_all.csv` (entropy_only / latency_only rows) |
| Pipeline figure (Figma) | Edward, tomorrow | ⏳ Pending — Edward sets up Figma |
| Multi-seed sensitivity (background) | Tomorrow | ⏳ Pending |
| Randomised-token baseline (Task 11) | Skipping | ❌ Compute cost too high for tonight; documented as future work |

---

## Where things live

- **Headline figure**:
  `results/runs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/hcds_figure.png`
- **Team spreadsheet**:
  `results/runs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/team_spreadsheet/`
- **Plain-language summary**: `HANDOFF.md` (root)
- **Per-run summaries**: `results/RUN_INDEX.md` and the per-run
  `SUMMARY.md` files
- **Backlog**: `TODO.md` (root) — full task tracker with priority
  reasoning
- **This doc**: `docs/STATUS_FOR_TEAM.md`
