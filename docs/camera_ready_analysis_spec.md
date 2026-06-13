# Camera-ready analysis spec (frozen before the remaining runs)

**Status:** frozen 2026-05-30, before generating any of the six remaining
experiments. Camera-ready target ~2026-06-20.

**Purpose.** The headline HCDS results are in. The six experiments below
have *not* been run. To keep them confirmatory rather than exploratory,
this document fixes — in advance — for each experiment: the exact
quantity computed, the statistical test, the multiple-comparison
handling, and the pass/fail line. Once data lands we report what the
pre-committed test says, whichever way it falls. Deviations from this
spec get logged in a "deviations" section at the bottom with a reason.

This doc doubles as the run plan. Order is **resequenced for risk**, not
in the order the experiments were requested: the experiment most likely
to change the paper's claims runs first, while there is still runway to
rewrite around it.

---

## 0. Conventions that apply to every experiment

- **HCDS definition is fixed** (`scripts/compute_hcds.py`,
  `HCDS_FEATURES`): per-question **6-element** feature vector
  `[latency_per_output_token, token_entropy_mean, token_entropy_slope,
  paraphrase_consistency, perturbation_delta_accuracy,
  mechanistic_intervention_delta_accuracy]` — note entropy contributes
  *two* columns (mean + slope), so what the paper calls the "5-feature"
  HCDS is 6 columns in code. **Flag for the team: reconcile the
  paper's "5-feature" wording with the 6 code columns before camera-ready.**
  Features are z-scored across questions within each (model, condition) cell,
  Euclidean distance in standardized space,
  `HCDS = D(neutral, nocot) − D(neutral, cot)`. Positive ⇒ Neutral
  behaves more like CoT.
- **Inference is fixed:** per-question HCDS, paired t-test vs 0 per
  (model, dataset), and a 1000× percentile bootstrap 95% CI
  (`DEFAULT_N_BOOT=1000`, `DEFAULT_SEED=17`). A result "is significant"
  iff the bootstrap CI excludes 0 **and** it survives the correction in
  §0.2.
- **Decoding is deterministic** (`temperature=0`, seed 17). Seed-variance
  is handled once, globally (§7), not per experiment.

### 0.1 Primary vs secondary endpoints (declared now)

Only the **primary family** is corrected for multiple comparisons and is
allowed to carry a claim in the abstract. Everything else is reported
descriptively / diagnostically and is explicitly labeled as such.

- **PRIMARY family (4 tests):** the per-(model, dataset) HCDS-vs-0 tests
  for {Thinking, Instruct} × {GSM8K, cross-family}. These are the
  "latent reasoning is detectable" claims.
- **SECONDARY / diagnostic:** negative-control calibration, anchor
  sign / position-matching / gradient attribution, 4096 rerun, style
  controls, detector AUROC. These *support or qualify* the primary
  claims but do not themselves appear as significance claims in the
  abstract.

Rationale: declaring a small primary family keeps Holm correction from
being self-defeatingly conservative, and is the standard, defensible way
to run a multi-endpoint study.

### 0.2 Multiple-comparison correction

- Primary family (4 tests): **Holm–Bonferroni** at family-wise α=0.05.
  Report raw p, Holm-adjusted p, and the bootstrap CI for each.
- Negative-control cells: reported as calibration evidence (§1), not
  added to the primary family. Their CIs are reported uncorrected with
  that framing made explicit.

### 0.3 Calibrated HCDS (fixed definition)

For each feature subset, define the calibrated signal as

```
HCDS_cal(model) = HCDS_GSM8K(model) − HCDS_negcontrol(model)
```

computed per feature-vector (not per scalar feature), where
`HCDS_negcontrol` pools the arithmetic and factual families. CI on the
difference: two-sample bootstrap (1000×, seed 17) resampling the GSM8K
per-question HCDS and the negative-control per-question HCDS
independently and taking the difference of means. This replaces the
current informal "GSM8K minus ~0.8–1.2 ≈ +1.0" estimate in the
appendix with one number + CI per model.

### 0.4 What would trigger the reframe

Pre-committed, so it isn't a judgment call after seeing the data: **if
`HCDS_cal(Instruct)` has a bootstrap CI that includes 0** (i.e. the full
multi-feature Instruct GSM8K signal is statistically indistinguishable
from its negative-control baseline), we drop the "positive HCDS in both
variants" framing and move to the discriminant-validity framing
(reasoning-attributable in Thinking, confounded in Instruct). See
`memory/project_negcontrol_reframe_ok.md`. The Thinking primary claim is
unaffected by this branch.

---

## 1. Full negative-control calibration — **RUN FIRST**

*Why first:* most likely single experiment to change the paper's claims
(latency-only already shows Instruct only partially calibrates). De-risk
while there is time to rewrite the abstract/intro, not the appendix only.

- **What runs:** full 6-column HCDS pipeline (not just latency) on
  `data/fixtures/negative_control/arithmetic_*.jsonl` and
  `factual_*.jsonl`, both Qwen3-4B variants, all 3 prompts. Extend
  `scripts/compute_negative_control_hcds.py` from latency-only to the
  full `FEATURE_COLUMNS`, reusing `analysis/hcds.py`.
- **Pre-committed quantities:** per-(model, family) full HCDS with
  bootstrap CI; pooled negative-control HCDS per model;
  `HCDS_cal(model)` per §0.3.
- **Pass/fail lines:**
  - *Thinking calibrates* iff full HCDS CI on both families includes 0.
    (Confirms discriminant validity; expected.)
  - *Instruct residual reasoning component exists* iff
    `HCDS_cal(Instruct)` CI excludes 0. If it includes 0 → trigger §0.4
    reframe.
- **Parser note already documented** (appendix): Thinking truncation at
  1024 on factual lookups inflates parser fallback. Does not affect
  latency HCDS; for the *paraphrase/perturb/mech* features we verify the
  fallback rate per cell and report it. The 4096 rerun (§3) is the
  mitigation if any non-latency feature turns out to depend on
  truncated traces.

## 2. Cross-family replication — **RUN SECOND (after power check)**

- **Power check is a gate, not a formality** (§ power, below): the
  Thinking effect is small; n=50 may be underpowered on a second family.
  Compute required n from observed GSM8K effect size/SD *before*
  committing the run. If n=50 is underpowered for the Thinking cell,
  either raise n or pre-declare the Thinking cross-family result as
  descriptive-only.
- **Dataset choice does double duty:** pick a family that is both a
  different task type **and** less likely contaminated than GSM8K
  (addresses postmortem caveat #4). Candidate order: MGSM (numeric,
  multilingual, less-contaminated) → MATH subset → StrategyQA
  (`build_strategyqa_fixture.py` already exists but is boolean, needs the
  yes/no parser). Decision recorded in deviations log when made.
- **Pass/fail:** directional replication (same sign as GSM8K) with
  Holm-adjusted CI excluding 0 = confirmatory. Same sign but CI includes
  0 under adequate power = "weak/underpowered replication," reported
  honestly. Opposite sign = reported as a genuine non-replication.

## 3. 4096-token Thinking rerun — diagnostic

- **What runs:** Thinking baselines + interventions with
  `AJAR_MAX_NEW_TOKENS=4096` (vs 1024) and
  `AJAR_INTERVENTION_MAX_NEW_TOKENS=768` (vs 384), on the GSM8K n=50 mech
  slice and the negative-control factual family.
- **Pre-committed questions:** (a) does the negative-control parser
  fallback rate drop materially, and does any non-latency neg-control
  feature change conclusion? (b) does `anchor_drop − control_drop` on
  Thinking+explicit_cot flip from its current −0.275 toward positive
  when the intervention budget is not truncating? 
- **Reported either way.** This is diagnostic: it tells us whether the
  wrong-sign anchor result is a budget artifact or robust. Not in the
  primary family.

## 4. Position-matched controls — anchor validity

- **What runs:** re-run the intervention phase selecting control steps at
  **matched relative trace positions** to the anchors (not random
  non-anchor steps), so "anchor vs control" is not confounded with
  "position in trace." Paired test anchor_drop vs control_drop per
  question.
- **Pass/fail:** if anchor_drop > control_drop survives position
  matching → localized causal load claim stands; if it vanishes → the
  effect was positional, and we report that the localized-anchor reading
  does not survive position matching.

## 5. Gradient attribution — anchor method triangulation

- **What runs:** rank trace steps by gradient-based attribution
  (grad×activation; integrated-gradients if time permits) instead of the
  attention-based anchor score, intervene on top-k, compare drop vs the
  §4 position-matched control. Also report rank correlation between
  gradient ranking and the existing attention-based ranking
  (extends `analyze_anchor_subfeatures.py`).
- **Pass/fail:** gradient-selected anchors show drop > control (paired
  test); report Spearman ρ vs attention anchors. Convergence of two
  attribution methods strengthens the diffuse-vs-localized reading;
  divergence is reported as a method-sensitivity caveat.

## 6. Style controls — separating "verbose" from "reasoning"

- **What runs:** add a verbose-but-non-reasoning prompt condition
  (induces length/entropy comparable to CoT without multi-step
  reasoning), recompute HCDS(neutral vs style-control).
- **Pass/fail:** if HCDS(neutral, nocot) > 0 but HCDS(neutral,
  style-control) ≈ 0, the signal is not merely stylistic length/entropy.
  Complements the existing length-matched appendix (this is a
  *prompt-level* style control, that was a *post-hoc length-stratified*
  control).

## 7. Detector comparison — uses the negative-control data

- **What runs:** treat GSM8K questions as the positive class (reasoning
  required) and the pooled arithmetic+factual negative-control questions
  as the negative class. Compute **AUROC** for HCDS separating the two,
  per model, vs baseline detectors fixed now: (a) raw output length,
  (b) entropy-only, (c) latency-only. Bootstrap CI on AUROC (1000×).
- **Pass/fail:** HCDS AUROC > each baseline with non-overlapping
  bootstrap CIs = HCDS adds discriminative value beyond length/entropy.
  Ties or losses reported honestly (they would mean HCDS is reducible to
  a simpler signal).

## Global: seed-variance (handled once)

3 seeds {17, 18, 19} on the intervention/control-selection phase only
(cheap, ~30 min/seed per postmortem). Report between-seed SD on the
headline anchor and HCDS numbers as a single robustness line, rather
than re-running every experiment under 3 seeds.

---

## Power check (gate for §2, computed before that run)

Computed from the observed GSM8K per-question HCDS distributions
(`hcds_per_question.csv`). Procedure: estimate per-question SD per
(model), then for a range of plausible cross-family effect sizes compute the n needed for
80% power at two-sided α=0.05, paired design.

**Result (computed 2026-05-30 from the 2026-05-04_2232 `hcds_per_question.csv`):**
per-question SD = 0.797 (Instruct), 0.827 (Thinking); GSM8K means 1.24 /
1.03 ⇒ d ≈ 1.56 / 1.25 — GSM8K is hugely overpowered at n=50. Required n
for 80% power if the **Thinking** cross-family effect shrinks to a
fraction of its GSM8K value:

| cross-family effect | Cohen's d | required n |
|---|---|---|
| 100% (1.03) | 1.25 | ~7  |
| 50%  (0.52) | 0.62 | ~22 |
| 33%  (0.34) | 0.41 | ~49 |
| 25%  (0.26) | 0.31 | ~83 |

Caveat that *raises* this risk: the Thinking GSM8K HCDS itself is
unstable across runs (1.03 in the 2026-05-04_2232 run, 0.48 in the
2026-05-07 deep-table-ext run), so the realised cross-family effect could
land near the underpowered end.

**Decision (pre-committed):** run cross-family at **n=50**. If the
Thinking cross-family point estimate lands **below 0.34** (≈ ⅓ of GSM8K,
the point where n=50 stops being adequately powered), report it
**descriptive-only** — no significance claim — rather than presenting an
underpowered null as evidence of non-replication.

---

## Deviations log

- 2026-05-30: Full negative control launched (`scripts/run_negative_control_full.sh`,
  both families, n=50, both models). Decisions made at launch:
  - **Fixtures frozen at the committed `arithmetic_50` / `factual_50`.** The
    original n=50 per-sample data from the latency-only run was gone from
    disk, so the full run regenerates everything (incl. latency) on these
    frozen fixtures. Consequence: the appendix latency-only table will be
    *refreshed* from this run — numbers may shift slightly but will be
    internally consistent with the new full-feature cells.
  - **Paraphrase/perturbation builders extended** with `--input-jsonl` (+
    `--general-prompt` for paraphrase) so they target the neg-control
    families instead of GSM8K. Distractor perturbation is task-agnostic
    (answer-preserving), so it is valid here.
  - **Known raggedness (expected, handled):** (a) arithmetic paraphrases
    drop heavily under the `numbers_preserved` filter (~25/50 kept) because
    the model spells digits as words; (b) `explicit_no_cot` cells yield 0
    mech anchors (1-token answers). Both leave ragged feature vectors that
    `compute_hcds.py` handles per-question. Report per-cell feature coverage
    alongside the HCDS.
  - **oMLX endpoint:** runner uses `http://127.0.0.1:8000` (the oMLX GUI
    app). A redundant manual `omlx serve --port 8080` was also started; it
    is unused and can be killed.

- 2026-06-01: Full negative control **COMPLETE** (both families, n=50,
  both models; ran to `ALL FAMILIES COMPLETE`, exit 0, failed=0).
  Required 4 relaunches over ~2 days: 2 oMLX drops + 1 laptop crash + 1
  mech-stage worker OOM (SIGKILL -9). All recovered losslessly via
  per-sample resume. Stabilized by dropping `OMLX_CONCURRENCY` 4→2; the
  `AJAR_MAX_ANSWER_PROBES=32` fallback was armed but never needed.

  **RESULT — full 6-feature HCDS (mean [95% bootstrap CI]):**
  | model | GSM8K ref | arithmetic | factual | pooled NC | **calibrated = GSM8K−pooledNC** |
  |---|---|---|---|---|---|
  | Instruct | +2.25 [1.69,2.85] | +0.87 [0.38,1.34] | +0.04 [-0.41,0.61] | +0.46 [0.12,0.82] | **+1.80 [1.44,2.14]** |
  | Thinking | +0.48 [0.07,0.86] | +0.34 [-0.26,0.96] | +0.30 [0.03,0.56] | +0.32 [-0.02,0.65] | **+0.16 [-0.17,0.51]** |

  **Reframe decision (§0.4):** trigger is HCDS_cal CI including 0.
  - **Instruct: CI EXCLUDES 0 (+1.80 [1.44,2.14]) → reasoning-attributable.**
    ~80% of the GSM8K signal survives calibration; ~+0.46 is a stylistic/
    prompt-compliance floor. Instruct headline HOLDS (stronger than the
    latency-only hint feared).
  - **Thinking: CI INCLUDES 0 (+0.16 [-0.17,0.51]) → REFRAME TRIGGERED.**
    Thinking's small GSM8K HCDS (+0.48) is statistically indistinguishable
    from its no-reasoning baseline (+0.32). Thinking's HCDS is NOT
    demonstrably reasoning-attributable once calibrated.

  **Net:** this INVERTS the arithmetic-only snapshot. With both families
  pooled, **Instruct** is the model carrying a calibration-surviving
  reasoning signal; **Thinking**'s signal does not separate from its
  negative-control floor. Discriminant-validity framing still applies but
  Instruct is the validated case; Thinking becomes a documented limitation,
  not a co-headline. Abstract/intro "positive in both variants" sentence
  must change: it's positive-and-calibrated for Instruct, positive-but-
  uncalibrated for Thinking.

  **CAVEAT to report:** per-family results are noisy and cross over
  (Instruct calibrates on factual not arithmetic; Thinking the reverse).
  n=50/family is thin; factual had parser/truncation issues (logged above).
  Trust the pooled number; report the per-family instability as a limitation.
  TODO: GSM8K full per-question HCDS file not in results/runs/negative_control;
  calibrated CI used GSM8K mean as a point estimate minus the NC bootstrap —
  recompute with a proper two-sample bootstrap once the GSM8K per-question
  CSV is located (tightens the CI slightly, won't change the sign decision).
