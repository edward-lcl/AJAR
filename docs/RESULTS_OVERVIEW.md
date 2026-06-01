# AJAR — Full Results Overview

*Last updated: 2026-06-01. Supersedes `docs/STATUS_FOR_TEAM.md` (2026-05-06).*
*Negative-control calibration completed 2026-06-01 and materially changes the Thinking interpretation.*

---

## TL;DR

**Instruct is the validated finding. Thinking is not.**

The raw headline showed both Qwen3-4B variants with significant positive HCDS on GSM8K. The negative-control calibration (completed today) reveals that Thinking's signal is statistically indistinguishable from a non-reasoning baseline. Instruct's signal survives calibration cleanly — ~80% of it is reasoning-attributable.

---

## The Metric: HCDS

**Holistic Conditioning Divergence Score** measures whether neutral prompting produces behavioral and mechanistic profiles closer to explicit CoT than to explicit no-CoT, across 6 features:

1. Latency per output token
2. Token entropy mean
3. Token entropy slope
4. Paraphrase consistency
5. Perturbation delta accuracy
6. Mechanistic intervention delta accuracy

Per question: `HCDS = D(neutral, no_cot) − D(neutral, cot)` in z-scored feature space. Positive = neutral behaves like CoT.

---

## Headline Results: GSM8K n=50 (Primary Run, 2026-05-04)

| Model | Mean HCDS | 95% CI | p-value | Verdict |
|---|---|---|---|---|
| **Instruct** | **+2.25** | [+1.69, +2.85] | 2.2e-10 | Neutral aligns with CoT |
| **Thinking** | **+0.48** | [+0.07, +0.86] | 0.021 | Nominal significance |

Both are significant in isolation. The calibration below changes what this means.

---

## Negative-Control Calibration (Completed 2026-06-01)

We ran HCDS on two task families with no latent-reasoning interpretation:
- **Arithmetic controls**: digit counting, base conversion, mod arithmetic
- **Factual controls**: capital cities, element symbols, historical dates

Any HCDS signal in these families is **noise floor** — stylistic or format differences between conditions that are not reasoning-specific.

| Model | GSM8K HCDS | Noise Floor (pooled NC) | **Calibrated Signal** | **Verdict** |
|---|---|---|---|---|
| **Instruct** | +2.25 | +0.46 [+0.12, +0.82] | **+1.80 [+1.44, +2.14]** | CI excludes 0 ✅ |
| **Thinking** | +0.48 | +0.32 [−0.02, +0.65] | **+0.16 [−0.17, +0.51]** | CI includes 0 ✗ |

**Instruct**: 80% of the GSM8K HCDS is reasoning-attributable. Clean positive result.

**Thinking**: The raw GSM8K signal (+0.48) is not meaningfully larger than the non-reasoning baseline (+0.32). The calibrated effect is +0.16 with a CI spanning zero. We cannot claim latent reasoning detection for the Thinking variant on this evidence.

### Per-family breakdown

| Family | Instruct | p | Thinking | p |
|---|---|---|---|---|
| Arithmetic | +0.87 [+0.38, +1.34] | 0.0005 | +0.34 [−0.26, +0.96] | 0.103 |
| Factual | +1.18 [+0.92, +1.50] | <0.001 | −0.15 [−0.28, −0.03] | 0.034 |

*Note: Factual + Thinking had ~25% parser fallback due to 1024-token truncation. A 4096-token rerun is budgeted and would clean up the Thinking factual estimate.*

---

## Cross-Family Replication: StrategyQA n=50 (2026-05-06)

4-feature HCDS (no mechanistic features), yes/no questions.

| Model | Mean HCDS | 95% CI | p-value |
|---|---|---|---|
| **Instruct** | +1.87 | [+1.48, +2.28] | 3.5e-12 |
| **Thinking** | +0.60 | [+0.24, +0.95] | 0.002 |

Instruct replication is strong. Thinking is nominally significant here but this also predates negative-control calibration — the same noise-floor caveat likely applies.

---

## Scale Extension: GSM8K n=500 (2026-05-07)

3-feature HCDS (latency + entropy only).

| Model | Mean HCDS | 95% CI | p-value |
|---|---|---|---|
| **Instruct** | +2.38 | [+2.25, +2.50] | 1.98e-141 |
| **Thinking** | +0.33 | [+0.21, +0.45] | 1.85e-7 |

Instruct: essentially unlimited power at this scale — very robust. Thinking's effect shrinks to +0.33 at n=500, which is closer to the noise floor (+0.32) measured in the negative controls. Consistent with the calibration story.

---

## Per-Condition Behavioral Data (GSM8K n=50)

| Condition | Accuracy | Latency/tok | Output tokens | H mean | H slope | Para C |
|---|---:|---:|---:|---:|---:|---:|
| Instruct / explicit_cot | 0.98 | 0.037 | 310 | 0.112 | −0.0002 | 0.944 |
| Instruct / neutral | 0.96 | 0.037 | 313 | 0.095 | −0.0001 | 0.900 |
| Instruct / explicit_no_cot | 0.42 | 0.081 | 16 | 0.087 | +0.013 | 0.589 |
| Thinking / explicit_cot | 0.52 | 0.035 | 912 | 0.169 | −0.00004 | 0.433 |
| Thinking / neutral | 0.68 | 0.035 | 832 | 0.171 | −0.00007 | 0.644 |
| Thinking / explicit_no_cot | 0.82 | 0.034 | 597 | 0.204 | +0.00007 | 0.733 |

Key behavioral observations:

1. **Instruct no-CoT collapses to 42% accuracy** with 16-token outputs and 2× latency-per-token. The model genuinely relies on CoT and cannot compress it — this makes the CoT/no-CoT split a meaningful lower anchor.

2. **Thinking ignores the no-CoT directive**: 82% accuracy with 597-token outputs even when told to suppress reasoning. The explicit_no_cot condition is not a clean lower anchor for Thinking, which partly explains why Thinking's HCDS is small and noisy — the reference conditions are not well-separated.

3. **Entropy slope is positive only on no-CoT** (+0.013 Instruct, +0.00007 Thinking) — token-by-token confidence decreases when reasoning is suppressed, consistent with the model building on intermediate steps.

---

## Feature Ablation: What Drives the Signal?

Feature-isolated HCDS (Instruct, n=50):

| Feature | HCDS | p-value | Significant? |
|---|---|---|---|
| Latency alone | +1.80 | 2.6e-19 | ✅ |
| Entropy slope alone | +1.06 | 3.0e-6 | ✅ |
| Paraphrase consistency alone | +0.96 | 4.3e-6 | ✅ |
| Entropy mean alone | +0.74 | 6.6e-5 | ✅ |
| Perturbation delta alone | +0.25 | 0.261 | ✗ |
| Mechanistic intervention alone | ≈0 | 1.0 | ✗ |

Leave-one-out variants (Instruct stay significant across all; Thinking breaks when entropy is removed):

| Variant | Instruct HCDS | Thinking HCDS | Thinking sig? |
|---|---|---|---|
| Full 6-feature | +2.25 | +0.48 | ✅ |
| No mechanistic | +2.70 | +0.68 | ✅ |
| No paraphrase | +2.02 | +0.61 | ✅ |
| No perturbation | +2.18 | +0.51 | ✅ |
| **No entropy** | +1.72 | **+0.09** | **✗** |

**Instruct**: signal is broadly distributed across features — genuinely multi-dimensional.

**Thinking**: signal is concentrated in entropy. Remove entropy and Thinking HCDS is non-significant. Entropy is also the feature most susceptible to stylistic differences (longer outputs → different entropy profiles), which is exactly what the negative-control noise floor captures.

---

## Mechanistic Intervention Results

Anchor-selected steps vs. random control steps — how much does accuracy drop when a step is suppressed?

| Condition | Anchor drop | Control drop | Difference | Direction |
|---|---|---|---|---|
| Instruct / neutral | 0.08 | 0.04 | **+0.04** | ✅ Correct |
| Instruct / explicit_cot | 0.01 | 0.04 | −0.03 | ✗ |
| Instruct / explicit_no_cot | 0.125 | 0.125 | 0.00 | Null |
| Thinking / neutral | 0.275 | 0.25 | **+0.025** | ✅ Correct (small) |
| Thinking / explicit_cot | 0.105 | 0.38 | **−0.275** | ✗ Wrong direction |
| Thinking / explicit_no_cot | 0.14 | 0.14 | 0.00 | Null |

The Thinking + explicit_cot result is the problematic one: anchor-selected steps are *easier* to recover from than random controls — opposite of prediction. Most likely explanation: the anchor-scoring algorithm identifies late-trace summary steps (already committed reasoning) rather than causally load-bearing earlier steps. The model can recover from losing a summary token; it cannot recover from losing an intermediate calculation token that random selection hits.

Budgeted fix: position-matched anchor controls would resolve whether this is an anchor-selection artifact.

---

## What the Paper Can Claim

### Claims that survive the data

1. **Instruct: latent CoT alignment under neutral prompting is real.** HCDS = +2.25 raw, +1.80 calibrated. Multi-feature, cross-family, scales to n=500. This is a publishable, well-supported result.

2. **HCDS is a valid discriminator for Instruct-class models.** Feature ablation shows the signal is not reducible to a single proxy (length, entropy, etc.) — it's genuinely multi-dimensional for models where CoT suppression produces a meaningful behavioral floor.

3. **Thinking-class models require different methodology.** The core reason is that Thinking ignores no-CoT directives — the reference conditions collapse (no-CoT accuracy 82%, CoT 52%, reversed from Instruct), so the metric's anchoring breaks down. This is a finding about the limits of behavioral probing under always-on internal trace generation, not a null result on whether Thinking models reason.

### Claims that do not survive the data

- That HCDS detects latent reasoning in Thinking-class models (calibrated CI includes zero)
- That mechanistic intervention conclusively validates the anchor-selection methodology (Thinking CoT result is wrong-sign)

### Recommended framing for the paper

The paper becomes a **discriminant-validity** story: HCDS cleanly separates CoT-aligned neutral behavior in Instruct models, and simultaneously reveals that Thinking models' behavioral alignment is non-specific to reasoning tasks. That's a stronger and more honest contribution than claiming both variants — reviewers reward this kind of calibrated self-critique.

---

## Remaining Experiments

| Experiment | Purpose | Status | Est. runtime |
|---|---|---|---|
| 4096-token Thinking rerun (factual NC) | Fix ~25% parser fallback on factual negative control | ⏳ Not started | ~2 hr |
| Position-matched anchor controls | Disambiguate Thinking + CoT mechanistic anomaly | ⏳ Not started | ~4 hr |
| AUROC: HCDS vs length-only baseline | Show HCDS adds beyond output-length detection | ⏳ Not started | ~30 min (compute done, analysis only) |
| 3-seed variance check | Establish robustness of mechanistic intervention | ⏳ Not started | ~30 min per seed |

---

## Where Everything Lives

| Artifact | Path |
|---|---|
| Primary HCDS summary | `results/runs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/hcds_summary.csv` |
| Per-question HCDS (100 rows) | `results/runs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/hcds_per_question.csv` |
| Full feature table (300 rows) | `results/runs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/task6_table.csv` |
| Mechanistic anchor results | `results/runs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/task10_anchor_sensitivity.csv` |
| Feature ablation variants | `results/runs/sensitivity/hcds_subset_summary.csv` |
| n=500 extension | `results/runs/2026-05-07_0025_gsm8k500_qwen3-4b_deep-table-ext/hcds_summary.csv` |
| StrategyQA replication | `results/runs/2026-05-06_2121_strategyqa50_gsm8k50_qwen3-4b_deep-table/hcds_summary.csv` |
| Negative control (arithmetic) | `results/runs/negative_control/arithmetic/hcds_summary.csv` |
| Negative control (factual) | `results/runs/negative_control/factual/hcds_summary.csv` |
| Run index | `results/RUN_INDEX.md` |
| Camera-ready analysis spec | `docs/camera_ready_analysis_spec.md` |
| Scientific caveats | `docs/run_postmortem.md` |
