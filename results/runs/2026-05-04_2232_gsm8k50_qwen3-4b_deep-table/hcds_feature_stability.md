# HCDS feature-stability check

Robustness check on the headline HCDS result. We computed HCDS five additional times, each time dropping a different feature (or feature group) from the six-dimensional feature vector. If the conclusion is robust, the sign and significance should hold across every subset. If a single feature is carrying the signal, dropping it should make the result evaporate.

## Results

| Variant (features used) | Instruct mean | Instruct 95% CI | Instruct p | Thinking mean | Thinking 95% CI | Thinking p |
|---|---:|---:|---:|---:|---:|---:|
| **all 6 features** (baseline) | **+2.25** | [+1.69, +2.85] | **2.2e-10** | **+0.48** | [+0.07, +0.86] | **0.021** |
| drop `paraphrase_consistency` | +2.02 | [+1.51, +2.59] | 5.4e-10 | +0.61 | [+0.21, +0.97] | 0.0032 |
| drop `mechanistic_intervention_delta_accuracy` | +2.70 | [+2.15, +3.25] | 3.7e-13 | +0.68 | [+0.27, +1.06] | 0.0014 |
| drop both para + mech | +2.49 | [+1.99, +3.00] | 4.7e-13 | +0.88 | [+0.50, +1.24] | 3.6e-05 |
| drop `perturbation_delta_accuracy` | +2.18 | [+1.67, +2.72] | 8.1e-11 | +0.51 | [+0.17, +0.85] | 0.0075 |
| **drop entropy** (mean + slope) | +1.72 | [+1.24, +2.15] | 7.4e-9 | **+0.09** | [-0.30, +0.43] | **0.617** ❌ |

## Interpretation

**Instruct passes every subset.** Every variant produces HCDS strictly above zero with p < 1e-8 and 95% CIs comfortably above zero. The headline finding for the Instruct model survives any reasonable feature-set choice — it isn't an artefact of one particular metric.

**Thinking's signal is entropy-driven.** Dropping the two entropy features (`token_entropy_mean` + `token_entropy_slope`) collapses HCDS from +0.48 (p=0.021) to +0.09 (p=0.617). The 95% confidence interval crosses zero ([-0.30, +0.43]). Under that subset, we cannot reject the null hypothesis that neutral_strict and explicit_no_cot are equidistant from explicit_cot for the Thinking model.

This is consistent with what we already know about the Thinking model:

- Thinking + explicit_no_cot accuracy (82%) is HIGHER than Thinking + explicit_cot accuracy (52%, depressed by truncation).
- Thinking ignores the explicit_no_cot directive and emits ~600 tokens of reasoning anyway.
- Output_tokens, latency, paraphrase consistency, and perturbation Δ don't strongly differentiate Thinking's three prompt conditions because the underlying behaviour is similar across them.
- The thing that DOES differentiate is the *entropy pattern* — token-level confidence dynamics differ even when the visible behaviour looks similar.

**Two valid framings for the paper:**

1. *The Thinking model genuinely uses hidden CoT, and the mechanistic signature lives in entropy.* The model behaves outwardly the same way regardless of prompt, but its internal token-level uncertainty profile differs in a way that aligns the neutral condition with explicit CoT. This is a non-trivial finding — entropy as a fingerprint of latent reasoning.

2. *The Thinking model's HCDS signal is fragile and depends on a single feature class.* Conservatively, we should report Thinking with a much larger caveat than Instruct, or only report HCDS as a sanity check rather than as a headline number for that model.

The right framing depends on whether the entropy-driven Thinking signal replicates on a second dataset. If StrategyQA gives the same picture (Thinking-HCDS-collapses-without-entropy), that's a real finding. If it doesn't replicate, this is a single-dataset oddity and we should de-emphasise the Thinking result.

## What this means for the headline claim

The headline finding becomes:

> **Instruct: hidden CoT is detectable across every measurement axis.** HCDS positive with p < 1e-8 under any subset of the six features. Robust.
>
> **Thinking: hidden CoT is detectable, but the signal is concentrated in entropy.** HCDS positive (p=0.021) when all features are used, but drops to non-significance (p=0.617) when entropy features are removed. Suggests the Thinking model's internal token-level uncertainty patterns are the load-bearing evidence; surface behaviour is too similar across conditions to discriminate.

Both are interesting findings. The Thinking case is just a more nuanced one and needs the entropy framing in the paper rather than being reported with the same confidence as the Instruct case.

## Files generated

Each variant wrote `hcds_per_question_<label>.csv` and `hcds_summary_<label>.csv` to this directory. Labels: `no_paraphrase`, `no_mech`, `no_para_no_mech`, `no_perturb`, `no_entropy`. The unlabelled files (`hcds_per_question.csv`, `hcds_summary.csv`) remain the all-6-features baseline.

## Reproducing

```bash
TABLE=results/runs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/task6_table.csv
OUT=results/runs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table

python3 scripts/compute_hcds.py --task6-csv $TABLE --out-dir $OUT \
    --label no_paraphrase --exclude-features paraphrase_consistency
python3 scripts/compute_hcds.py --task6-csv $TABLE --out-dir $OUT \
    --label no_mech --exclude-features mechanistic_intervention_delta_accuracy
python3 scripts/compute_hcds.py --task6-csv $TABLE --out-dir $OUT \
    --label no_para_no_mech \
    --exclude-features paraphrase_consistency,mechanistic_intervention_delta_accuracy
python3 scripts/compute_hcds.py --task6-csv $TABLE --out-dir $OUT \
    --label no_perturb --exclude-features perturbation_delta_accuracy
python3 scripts/compute_hcds.py --task6-csv $TABLE --out-dir $OUT \
    --label no_entropy --exclude-features token_entropy_mean,token_entropy_slope
```
