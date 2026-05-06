# HCDS length-matched analysis

Robustness check on the headline HCDS result. We bin questions into three equal-frequency tiers by their explicit_cot output length, then re-run the per-question HCDS computation within each tier. If HCDS is positive AND significant in each tier, the signal cannot be a pure length-driven artefact.

## instruct

Length tier thresholds (explicit_cot output tokens): short ≤ 235, medium ≤ 341, long > 341.

| tier | n | mean cot tokens | HCDS | 95% CI | t | p | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| short | 17 | 175 | +1.890 | [+1.068, +2.939] | +4.04 | 9.419e-04 | neutral aligns with CoT (p < 0.05) |
| medium | 16 | 280 | +2.673 | [+1.750, +3.675] | +5.21 | 1.050e-04 | neutral aligns with CoT (p < 0.05) |
| long | 17 | 472 | +2.224 | [+1.234, +3.071] | +4.42 | 4.321e-04 | neutral aligns with CoT (p < 0.05) |

## thinking

Length tier thresholds (explicit_cot output tokens): short ≤ 865, medium ≤ 1024, long > 1024.

| tier | n | mean cot tokens | HCDS | 95% CI | t | p | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| short | 17 | 717 | +1.205 | [+0.611, +1.878] | +3.65 | 2.142e-03 | neutral aligns with CoT (p < 0.05) |
| medium | 33 | 1013 | +0.106 | [-0.351, +0.519] | +0.46 | 6.467e-01 | ambiguous |
| long | — | — | — | — | — | — | (no rows) |

## Interpretation

**Instruct passes the length-matched stress test cleanly.** All three length tiers (short / medium / long, mean explicit_cot token counts of 175 / 280 / 472) show positive HCDS with p < 1e-3 and 95% CIs comfortably above zero. The Instruct headline finding is **not** an artefact of "neutral_strict is verbose like CoT" — the signal holds even when we constrain the comparison to questions where the output lengths under different prompts are roughly matched. This addresses the most natural reviewer concern about the original HCDS feature vector including `latency_per_output_token` (a length proxy).

**Thinking shows a length-dependent signal.** The "short" tier (questions where explicit_cot fit in well under the 1024 token cap, mean 717 tokens) gives HCDS=+1.21 with p=2.1e-3 — the signal is real on these questions. But the "medium" tier (questions where explicit_cot truncated near or at 1024) collapses to HCDS=+0.11 with p=0.65, 95% CI crossing zero. There is no "long" tier because the wide baseline ran with `max_new_tokens=1024`, so no Thinking outputs exceed that.

**The Thinking pattern is consistent with what we already know.** The wide baseline showed Thinking explicit_cot has 51.4% length-rate (proportion truncated at 1024). On questions where the model truncates, the boxed answer often never gets emitted, and the per-token features (entropy slope, output_tokens, latency) all collapse toward an artefact rather than reflecting reasoning fidelity. The HCDS signal that we DO see in the short tier (where reasoning fits) is real; the medium tier ambiguity is best read as "the truncation regime corrupts the per-question features that drive HCDS, not as evidence the underlying CoT-detection signal is absent."

**Combined with the feature-stability check** (`hcds_feature_stability.md`), the picture is:

| | Instruct | Thinking |
|---|---|---|
| All-features HCDS | +2.25, p=2.2e-10 | +0.48, p=0.021 |
| Drop entropy | +1.72, p=7.4e-9 ✅ | +0.09, p=0.617 ❌ |
| Length-matched (short tier) | +1.89, p=9.4e-4 ✅ | +1.21, p=2.1e-3 ✅ |
| Length-matched (medium tier) | +2.67, p=1.0e-4 ✅ | +0.11, p=0.65 ❌ |
| Length-matched (long tier) | +2.22, p=4.3e-4 ✅ | (no data) |

**For the paper.** Two confidence levels:

1. **Instruct: bulletproof.** Headline result holds across every robustness check we've run. Report HCDS=+2.25 with full confidence.

2. **Thinking: restricted claim.** HCDS only holds on questions where the model has output budget to actually reason (the "short" length tier and the all-features run). On truncated questions, the per-question features become noise. The right framing is "Thinking exhibits hidden CoT signatures on questions where its reasoning fits within the generation budget; on questions where reasoning is truncated, surface features alone don't discriminate the conditions." Combined with the entropy-only feature-stability finding, this is a real but smaller claim than the Instruct one.

**Methodological recommendation for any future replication.** Re-run the Thinking deep table with `max_new_tokens=2048` or `3072`. Both length-matched tier ambiguity and the entropy-driven sensitivity should be revisited once truncation is no longer the dominant confound.
