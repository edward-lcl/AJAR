# Anchor-signal investigation — thinking / explicit_cot

This report investigates why the thinking model under `explicit_cot` shows the *wrong* sign on the `anchor_drop − control_drop` measurement: random control steps hurt accuracy more than the steps we picked as anchors. The proposal predicts the opposite.

## Distribution of `anchor_drop − control_drop` (n = 50)

- positive (anchor hurts more, expected): **4**
- zero (no difference): **25**
- negative (control hurts more, anomalous): **21**

## 1. Sub-feature comparison: do anchors actually score higher than controls?

If our anchor selection works, anchor steps should average higher on the three sub-features that go into `combined_anchor_score`. Comparing means across all anchor vs control steps for this condition:

| feature | anchor mean | control mean | anchor − control |
|---|---:|---:|---:|
| future_attn | 0.0515 | 0.0102 | 0.0413 |
| answer_attn | 0.0282 | 0.0044 | 0.0237 |
| activation_delta | 53.1977 | 51.0166 | 2.1811 |
| combined_score | 8.3119 | -0.0483 | 8.3602 |
| num_full_tokens | 18.7400 | 13.4400 | 5.3000 |

## 2. Top-1 anchor agreement under each sub-feature alone

If we picked the top anchor using ONLY one of the three sub-features (ignoring the other two), would we pick the same step the combined score did?

| sub-feature only | agreement |
|---|---:|
| future_attn | 3/50 (6%) |
| answer_attn | 3/50 (6%) |
| activation_delta | 3/50 (6%) |

## 3. Five worst cases — read what was actually intervened on

### Sample 1 (mech_dA = -1.000)

- Question (truncated): A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?…
- Baseline correct: True (prediction = 3, gold = 3)
- Anchor steps: [55, 52] → intervened-correct = 1
- Control steps: [1] → intervened-correct = 0
- Total reasoning steps: 57

**Anchor step texts:**
  - step 55: ### Final Answer…
  - step 52: \text{Total bolts} = \text{Blue fiber} + \text{White fiber} = 2 + 1 = 3…

**Control step texts:**
  - step 1: So the question is: A robe takes 2 bolts of blue fiber and half that much white fiber.…

### Sample 14 (mech_dA = -1.000)

- Question (truncated): In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the remaining enrolled in jazz dance, and the rest enrolled in h…
- Baseline correct: True (prediction = 60, gold = 60)
- Anchor steps: [105, 108] → intervened-correct = 1
- Control steps: [55] → intervened-correct = 0
- Total reasoning steps: 110

**Anchor step texts:**
  - step 105: \frac{12}{20} \times 100 = 60\%…
  - step 108: ### Final Answer…

**Control step texts:**
  - step 55: 25% of 80% is 20%, so jazz is 20% of the total.…

### Sample 15 (mech_dA = -1.000)

- Question (truncated): A merchant wants to make a choice of purchase between 2 purchase plans: jewelry worth $5,000 or electronic gadgets worth $8,000. His financi…
- Baseline correct: True (prediction = 125, gold = 125)
- Anchor steps: [70, 68] → intervened-correct = 1
- Control steps: [19] → intervened-correct = 0
- Total reasoning steps: 72

**Anchor step texts:**
  - step 70: ### **Final Answer**…
  - step 68: Since **$125 > $96**, the **jewelry** option yields a higher profit.…

**Control step texts:**
  - step 19: Then for electronic gadgets: they are worth $8,000 initially, market rises 1.2%, so profit is 8000 * 1.2% = 8000 * 0.012.…

### Sample 19 (mech_dA = -1.000)

- Question (truncated): Marissa is hiking a 12-mile trail. She took 1 hour to walk the first 4 miles, then another hour to walk the next two miles. If she wants her…
- Baseline correct: True (prediction = 6, gold = 6)
- Anchor steps: [109, 112] → intervened-correct = 1
- Control steps: [30] → intervened-correct = 0
- Total reasoning steps: 114

**Anchor step texts:**
  - step 109: \text{Required Speed} = \frac{\text{Remaining Distance}}{\text{Time Left}} = \frac{6}{1} = 6 \text{ miles per hour}…
  - step 112: ### Final Answer…

**Control step texts:**
  - step 30: \]…

### Sample 23 (mech_dA = -1.000)

- Question (truncated): A candle melts by 2 centimeters every hour that it burns. How many centimeters shorter will a candle be after burning from 1:00 PM to 5:00 P…
- Baseline correct: True (prediction = 8, gold = 8)
- Anchor steps: [52, 48] → intervened-correct = 1
- Control steps: [0] → intervened-correct = 0
- Total reasoning steps: 54

**Anchor step texts:**
  - step 52: After burning from 1:00 PM to 5:00 PM, the candle will be **8 centimeters shorter**.…
  - step 48: 2\ \text{cm/hour} \times 4\ \text{hours} = 8\ \text{centimeters}…

**Control step texts:**
  - step 0: Okay, let's try to figure out this problem step by step.…

## 4. Are anchor steps systematically later in the trace than control steps?

Position is normalised to [0.0, 1.0] over the reasoning chain length.

| step kind | mean relative position | n |
|---|---:|---:|
| anchor | 0.752 | 100 |
| control | 0.483 | 50 |

## Plain-language summary

**The anchor scoring formula is doing what it was designed to do, but the design is biased toward late-stage 'summary' or 'answer-formulation' steps rather than causally-important mid-reasoning steps.** Anchor steps score 5x higher than control steps on `future_attention_mean` and `answer_attention_mean` (the formula is picking exactly the steps the rest of the trace looks at the most). Read the worst-case study above: anchors land overwhelmingly on `### Final Answer` headers and on the line that emits the boxed equation. Control steps land much earlier — often on intermediate computation or restating the problem.

**Why this produces the wrong sign on the intervention measurement.** Suppressing residual or attention at a 'final answer' step is not very damaging because the model has already finished reasoning by that point and can recover the answer from earlier context. Suppressing the same magnitude of activation at an early/middle reasoning step disrupts a computation that downstream tokens haven't yet copied from. So control-step interventions hurt more than anchor-step interventions, and `anchor_drop − control_drop` comes out negative.

**This is a methodology finding, not a bug in the run data.** The anchor-scoring proxy (high future-attention + high answer-attention + high activation-delta) does not equal causal importance for long-trace Thinking-class outputs. It correlates with 'this is what other tokens look at', which by the end of the trace means 'this is the answer summary'. For the proposal's anchor-vs-control test to behave as predicted on this model, the scoring rule needs to be redesigned to favour mid-reasoning causal steps over post-hoc summaries.

**Suggested follow-ups, ranked by leverage:**

1. **Penalise late-trace steps in the anchor score.** Multiply `combined_anchor_score` by `(1 − step_index/num_steps)` or similar to shift selection earlier. Re-run intervention on a few samples and check the sign flips.
2. **Try gradient-based attribution instead of attention proxies.** Compute `d(answer_logit)/d(activation_at_step_i)` and pick anchors by attribution score. This is a real algorithmic change but would likely give causally-meaningful anchors.
3. **Drop `future_attention_mean` from the anchor score.** It's the component most biased toward late-trace summary steps. Score on `activation_delta_mean` alone, or `activation_delta + answer_attention` only.
4. **Larger intervention budget** is unlikely to flip the sign because the intervention generates from immediately after the anchor step — and on these cases the anchor step IS late in the trace, so the budget already reaches the answer.

**For the paper.** This is worth reporting as a finding, not just a caveat: the proposal's anchor-vs-control test is a useful diagnostic, but the specific anchor-scoring formula matters and attention-based proxies systematically pick late-stage steps for long traces. Either redesign the score to favour early/mid steps, or report the result alongside an alternative scoring rule and show the sign flip.
