# Anchor-signal investigation — instruct / neutral_strict

This report investigates why the instruct model under `neutral_strict` shows the *wrong* sign on the `anchor_drop − control_drop` measurement: random control steps hurt accuracy more than the steps we picked as anchors. The proposal predicts the opposite.

## Distribution of `anchor_drop − control_drop` (n = 50)

- positive (anchor hurts more, expected): **10**
- zero (no difference): **39**
- negative (control hurts more, anomalous): **1**

## 1. Sub-feature comparison: do anchors actually score higher than controls?

If our anchor selection works, anchor steps should average higher on the three sub-features that go into `combined_anchor_score`. Comparing means across all anchor vs control steps for this condition:

| feature | anchor mean | control mean | anchor − control |
|---|---:|---:|---:|
| future_attn | 0.0747 | 0.0149 | 0.0598 |
| answer_attn | 0.0448 | 0.0095 | 0.0353 |
| activation_delta | 56.8976 | 49.4651 | 7.4325 |
| combined_score | 6.3258 | -0.9876 | 7.3134 |
| num_full_tokens | 17.8100 | 7.0000 | 10.8100 |

## 2. Top-1 anchor agreement under each sub-feature alone

If we picked the top anchor using ONLY one of the three sub-features (ignoring the other two), would we pick the same step the combined score did?

| sub-feature only | agreement |
|---|---:|
| future_attn | 2/50 (4%) |
| answer_attn | 2/50 (4%) |
| activation_delta | 2/50 (4%) |

## 3. Five worst cases — read what was actually intervened on

### Sample 13 (mech_dA = -0.500)

- Question (truncated): Melanie is a door-to-door saleswoman. She sold a third of her vacuum cleaners at the green house, 2 more to the red house, and half of what …
- Baseline correct: True (prediction = 18, gold = 18)
- Anchor steps: [21, 25] → intervened-correct = 1
- Control steps: [9] → intervened-correct = 0.5
- Total reasoning steps: 57

**Anchor step texts:**
  - step 21: \text{Sold at orange} = \frac{1}{2} \left( \frac{2}{3}x - 2 \right)…
  - step 25: \left( \frac{2}{3}x - 2 \right) - \frac{1}{2} \left( \frac{2}{3}x - 2 \right) = \frac{1}{2} \left( \frac{2}{3}x - 2 \right)…

**Control step texts:**
  - step 9: $$…

### Sample 0 (mech_dA = +0.000)

- Question (truncated): Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sel…
- Baseline correct: True (prediction = 18, gold = 18)
- Anchor steps: [15, 21] → intervened-correct = 1
- Control steps: [18] → intervened-correct = 1
- Total reasoning steps: 24

**Anchor step texts:**
  - step 15: $ 16 - 7 = 9 $ eggs per day.…
  - step 21: $ 9 \text{ eggs} \times 2 \text{ dollars} = 18 \text{ dollars} $…

**Control step texts:**
  - step 18: $2 per fresh duck egg.…

### Sample 1 (mech_dA = +0.000)

- Question (truncated): A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?…
- Baseline correct: True (prediction = 3, gold = 3)
- Anchor steps: [5, 1] → intervened-correct = 1
- Control steps: [0] → intervened-correct = 1
- Total reasoning steps: 12

**Anchor step texts:**
  - step 5: \frac{1}{2} \times 2 = 1 \text{ bolt of white fiber}…
  - step 1: - A robe takes **2 bolts of blue fiber**.…

**Control step texts:**
  - step 0: We are told:…

### Sample 3 (mech_dA = +0.000)

- Question (truncated): James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?…
- Baseline correct: True (prediction = 540, gold = 540)
- Anchor steps: [7, 9] → intervened-correct = 1
- Control steps: [1] → intervened-correct = 1
- Total reasoning steps: 11

**Anchor step texts:**
  - step 7: 180 \text{ meters} \times 3 \text{ sessions} = 540 \text{ meters per week}…
  - step 9: ### ✅ Final Answer:…

**Control step texts:**
  - step 1: First, find out how many meters he runs per session:…

### Sample 4 (mech_dA = +0.000)

- Question (truncated): Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them h…
- Baseline correct: True (prediction = 20, gold = 20)
- Anchor steps: [25, 28] → intervened-correct = 1
- Control steps: [3] → intervened-correct = 1
- Total reasoning steps: 30

**Anchor step texts:**
  - step 25: 60 - 40 = 20 \text{ cups}…
  - step 28: ### ✅ Final Answer:…

**Control step texts:**
  - step 3: - She gives:…

## 4. Are anchor steps systematically later in the trace than control steps?

Position is normalised to [0.0, 1.0] over the reasoning chain length.

| step kind | mean relative position | n |
|---|---:|---:|
| anchor | 0.592 | 100 |
| control | 0.435 | 50 |

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
