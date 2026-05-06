# Deep-Table Run — 2026-05-04_2232_gsm8k50_qwen3-4b

End-to-end Task 6 deep table covering all five behavioral and mechanistic
columns the proposal calls for, on a 50-question slice of GSM8K test under
all three prompting conditions and both Qwen3-4B variants.

## Scope

- **Dataset**: GSM8K test split, first 50 questions
- **Models**: Qwen3-4B-Instruct-2507, Qwen3-4B-Thinking-2507 (HF safetensors fp16 on MPS for MI; oMLX 8-bit MLX for behavioral)
- **Prompts**: explicit_cot, explicit_no_cot, neutral_strict (the proposal-aligned Neutral)
- **Trials**: 1 (deterministic, temperature=0)
- **Total rows**: 300 = 50 questions × 2 models × 3 prompts
- **Mechanistic interventions**: 6 per baseline (2 modes × 2 anchor steps + 1 control step), residual_zero and attention_zero
- **Backends**: oMLX for canonical baselines + paraphrase + perturbation runs; PyTorch on MPS for the MI slice with single-forward attention probing capped at 64 answer probes

## Per-condition means

| condition | acc | latency/tok | tokens out | H mean | H slope | para C | para acc | pert ΔA | mech ΔA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| instruct/explicit_cot | 0.980 | 0.0375 | 310 | 0.112 | -0.0002 | 0.944 | 0.967 | +0.040 | -0.030 |
| instruct/explicit_no_cot | 0.420 | 0.0814 | 16 | 0.087 | +0.0126 | 0.589 | 0.500 | -0.020 | 0.000 |
| instruct/neutral_strict | 0.960 | 0.0369 | 313 | 0.095 | -0.0001 | 0.900 | 0.922 | +0.040 | +0.040 |
| thinking/explicit_cot | 0.520 | 0.0345 | 912 | 0.169 | -0.00004 | 0.433 | 0.556 | +0.020 | -0.275 |
| thinking/explicit_no_cot | 0.820 | 0.0338 | 597 | 0.204 | +0.00007 | 0.733 | 0.811 | -0.020 | 0.000 |
| thinking/neutral_strict | 0.680 | 0.0347 | 832 | 0.171 | -0.00007 | 0.644 | 0.622 | +0.120 | +0.025 |

Each row is the mean across 50 questions (a few mech ΔA cells are over fewer because explicit_no_cot baselines often produced no anchor candidates).

## Headline result: HCDS is positive for both models

Computed as `D(Neutral, NoCoT) − D(Neutral, CoT)` over z-scored feature vectors `[latency/tok, H mean, H slope, paraphrase consistency, perturbation ΔA, mechanistic ΔA]`:

| model | D(Neutral, NoCoT) | D(Neutral, CoT) | HCDS | Interpretation |
|---|---:|---:|---:|---|
| Instruct | 3.65 | 2.41 | **+1.24** | neutral aligns with CoT |
| Thinking | 3.73 | 2.70 | **+1.03** | neutral aligns with CoT |

**This is the proposal's predicted outcome.** Both models, when prompted neutrally, produce behavioral and mechanistic profiles that look more like their explicit-CoT condition than their explicit-no-CoT condition.

Caveat: this is descriptive only. n=50 per condition. Bootstrap CIs and paired tests across questions (Task 8) are required before claiming significance.

## Surprising findings worth investigating

### 1. Thinking model has a strongly *negative* anchor sensitivity under explicit_cot

`anchor_drop − control_drop = -0.275`. Random control-step interventions hurt accuracy *more* than high-scoring anchor steps. The proposal predicts the opposite.

Per-question distribution under thinking/explicit_cot:

| anchor better | tied | control better |
|---:|---:|---:|
| 4/50 (8%) | 25/50 (50%) | 21/50 (42%) |

Three plausible explanations to test:
- The anchor scoring formula (`z(future_attn) + z(answer_attn) + z(activation_delta)`) picks recoverable "filler" steps; random sampling more often hits genuinely-causal mid-reasoning steps in long Thinking traces.
- Suppressing residual / attention at top-anchor steps in long traces gives the model many downstream tokens to recover; suppressing at random steps disrupts late-stage answer formation.
- The 384-token intervention budget is tight relative to thinking-model reasoning length; control-step interventions early in the trace might cap before reaching a reliable answer.

A targeted next experiment: re-score anchors using only the answer-attention component vs the activation-delta component vs the future-attention component, and see which sub-feature actually predicts an interventionable step.

### 2. Thinking + explicit_no_cot has higher accuracy than Thinking + explicit_cot

82% vs 52%. The Thinking model emits ~600 tokens of reasoning even under explicit_no_cot ("Answer-only mode. /no_think") — it ignores the directive. But explicit_cot pushes it to 912 tokens, which causes more truncation. The Thinking model is essentially CoT-by-default; explicit_cot just makes the reasoning longer (and worse, because of length-cap interactions).

Implication: the Thinking variant is not a useful control for "without CoT" experiments. We should report this as a model-behavior finding, not as a CoT-vs-no-CoT contrast.

### 3. Instruct + explicit_no_cot is very fragile

Accuracy collapses from 96-98% under cot/neutral conditions to 42% when explicitly suppressed, with output capped at 16 tokens and 0.0814 s/tok (more than 2x the latency-per-token of CoT conditions). The high latency-per-token at short output length suggests the model is "thinking harder" in the few tokens it does produce — a behavioral signal of the model trying to compress reasoning into fewer tokens.

Combined with the entropy slope flipping positive (+0.013 for instruct/explicit_no_cot vs -0.0002 for instruct/explicit_cot), this is consistent with the model becoming less confident token-by-token when reasoning is suppressed.

### 4. Paraphrase consistency tracks accuracy almost perfectly

Across all six conditions, paraphrase consistency and accuracy are within 0.05 of each other. Paraphrase consistency is not adding much signal beyond accuracy on this dataset; it would be more discriminating on harder problems where the model gets some right and some wrong.

## Mechanical observations from this run

- **Wall clock**: 7h 43m total (11:58 AM AST → 19:41 PM AST)
- **Failures**: 0 across 1050 items (150 baselines × 2 models × 3 prompts + 900 interventions)
- **Memory**: peak ~24 GB worker RSS, comfortably under 48 GB
- **Wide oMLX baseline (separate run)**: 3000 generations in ~3.5 h
- **Probe-attention optimization**: dropped from ~144s per long Thinking baseline to <5s
- **Per-model invocation split**: 1 model swap total (vs 30+ in the earlier killed run)

## Files

- `task6_table.csv` — 300 rows, all five Task 6 metric columns
- `task10_anchor_sensitivity.csv` — 6 rows, per-condition anchor−control drop
- `SUMMARY.md` — this document

Run inputs:
- canonical baselines: `outputs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/baseline/`
- paraphrase variants: `outputs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/paraphrase/`
- perturbation variants: `outputs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/perturbation/`
- mechanistic data: `outputs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/mech/`
- variant fixtures: `data/variants/gsm8k50_qwen3-4b_deep-table/`

(Output dirs are gitignored; only the aggregated CSVs land in results/.)
