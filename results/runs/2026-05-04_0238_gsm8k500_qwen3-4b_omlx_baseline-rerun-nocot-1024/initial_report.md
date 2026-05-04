# Qwen3 GSM8K oMLX Initial Analysis

## Run Scope

- Output directory: `outputs/qwen3_gsm8k_tonight`
- Dataset: GSM8K `test` split, first 500 examples
- Rows analyzed: 3000 baseline generations
- Models: thinking, instruct
- Prompt conditions: explicit_cot, explicit_no_cot, neutral
- Decoding: deterministic, `temperature=0.0`, `max_new_tokens=1024`
- Mechanistic analysis and interventions: not run in oMLX mode

## Main Result

The strongest condition in this run is `instruct` / `explicit_cot` at 468/500 correct (93.6%). The Thinking model aggregate accuracy should be treated carefully because many Thinking generations hit `finish_reason=length`, often before a boxed final answer.

## Condition Summary

| Model | Prompt | Accuracy | 95% bootstrap CI | Stop rate | Length rate | Boxed rate | Mean seconds | Mean output tokens | Mean steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| instruct | explicit_cot | 468/500 (93.6%) | 91.4%-95.6% | 97.6% | 2.4% | 97.6% | 11.46 | 302.4 | 27.5 |
| instruct | explicit_no_cot | 209/500 (41.8%) | 37.6%-46.2% | 99.8% | 0.2% | 99.8% | 1.17 | 24.0 | 1.2 |
| instruct | neutral | 463/500 (92.6%) | 90.2%-94.8% | 98.0% | 2.0% | 98.0% | 9.30 | 246.4 | 22.8 |
| thinking | explicit_cot | 270/500 (54.0%) | 49.6%-58.2% | 48.0% | 52.0% | 48.0% | 30.83 | 880.9 | 32.9 |
| thinking | explicit_no_cot | 410/500 (82.0%) | 78.6%-85.2% | 79.8% | 20.2% | 80.0% | 20.20 | 578.1 | 12.0 |
| thinking | neutral | 338/500 (67.6%) | 63.4%-71.6% | 62.8% | 37.2% | 62.8% | 26.98 | 779.6 | 24.6 |

## Prompt Effect

| Model | Prompt contrast | Accuracy delta | 95% bootstrap CI | Discordant pairs | McNemar p | Seconds delta | Tokens delta |
|---|---|---:|---:|---:|---:|---:|---:|
| instruct | neutral - explicit_cot | -1.0% | -2.4%-0.6% | 5 improved / 10 regressed | 0.3018 | -2.16 | -56.0 |
| instruct | neutral - explicit_no_cot | 50.8% | 46.2%-55.4% | 260 improved / 6 regressed | 8.024e-69 | 8.13 | 222.4 |
| instruct | explicit_cot - explicit_no_cot | 51.8% | 47.2%-56.4% | 264 improved / 5 regressed | 2.43e-71 | 10.29 | 278.4 |
| thinking | neutral - explicit_cot | 13.6% | 9.6%-17.6% | 94 improved / 26 regressed | 3.14e-10 | -3.85 | -101.3 |
| thinking | neutral - explicit_no_cot | -14.4% | -18.2%--10.6% | 17 improved / 89 regressed | 5.912e-13 | 6.78 | 201.5 |
| thinking | explicit_cot - explicit_no_cot | -28.0% | -32.6%--23.6% | 12 improved / 152 regressed | 4.854e-32 | 10.63 | 302.8 |

## Model Effect

| Prompt | Instruct - Thinking accuracy | 95% bootstrap CI | Discordant pairs | McNemar p | Instruct - Thinking seconds | Instruct - Thinking tokens |
|---|---:|---:|---:|---:|---:|---:|
| explicit_cot | 39.6% | 35.0%-44.0% | 201 Instruct-only / 3 Thinking-only | 1.101e-55 | -19.37 | -578.6 |
| explicit_no_cot | -40.2% | -45.2%--35.2% | 20 Instruct-only / 221 Thinking-only | 4.963e-44 | -19.03 | -554.1 |
| neutral | 25.0% | 21.2%-29.0% | 132 Instruct-only / 7 Thinking-only | 5.166e-31 | -17.68 | -533.2 |

## Truncation Check

| Model | Prompt | Finish reason | n | Accuracy | Boxed rate | Median tokens | Median steps |
|---|---|---:|---:|---:|---:|---:|---:|
| instruct | explicit_cot | length | 12 | 8.3% | 0.0% | 1024 | 72 |
| instruct | explicit_cot | stop | 488 | 95.7% | 100.0% | 260 | 24 |
| instruct | explicit_no_cot | length | 1 | 0.0% | 0.0% | 1024 | 58 |
| instruct | explicit_no_cot | stop | 499 | 41.9% | 100.0% | 6 | 0 |
| instruct | neutral | length | 10 | 0.0% | 0.0% | 1024 | 74 |
| instruct | neutral | stop | 490 | 94.5% | 100.0% | 210 | 20 |
| thinking | explicit_cot | length | 260 | 11.9% | 0.0% | 1024 | 49 |
| thinking | explicit_cot | stop | 240 | 99.6% | 100.0% | 736 | 24 |
| thinking | explicit_no_cot | length | 101 | 14.9% | 1.0% | 1024 | 59 |
| thinking | explicit_no_cot | stop | 399 | 99.0% | 100.0% | 404 | 0 |
| thinking | neutral | length | 186 | 14.0% | 0.0% | 1024 | 51 |
| thinking | neutral | stop | 314 | 99.4% | 100.0% | 622 | 10 |

## Interpretation

- The strongest baseline result is `instruct` under `explicit_cot`: 468/500 correct, 93.6%.
- Explicit CoT is not helping on this run. It costs tokens and latency, and it increases truncation risk.
- The Thinking model should not be judged from aggregate accuracy until the max-token cap is fixed. When it stops normally, its accuracy is near the Instruct model; when it hits the cap, accuracy collapses and boxed-answer extraction usually fails.
- This run does not yet support a full Hidden CoT Detection Score because it lacks: entropy/logprob traces, perturbation runs, paraphrase consistency, mechanistic interventions.
- Treat current results as a successful throughput and baseline-behavior run, not as final evidence for or against hidden CoT.

## Recommended Next Run

1. Add the third prompt condition: explicit No-CoT / answer-only.
2. Increase Thinking model `max_new_tokens` substantially, or use a two-budget setup: small answer-only budget and larger CoT/thinking budget.
3. Preserve boxed-answer formatting across all conditions so scoring remains comparable.
4. Add async OpenAI-compatible oMLX requests with a configurable concurrency limit and benchmark concurrency levels 1, 2, 4, and 8.
5. Add a perturbation/paraphrase slice before scaling to full GSM8K: enough to compute preliminary consistency and perturbation fragility.
6. Use MLX-LM Python or a Torch backend for logits/entropy and mechanistic traces; the current oMLX API result records do not include token-level entropy.

## Generated Files

- `condition_summary.csv`
- `finish_reason_summary.csv`
- `paired_prompt_deltas.csv`
- `paired_model_deltas.csv`
- `prompt_disagreement_examples.csv`
