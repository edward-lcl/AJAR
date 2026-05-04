# Qwen3 GSM8K oMLX Initial Analysis

## Run Scope

- Output directory: `outputs/qwen3_gsm8k_real`
- Dataset: GSM8K `test` split, first 500 examples
- Rows analyzed: 2000 baseline generations
- Models: thinking, instruct
- Prompt conditions: explicit_cot, neutral
- Decoding: deterministic, `temperature=0.0`, `max_new_tokens=512`
- Mechanistic analysis and interventions: not run in oMLX mode

## Main Result

The strongest condition in this run is `instruct` / `neutral` at 461/500 correct (92.2%). The Thinking model aggregate accuracy should be treated carefully because many Thinking generations hit `finish_reason=length`, often before a boxed final answer.

## Condition Summary

| Model | Prompt | Accuracy | 95% bootstrap CI | Stop rate | Length rate | Boxed rate | Mean seconds | Mean output tokens | Mean steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| instruct | explicit_cot | 447/500 (89.4%) | 86.6%-92.0% | 90.8% | 9.2% | 91.4% | 4.94 | 281.0 | 26.1 |
| instruct | neutral | 461/500 (92.2%) | 89.8%-94.4% | 95.4% | 4.6% | 96.0% | 4.09 | 229.9 | 21.7 |
| thinking | explicit_cot | 117/500 (23.4%) | 20.0%-27.2% | 7.6% | 92.4% | 7.6% | 8.80 | 506.8 | 25.7 |
| thinking | neutral | 165/500 (33.0%) | 29.0%-37.0% | 19.8% | 80.2% | 20.2% | 8.50 | 488.1 | 22.1 |

## Prompt Effect

| Model | Prompt contrast | Accuracy delta | 95% bootstrap CI | Discordant pairs | McNemar p | Seconds delta | Tokens delta |
|---|---|---:|---:|---:|---:|---:|---:|
| instruct | neutral - explicit_cot | 2.8% | 1.2%-4.8% | 18 improved / 4 regressed | 0.004344 | -0.85 | -51.1 |
| thinking | neutral - explicit_cot | 9.6% | 4.8%-14.4% | 102 improved / 54 regressed | 0.0001504 | -0.30 | -18.6 |

## Model Effect

| Prompt | Instruct - Thinking accuracy | 95% bootstrap CI | Discordant pairs | McNemar p | Instruct - Thinking seconds | Instruct - Thinking tokens |
|---|---:|---:|---:|---:|---:|---:|
| explicit_cot | 66.0% | 61.8%-70.2% | 330 Instruct-only / 0 Thinking-only | 9.144e-100 | -3.86 | -225.8 |
| neutral | 59.2% | 54.8%-63.6% | 297 Instruct-only / 1 Thinking-only | 1.174e-87 | -4.42 | -258.2 |

## Truncation Check

| Model | Prompt | Finish reason | n | Accuracy | Boxed rate | Median tokens | Median steps |
|---|---|---:|---:|---:|---:|---:|---:|
| instruct | explicit_cot | length | 46 | 15.2% | 6.5% | 512 | 40 |
| instruct | explicit_cot | stop | 454 | 96.9% | 100.0% | 248 | 23 |
| instruct | neutral | length | 23 | 21.7% | 13.0% | 512 | 36 |
| instruct | neutral | stop | 477 | 95.6% | 100.0% | 205 | 20 |
| thinking | explicit_cot | length | 462 | 17.3% | 0.0% | 512 | 30 |
| thinking | explicit_cot | stop | 38 | 97.4% | 100.0% | 451 | 12 |
| thinking | neutral | length | 401 | 16.7% | 0.5% | 512 | 29 |
| thinking | neutral | stop | 99 | 99.0% | 100.0% | 394 | 5 |

## Interpretation

- The strongest baseline result is `instruct` under `neutral`: 461/500 correct, 92.2%.
- Explicit CoT is not helping on this run. It costs tokens and latency, and it increases truncation risk.
- The Thinking model should not be judged from aggregate accuracy until the max-token cap is fixed. When it stops normally, its accuracy is near the Instruct model; when it hits the cap, accuracy collapses and boxed-answer extraction usually fails.
- This run does not yet support a full Hidden CoT Detection Score because it lacks: explicit No-CoT condition, entropy/logprob traces, perturbation runs, paraphrase consistency, mechanistic interventions.
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
