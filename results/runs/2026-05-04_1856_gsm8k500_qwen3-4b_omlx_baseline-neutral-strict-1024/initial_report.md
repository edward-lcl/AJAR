# Qwen3 GSM8K oMLX Initial Analysis

## Run Scope

- Output directory: `outputs/2026-05-04_1856_gsm8k500_qwen3-4b_omlx_baseline-neutral-strict-1024`
- Dataset: GSM8K `test` split, first 500 examples
- Rows analyzed: 3000 baseline generations
- Models: instruct, thinking
- Prompt conditions: explicit_cot, explicit_no_cot, neutral_strict
- Decoding: deterministic, `temperature=0.0`, `max_new_tokens=1024`
- Mechanistic analysis and interventions: not run in oMLX mode

## Main Result

The strongest condition in this run is `instruct` / `explicit_cot` at 469/500 correct (93.8%). The Thinking model aggregate accuracy should be treated carefully because many Thinking generations hit `finish_reason=length`, often before a boxed final answer.

## Condition Summary

| Model | Prompt | Accuracy | 95% bootstrap CI | Stop rate | Length rate | Boxed rate | Mean seconds | Mean output tokens | Mean steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| instruct | explicit_cot | 469/500 (93.8%) | 91.6%-95.8% | 97.8% | 2.2% | 97.8% | 11.51 | 300.7 | 27.4 |
| instruct | explicit_no_cot | 210/500 (42.0%) | 37.8%-46.2% | 99.8% | 0.2% | 99.8% | 1.17 | 24.0 | 1.2 |
| instruct | neutral_strict | 465/500 (93.0%) | 90.8%-95.2% | 97.8% | 2.2% | 42.8% | 10.49 | 276.9 | 26.1 |
| thinking | explicit_cot | 286/500 (57.2%) | 52.8%-61.4% | 48.6% | 51.4% | 48.8% | 31.27 | 886.6 | 33.3 |
| thinking | explicit_no_cot | 409/500 (81.8%) | 78.4%-85.2% | 79.2% | 20.8% | 79.4% | 19.97 | 574.9 | 12.1 |
| thinking | neutral_strict | 304/500 (60.8%) | 56.6%-65.0% | 56.2% | 43.8% | 25.0% | 29.06 | 818.0 | 27.0 |

## Prompt Effect

| Model | Prompt contrast | Accuracy delta | 95% bootstrap CI | Discordant pairs | McNemar p | Seconds delta | Tokens delta |
|---|---|---:|---:|---:|---:|---:|---:|
| instruct | explicit_cot - explicit_no_cot | 51.8% | 47.2%-56.6% | 265 improved / 6 regressed | 2.806e-70 | 10.34 | 276.7 |
| thinking | explicit_cot - explicit_no_cot | -24.6% | -28.6%--20.4% | 12 improved / 135 regressed | 1.647e-27 | 11.30 | 311.7 |

## Model Effect

| Prompt | Instruct - Thinking accuracy | 95% bootstrap CI | Discordant pairs | McNemar p | Instruct - Thinking seconds | Instruct - Thinking tokens |
|---|---:|---:|---:|---:|---:|---:|
| explicit_cot | 36.6% | 32.2%-40.8% | 185 Instruct-only / 2 Thinking-only | 1.792e-52 | -19.76 | -585.9 |
| explicit_no_cot | -39.8% | -44.8%--34.8% | 22 Instruct-only / 221 Thinking-only | 1.595e-42 | -18.80 | -550.9 |
| neutral_strict | 32.2% | 28.0%-36.4% | 166 Instruct-only / 5 Thinking-only | 7.91e-43 | -18.57 | -541.1 |

## Truncation Check

| Model | Prompt | Finish reason | n | Accuracy | Boxed rate | Median tokens | Median steps |
|---|---|---:|---:|---:|---:|---:|---:|
| instruct | explicit_cot | length | 11 | 9.1% | 0.0% | 1024 | 76 |
| instruct | explicit_cot | stop | 489 | 95.7% | 100.0% | 259 | 24 |
| instruct | explicit_no_cot | length | 1 | 0.0% | 0.0% | 1024 | 62 |
| instruct | explicit_no_cot | stop | 499 | 42.1% | 100.0% | 6 | 0 |
| instruct | neutral_strict | length | 11 | 0.0% | 0.0% | 1024 | 65 |
| instruct | neutral_strict | stop | 489 | 95.1% | 43.8% | 233 | 23 |
| thinking | explicit_cot | length | 257 | 17.1% | 0.4% | 1024 | 49 |
| thinking | explicit_cot | stop | 243 | 99.6% | 100.0% | 750 | 25 |
| thinking | explicit_no_cot | length | 104 | 16.3% | 1.0% | 1024 | 58 |
| thinking | explicit_no_cot | stop | 396 | 99.0% | 100.0% | 394 | 0 |
| thinking | neutral_strict | length | 219 | 15.5% | 2.3% | 1024 | 48 |
| thinking | neutral_strict | stop | 281 | 96.1% | 42.7% | 660 | 12 |

## Interpretation

- The strongest baseline result is `instruct` under `explicit_cot`: 469/500 correct, 93.8%.
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
