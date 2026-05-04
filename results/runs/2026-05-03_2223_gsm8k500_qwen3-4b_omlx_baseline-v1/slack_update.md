# Slack Update Draft

Initial GSM8K baseline is complete. We ran 2,000 deterministic oMLX generations across the first 500 GSM8K test examples, Qwen3-4B-Thinking-2507-MLX-8bit and Qwen3-4B-Instruct-2507-MLX-8bit, under explicit CoT and neutral/direct prompts.

Top-line result: Qwen3 Instruct is strong on this slice. Neutral/direct got 461/500 correct, 92.2%, and explicit CoT got 447/500, 89.4%. Neutral/direct was also shorter and faster. Thinking looks weak in aggregate, but that is mostly a token-budget artifact: 80-92% of Thinking generations hit the 512-token cap, often before producing a boxed final answer. When Thinking stopped normally, accuracy was near 97-99%.

Interpretation: this is a successful throughput and baseline-behavior run, not yet a hidden-CoT claim. It aligns with the research plan by validating the model/dataset/prompt scaffold and exposing the first control issue: visible CoT length and answer extraction are confounded by truncation. We still need No-CoT, entropy/logprob traces, perturbations, paraphrase consistency, and mechanistic interventions before computing HCDS.

What I changed for the overnight rerun:
- Added `explicit_no_cot` as the third prompt condition.
- Added bounded oMLX request concurrency via `AJAR_OMLX_CONCURRENCY`.
- Increased the recommended overnight token budget to 1024.
- Made the analysis script reusable for any output directory.

Overnight rerun target: 500 GSM8K examples x 2 models x 3 prompts = 3,000 baseline generations. The goal is to wake up with the missing No-CoT control, lower truncation for Thinking, and a better comparison between neutral, explicit CoT, and answer-only behavior.

Main caveat: Qwen docs indicate Qwen3-Thinking-2507 supports thinking mode only, so the No-CoT condition for that model is prompt suppression, not a guaranteed hard non-thinking engine switch. That is still useful for the research question, but we should label it carefully.
