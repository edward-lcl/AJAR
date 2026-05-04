# Slack Update Draft: Overnight Rerun

Overnight rerun completed cleanly: 3,000/3,000 generations, 0 failures. Runtime was about 3h28m with `AJAR_OMLX_CONCURRENCY=4`.

Run scope:
- GSM8K test split, first 500 examples
- Qwen3-4B-Thinking-2507-MLX-8bit and Qwen3-4B-Instruct-2507-MLX-8bit
- Three prompt conditions: explicit CoT, explicit No-CoT / answer-only, and neutral
- Deterministic decoding, `temperature=0.0`
- `max_new_tokens` increased from 512 to 1024

Main results:
- Instruct explicit CoT: 468/500, 93.6%
- Instruct neutral: 463/500, 92.6%
- Instruct No-CoT: 209/500, 41.8%
- Thinking explicit CoT: 270/500, 54.0%
- Thinking neutral: 338/500, 67.6%
- Thinking No-CoT: 410/500, 82.0%

What changed relative to the first run:
- The added No-CoT control was very informative. For Instruct, answer-only prompting collapses performance, while neutral and explicit CoT remain high. This suggests the model benefits heavily from producing intermediate reasoning text on GSM8K.
- Raising the token cap helped the Thinking model, but did not fully solve truncation. Thinking explicit CoT truncation dropped from 92.4% to 52.0%; Thinking neutral truncation dropped from 80.2% to 37.2%.
- When Thinking stops normally, it is highly accurate: 99.6% for explicit CoT, 99.0% for No-CoT, and 99.4% for neutral. The aggregate score is still mostly a truncation/completion-budget story.
- Concurrency worked: oMLX processed four requests in parallel without increasing failures.

Interpretation:
This rerun is a much stronger baseline than the first one. We now have the three prompt conditions needed for the behavioral side of HCDS. The strongest hidden-CoT-relevant result is that Instruct neutral behaves much closer to explicit CoT than to No-CoT in accuracy and output structure. For Thinking, interpretation is more complicated because the model is designed to think and often ignores No-CoT suppression; No-CoT should be labeled as prompt-level suppression, not a hard non-thinking mode.

Still missing before full HCDS:
- Token entropy / logprob traces
- Perturbation and paraphrase consistency
- Mechanistic interventions
- Controls for output length and truncation

Recommended next step:
Run a smaller perturbation/paraphrase slice with the same three prompt conditions, then add entropy/logprob capture via MLX-LM or a lower-level API path.
