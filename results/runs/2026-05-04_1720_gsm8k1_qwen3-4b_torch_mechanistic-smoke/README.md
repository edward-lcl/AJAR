# Qwen3 Torch Mechanistic Smoke

This run verifies that the Torch backend can produce real mechanistic artifacts for Qwen3-4B-Instruct on this machine.

Generated locally:

- `baseline.json`
- `tokens.csv`
- `hidden_summary.npz`
- `intervention_manifest.json`
- intervention JSON file
- `interventions.jsonl`
- `intervention_summary.csv`

Committed here:

- `baseline_summary.csv`
- `intervention_summary.csv`
- `sample_0000_tokens.csv`
- `run_config.json`
- `RUN_METADATA.md`

The first real Qwen mechanistic smoke completed successfully. It selected anchor step 14, targeted layers 18 and 19, ran `residual_scale_0.25`, and preserved the correct final answer.

Next scale target: 5 GSM8K examples with the same conservative mechanistic settings.

