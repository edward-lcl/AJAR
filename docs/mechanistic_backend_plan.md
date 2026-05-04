# Mechanistic Backend Plan

The current oMLX runs produced baseline text-generation metrics only. This is expected: the oMLX OpenAI-compatible API returns generated text and usage/timing metadata, but not hidden states, attentions, logits/logprobs, or forward hooks.

The existing runner already contains a Torch backend path that can produce mechanistic artifacts:

- `tokens.csv`
- token logprob rows
- hidden-state delta summaries
- attention probe summaries
- step scores
- anchor/control step selections
- intervention manifests
- `interventions.jsonl`
- `intervention_summary.csv`

## Recommended First Mechanistic Run

Start with a small slice before scaling:

```bash
AJAR_BACKEND=torch \
AJAR_OUTPUT_DIR=outputs/qwen3_gsm8k_mech_slice \
AJAR_NUM_SAMPLES=5 \
AJAR_PROMPTS=explicit_cot,neutral \
AJAR_MAX_NEW_TOKENS=512 \
AJAR_RUN_MI=1 \
python3 scripts/run_qwen3_gsm8k_mi.py
```

## Why Start Small

Mechanistic analysis is much more expensive than baseline generation. Each baseline can create multiple intervention continuations across:

- selected anchor steps
- selected control steps
- target layers
- intervention modes

Starting with 5-10 samples lets us validate the artifacts before committing to a larger run.

## Scaling Recommendation

1. Run 5 samples to validate output schema.
2. Run 25-50 samples for preliminary mechanistic trends.
3. Only then scale to 500+ samples.

Do not jump directly from the oMLX 500-example baseline to a full mechanistic run without confirming runtime and artifact size.

## Relation To Task 6

The current Task 6 starter table is available at:

```text
results/runs/2026-05-04_0238_gsm8k500_qwen3-4b_omlx_baseline-rerun-nocot-1024/task6_starter_table.csv
```

It contains baseline metrics. The mechanistic columns are placeholders until the Torch backend slice is run.
