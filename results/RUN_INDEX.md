# Run Index

This index lists the committed, human-reviewable run summaries. Large raw outputs are ignored by git and live locally under `outputs/`.

| Run ID | Date | Purpose | Scope | Notes |
|---|---|---|---|---|
| `2026-05-03_2223_gsm8k500_qwen3-4b_omlx_baseline-v1` | 2026-05-03 22:23 | Initial baseline | GSM8K 500, 2 models, explicit CoT + neutral | Found major Thinking truncation at 512 tokens. |
| `2026-05-04_0238_gsm8k500_qwen3-4b_omlx_baseline-rerun-nocot-1024` | 2026-05-04 02:38 | Baseline rerun | GSM8K 500, 2 models, explicit CoT + explicit No-CoT + neutral | Added No-CoT, raised max tokens to 1024, used oMLX concurrency 4. |

## Next Planned Run

Recommended:

```text
YYYY-MM-DD_HHMM_gsm8k5_qwen3-4b_torch_mechanistic-slice
```

Purpose: verify that the Torch backend produces `tokens.csv`, step scores, anchor indices, and intervention records before scaling mechanistic analysis.

