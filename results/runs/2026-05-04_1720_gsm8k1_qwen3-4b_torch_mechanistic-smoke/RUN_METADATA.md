# Run Metadata

- Run ID: `2026-05-04_1720_gsm8k1_qwen3-4b_torch_mechanistic-smoke`
- Local raw output directory: `outputs/2026-05-04_1720_gsm8k1_qwen3-4b_torch_mechanistic-smoke/`
- Backend: `torch`
- Dataset: GSM8K test split, first 1 example
- Model: Qwen3-4B-Instruct-2507
- Prompt: neutral
- Max new tokens: 128
- Device: Apple MPS
- Precision: float16
- Mechanistic settings:
  - top anchor steps: 1
  - control steps: 0
  - top anchor layers: 2
  - interventions: `residual_scale_0.25`
  - full probe attention saved: no
- Result:
  - baselines: 1
  - interventions: 1
  - failures: 0
  - baseline correct: true
  - intervention correct: true
  - selected anchor step: 14
  - target layers: 18, 19

