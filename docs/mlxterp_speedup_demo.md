# mlxterp.generate speedup + intervention demo on AJAR baselines

Same five GSM8K questions used in the 2026-05-04 deep-table run, same Qwen3-4B-Instruct-MLX-8bit weights. Reference timings are the recorded `generation_seconds` from oMLX's HTTP server during that run; mlxterp timings are from `InterpretableModel.generate` in-process.

Model: `lmstudio-community/Qwen3-4B-Instruct-2507-MLX-8bit`

Max tokens: 512

| sample | gold | oMLX (s) | mlxterp (s) | speedup | +iv L5 (s) | oMLX pred | mlxterp pred | +iv pred |
|---|---|---|---|---|---|---|---|---|
| 0 | 18 | 10.44 | 9.02 | 1.16x | 10.86 | 18 | 18 | 18 |
| 1 | 3 | 4.20 | 8.98 | 0.47x | 10.66 | 3 | 3 | 3 |
| 2 | 70000 | 15.62 | 8.98 | 1.74x | 10.63 | 70000 | 70000 | 70000 |
| 3 | 540 | 6.79 | 8.93 | 0.76x | 10.59 | 540 | 540 | 540 |
| 4 | 20 | 11.19 | 9.08 | 1.23x | 10.70 | 20 | 20 | 20 |

## Aggregate

- oMLX mean:    9.65s
- mlxterp mean: 9.00s
- mean speedup: **1.07x**
- +iv mean:     10.69s
- prediction agreement (oMLX vs mlxterp):  5/5
- correctness agreement (oMLX vs mlxterp): 5/5
- intervention changed answer on:          0/5

## Caveats

- Reference oMLX timings include HTTP overhead and JSON serialisation, so the speedup is for the AJAR pipeline as a whole, not a pure inference comparison. That is the right comparison: AJAR was paying that overhead.
- The intervention pass uses a tiny additive vector (N(0, 0.05) at layer 5, generated tokens only). It is meant to demonstrate the hook fires, not to break the model.
- The reference oMLX prompts went through the chat template; the mlxterp call does not. Predicted-numeric agreement is the apples-to-apples signal.
