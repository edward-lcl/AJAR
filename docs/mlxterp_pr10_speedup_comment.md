Wall-clock follow-up. After the parity comment above, I wired
`InterpretableModel.generate` into my downstream project's actual
baseline-generation pipeline (it had been using oMLX's HTTP server) and
benchmarked on the same 5 GSM8K questions, same `Qwen3-4B-Instruct-MLX-8bit`,
same `max_tokens=512`.

| sample | gold | oMLX HTTP (s) | mlxterp.generate (s) | speedup | mlxterp pred |
|---|---|---|---|---|---|
| 0 | 18 | 10.44 | 9.02 | 1.16x | 18 ✓ |
| 1 | 3  |  4.20 | 8.98 | 0.47x | 3 ✓ |
| 2 | 70000 | 15.62 | 8.98 | 1.74x | 70000 ✓ |
| 3 | 540 |  6.79 | 8.93 | 0.76x | 540 ✓ |
| 4 | 20 | 11.19 | 9.08 | 1.23x | 20 ✓ |
| **mean** |  | **9.65** | **9.00** | **1.07x** |  |

Predictions agree **5/5** with the oMLX-server reference (already in the
parity comment, now reconfirmed in-process with the new API). mlxterp's
per-question time is steady at ~9.0s, which is the realistic story:
oMLX HTTP variance is what's noisy, not mlxterp.

I also exercised the new intervention-during-generation path on the
same prompts: `interventions={"layers.5": iv.add_vector(steer)}` with
`intervention_tokens="generated"` and an `N(0, 0.05)` steering vector
on layer 5.

| | mean (s) per question |
|---|---|
| plain mlxterp.generate | 9.00 |
| `+iv L5 add_vector(N(0, 0.05))` | 10.69 |

That's **+1.69s / question** of measurable per-token overhead, which is
the unambiguous "the hook fires on every generated-token forward"
signal — there's nowhere else the time can go. The intervention itself
didn't flip any of the 5 answers (this scale of steering on a residual
stream isn't enough to break a model on simple GSM8K), so I'm reporting
"hook executes" via the timing channel rather than via answer drift.

Repro:
[`scripts/mlxterp_speedup_demo.py`](https://github.com/edward-lcl/AJAR/blob/main/scripts/mlxterp_speedup_demo.py)
on the AJAR fork; full run + caveats in
[`docs/mlxterp_speedup_demo.md`](https://github.com/edward-lcl/AJAR/blob/main/docs/mlxterp_speedup_demo.md).
