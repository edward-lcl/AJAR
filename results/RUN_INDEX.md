# Run Index

Human-reviewable summaries of every committed run, plus pointers to the
artifacts that landed. Large raw outputs (per-sample baseline.json, hidden
state .npz, intervention json, etc.) are gitignored under `outputs/`; only
the aggregate CSVs and SUMMARY.md docs live here.

## Completed runs

| Run ID | Date (AST) | Purpose | Scope | Outcome |
|---|---|---|---|---|
| `2026-05-03_2223_gsm8k500_qwen3-4b_omlx_baseline-v1` | 2026-05-03 22:23 | Initial baseline | GSM8K 500, 2 models, explicit CoT + neutral | Major Thinking truncation at 512 tokens. |
| `2026-05-04_0238_gsm8k500_qwen3-4b_omlx_baseline-rerun-nocot-1024` | 2026-05-04 02:38 | Baseline rerun | GSM8K 500, 2 models, explicit_cot + explicit_no_cot + neutral, 1024 token cap | 3000/3000 generations, 0 failures. Identified `neutral` prompt contamination ("Answer the question directly" + boxed directive). |
| `2026-05-04_1720_gsm8k1_qwen3-4b_torch_mechanistic-smoke` | 2026-05-04 17:20 | MI smoke | GSM8K 1, Qwen3-4B-Instruct, neutral | Verified Torch backend produces tokens.csv, step scores, anchor indices, and one intervention. |
| `2026-05-04_1856_gsm8k500_qwen3-4b_omlx_baseline-neutral-strict-1024` | 2026-05-04 18:56 | HCDS-clean wide baseline | GSM8K 500, 2 models, explicit_cot + explicit_no_cot + **neutral_strict** | 3000/3000 generations. Replaces the contaminated `neutral` rerun for HCDS purposes. |
| **`2026-05-04_2232_gsm8k50_qwen3-4b_deep-table`** | 2026-05-04 22:32 | **Task 6 deep table** | GSM8K 50, 2 models, 3 prompts, full MI + paraphrase + perturbation | **Headline result. HCDS positive for both models.** See `runs/.../SUMMARY.md`. |

## Open questions investigated post-run

- **Anchor signal validity on Thinking + explicit_cot**: the negative
  `anchor_drop − control_drop` (-0.275) is a methodology finding —
  anchor scoring is biased toward late-trace summary steps (mean
  relpos 0.75) which are recoverable under intervention. Reports at
  `results/runs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/anchor_investigation_*.md`.

## Open and planned

- Wide HCDS-clean baseline analysis: the 2026-05-04_1856 run completed but
  has not yet been passed through `analyze_baseline_run.py`. Should land
  alongside an updated `analysis/<run_id>/initial_report.md`.
- StrategyQA replication of the deep-table flow (Task 4). Same orchestrator,
  swap GSM8K loader for StrategyQA loader, swap numeric answer parser for
  yes/no. See TODO.md.
- Paired diagnostic benchmark (Task 5). Pending benchmark file.
- Second perturbation operator (dependency-broken variants). v1 ships
  distractor-irrelevant only.

## Naming convention

```
YYYY-MM-DD_HHMM_<dataset><sample-count>_<model-family>_<backend>_<purpose>
```

`<purpose>` is hyphenated and may carry config tags, e.g.
`baseline-neutral-strict-1024`, `deep-table`, `mechanistic-smoke`.

The deep-table orchestrator (`scripts/run_deep_table.sh`) constructs run
IDs automatically; pass `RUN_STAMP=...` to reuse a prior run's directory
when resuming.
