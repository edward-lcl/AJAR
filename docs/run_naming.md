# Run Naming

Use timestamped, descriptive run IDs everywhere: raw outputs, analysis exports, bundles, and committed summaries.

## Format

```text
YYYY-MM-DD_HHMM_<dataset><sample-count>_<model-family>_<backend>_<purpose>
```

Optional suffixes should describe the experimental change:

```text
YYYY-MM-DD_HHMM_gsm8k500_qwen3-4b_omlx_baseline-rerun-nocot-1024
YYYY-MM-DD_HHMM_gsm8k10_qwen3-4b_torch_mechanistic-slice
YYYY-MM-DD_HHMM_strategyqa500_qwen3-4b_omlx_baseline
```

## Field Meanings

- `YYYY-MM-DD_HHMM`: local start time of the run.
- `dataset`: benchmark or data source.
- `sample-count`: number of examples requested, such as `500` or `10`.
- `model-family`: short model group, such as `qwen3-4b`.
- `backend`: `omlx`, `torch`, or `mlx-lm`.
- `purpose`: baseline, rerun, mechanistic-slice, perturbation-slice, etc.

## Repository Convention

Raw outputs should live under:

```text
outputs/<run-id>/
```

Generated analysis artifacts should live under:

```text
analysis/<run-id>/
```

Committed, human-reviewable summaries should live under:

```text
results/runs/<run-id>/
```

Avoid informal names like `tonight`, `real`, `final`, or `new`. They are hard to interpret after multiple evals.

