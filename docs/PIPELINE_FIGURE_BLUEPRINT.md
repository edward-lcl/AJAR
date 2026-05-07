# Methods Pipeline Figure — Blueprint for Figma

This is the structural spec for the methods figure in the AJAR ICML
paper. Build it as a single full-page diagram. Box-and-arrow flow,
top-to-bottom, with annotated metrics and the HCDS computation.

---

## Layout (top → bottom)

### Panel 1 — Inputs

Two parallel input boxes side-by-side, joined by a `+` symbol:

```
┌──────────────────────────┐         ┌──────────────────────────┐
│  GSM8K                    │         │  StrategyQA               │
│  • n=50 deep-table        │    +    │  • n=50 deep-table        │
│  • n=500 wide-baseline    │         │                            │
└──────────────────────────┘         └──────────────────────────┘
```

Annotation under each: number of questions, source citation, license.

### Panel 2 — Models × prompts → 6 conditions per dataset

A 2×3 grid of conditions per dataset, hinted at via a single grid
element with hover-style labelling:

```
                Model                Prompt
            ┌─────────────┐    ┌────────────────┐
            │  Qwen3-4B   │  ×  │ explicit_cot    │
            │  Instruct   │    │ explicit_no_cot │
            │             │    │ neutral_strict  │
            ├─────────────┤    └────────────────┘
            │  Qwen3-4B   │
            │  Thinking   │  ─────►  6 conditions per question
            └─────────────┘
```

Annotation: deterministic decoding (greedy, do_sample=False); 1024-token
cap on baselines, 1536 on Thinking mech, 384 on intervention probes.

### Panel 3 — Per-question feature vector (6 features)

For each (question, model, prompt) cell, six measurements feed a
feature vector. Stack them as labelled rows in a single box:

```
┌───────────────────────────────────────────────────┐
│  Per-question feature vector (per condition)       │
│                                                     │
│  1. Accuracy                  (boolean)            │
│  2. Latency per output token  (sec/token)          │
│  3. Token entropy (mean + slope)  (bits)           │
│  4. Paraphrase consistency    (mean correct)       │
│  5. Perturbation Δ accuracy   (orig − perturbed)   │
│  6. Mechanistic Δ accuracy    (anchor suppr drop)  │
│                                                     │
│  → z-scored per-model across all 150 condition rows│
└───────────────────────────────────────────────────┘
```

Sub-arrows from each feature to the source pipeline:

- **#3 Token entropy** ← attention probe over response tokens
  (single forward pass, one probe per output position)
- **#4 Paraphrase consistency** ← `build_paraphrases.py` (oMLX-generated, 2
  paraphrases/question, numeric-preservation filter)
- **#5 Perturbation Δ** ← `build_perturbations.py`
  (distractor-irrelevant operator)
- **#6 Mechanistic Δ** ← anchor selection + targeted residual /
  attention zero-out re-generation (top-2 anchors + 1 control)

### Panel 4 — HCDS computation

Three feature vectors per question (one per prompt). Compute Euclidean
distance pairs:

```
                  f_neutral_q
                       │
      ┌────────────────┼────────────────┐
      │                                  │
      ▼                                  ▼
   D(f_neutral, f_no_cot)          D(f_neutral, f_cot)
      │                                  │
      └────────────────┐  ┌──────────────┘
                       ▼  ▼
            HCDS_q = D(N, NoCoT) − D(N, CoT)
```

Annotation:
- Z-scoring is per-model across all 150 condition rows (3 prompts ×
  50 questions)
- Distance is Euclidean
- Missing-data: per-question, drop only the features missing in any
  of the three prompts for that question
- Result: one HCDS scalar per (model, question)

### Panel 5 — Statistical testing

Two side-by-side outputs:

```
┌─────────────────────────┐    ┌──────────────────────────┐
│  Bootstrap CI            │    │  Paired t-test           │
│  (1000× resample over    │    │  H₀: HCDS = 0            │
│   the 50 questions)      │    │  vs                       │
│                          │    │  H₁: HCDS > 0            │
│  → 2.5% / 97.5% percentile│   │  one-sample on the 50    │
│                          │    │  per-question values      │
│  Reported: mean, 95% CI  │    │  Reported: t-stat, p     │
└─────────────────────────┘    └──────────────────────────┘
```

### Panel 6 — Verdict

Final box: per-(model, dataset), report the four numbers:

```
┌────────────────────────────────────────────────────────┐
│  Verdict per (model, dataset):                          │
│                                                          │
│  Mean HCDS  |  95% CI  |  p-value  | Reading            │
│  +2.25       [+1.69, +2.85]  2.2e-10   Instruct (GSM8K) │
│  +0.48       [+0.07, +0.86]  0.021     Thinking (GSM8K) │
│  ...                                                     │
└────────────────────────────────────────────────────────┘
```

---

## Visual conventions

- **Colour palette**:
  - Inputs / data → muted grey-blue (#5B7C99)
  - Conditions / experimental factors → muted teal (#3F8E8C)
  - Per-question features → warm amber (#D9A35A)
  - HCDS computation → deep blue (#2166AC)
  - Statistical tests → muted purple (#6E5797)
  - Verdict → forest green (#2C7A39)
- **Typography**: monospace for code-tagged labels (feature names,
  variable names, file paths); sans-serif for prose annotations.
- **Arrows**: solid for data flow, dashed for "depends on" (e.g.
  paraphrase row depends on the canonical question row).

---

## Robustness sub-figure (optional second figure)

If you have space for a second figure, panel-grid the robustness
checks. One row per check; one column per check's verdict (Instruct /
Thinking):

| Check | Instruct verdict | Thinking verdict |
|---|---|---|
| Full HCDS (n=50, 6 features) | +2.25 (p=2e-10) | +0.48 (p=0.02) |
| no_paraphrase | +X | +X |
| no_mech | +X | +X |
| no_entropy | +X | collapses (p=0.617) |
| no_perturb | +X | +X |
| latency_only | **+1.79 (p=2.6e-19)** | +0.41 (p=4e-04) |
| entropy_only | +1.46 (p=6e-08) | **+0.83 (p=2.5e-06)** |
| length_matched (short) | +X (p=X) | +X (p=X) |
| length_matched (medium) | +X (p=X) | ambiguous |
| length_matched (long) | +X (p=X) | +X (p=X) |
| n=500 (5 features) | (filling tomorrow) | (filling tomorrow) |
| StrategyQA n=50 | (filling tomorrow) | (filling tomorrow) |

(Fill the blanks from `team_spreadsheet/hcds_summary_all.csv` once
the runs land.)

---

## Files to look at while drawing

- Raw numbers: `results/runs/<deep-table>/team_spreadsheet/AJAR_team.xlsx`
- Existing figure for inspiration: `results/runs/<deep-table>/hcds_figure.png`
  (matplotlib, paper-ready, but plot not pipeline)
- Per-question features: `team_spreadsheet/per_question_features.csv`
- Per-question HCDS: `team_spreadsheet/per_question_hcds.csv`
- HCDS variants: `team_spreadsheet/hcds_summary_all.csv`

---

## Suggested first 30 minutes in Figma

1. Make a single page, 8.5 × 11 inch (or A4) target — paper-width.
2. Drop in 6 stacked rectangles for the 6 panels above.
3. Title each panel with the heading text from this doc.
4. Pick the colour palette from the conventions section.
5. Wire in the arrows between panels.
6. Iterate on Panel 3 (the feature vector box) since that's the
   densest and most informative panel.
