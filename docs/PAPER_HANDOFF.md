# AJAR — Paper handoff for mentor/PI review

This document summarizes the state of `main.tex` after a polish
iteration done in collaboration with Claude Code, working from
Codex Prism's review of the prior draft. It is intended for fast
context-loading before a mentor read-through.

## Submission venue

**Mechanistic Interpretability Workshop @ ICML 2026** (Seoul, July).
The paper is well-aligned to this audience: HCDS combines behavioural
detection with mechanistic anchor analysis, the negative-control
calibration speaks to interpretability validation, and the
prompt-invariance reading of the Thinking-model results connects
directly to mech interp work on internal reasoning representations
(He et al. 2026 "Reasoning Beyond Chain-of-Thought").

The paper is sized and styled for the workshop — single-model-family
scope is appropriate at workshop scale, and honest reporting of
methodology limitations (anchor-scoring bias, no_entropy ablation,
partial Instruct calibration) is the kind of work-in-progress
workshops welcome and discuss.

If the work later targets a main conference (NeurIPS, ICLR), the
priority strengthening pass is **cross-family replication**
(Llama-3.1-8B-Instruct + DeepSeek-R1-Distill-Llama-8B is the cleanest
pair), estimated at 16–24 compute-hours.

## State of the paper at handoff

- **Length:** 6 main pages + 1 page of references + 1 page of
  appendix (prompts, hyperparameters, length-matched table).
- **Figures:** 4 in main text — methods pipeline (fig0), cross-dataset
  HCDS (fig1), feature ablation (fig2), anchor sensitivity (fig4).
  Figures 3 (length-matched) and 5 (output-length) exist as PDFs in
  `figures/` but are not referenced inline; their content is summarized
  in prose + Appendix C.
- **Numbers:** every quoted statistic in the paper has been
  cross-checked against the source CSVs in `results/runs/`.
- **Reproducibility:** exact prompt text, decoding settings, and seed
  are in Appendices A–B. The HCDS computation is `compute_hcds.py` and
  the figures regenerate from `scripts/make_figures.py`.

## What was changed in the polish pass

**Tier 1 — must-fix correctness:**
- H₁ statement corrected from one-sided ($> 0$) to two-sided
  ($\neq 0$) to match the actual two-sided `ttest_1samp` call in
  `compute_hcds.py`.
- Running title replaced (was the ICML template stub).
- Acknowledgements section removed (blind-review violation).
- Impact Statement replaced with the canned ICML-approved sentence.
- Removed `\setcounter{section}{2}` hack — verified numbering preserved.
- Image paths: `prism-uploads/` → `figures/` (canonical).
- `\setcounter` and other template residue: removed.

**Tier 2 — figure cleanup:**
- Removed the green "Headline verdict" card from `fig0` — Codex Prism
  flagged it as duplicating fig1 and blurring the Method/Results
  boundary. Replaced with an interpretation card explaining what
  HCDS sign means (>0 hidden reasoning, ≈0 ambiguous, <0 no hidden
  reasoning).
- Unified `neutral` ↔ `neutral_strict` naming. Italics references in
  Methods + Setup now use the code-name form `neutral_strict` to match
  the figures and the prompt registry; prose uses the conceptual short
  form "neutral" with a one-line gloss.
- Unified `Qwen3 4B` / `Qwen3-4B` hyphenation throughout.

**Tier 3 — sentence-level polish:**
- Cross-Dataset prose tightened from 5 sentences with 8+ statistics
  down to 4 sentences with one headline number. Figure now carries the
  numeric load; the prose does interpretation.
- Added `sec:discussion`, `eq:hcds-q`, `eq:hcds-mean` labels.
- Trimmed redundant Thinking output-length numbers from Limitations
  (they were stated identically in Discussion six lines earlier).

**Reproducibility additions (new appendix):**
- A: Prompt Templates (verbatim from `run_qwen3_gsm8k_mi.py`).
- B: Decoding & Hyperparameters (one small table).
- C: Length-Matched HCDS (per-tier table, sourced from
  `hcds_length_matched.csv`).

## What was deliberately *not* changed

These were considered and skipped to preserve the first author's voice
and leave room for mentor judgment:

- **Results section ordering.** Prism suggested moving Mechanistic
  before Robustness so the mechanistic contribution doesn't feel
  third given the title's "linguistic, behavioral, and mechanistic"
  promise. Not done — first author's structural call.
- **Conclusion prose.** Slightly boilerplate but functional.
- **Related Work citations.** Likely missing: Hao et al. ("Coconut"
  / latent CoT), Goyal et al. (pause tokens), Wang & Zhou (CoT
  without prompting). Adding requires bib-library check.
- **Anchor terminology.** Three forms used:
  "anchor-suppression analysis" (abstract, background),
  "anchor-perturbation interventions" (intro), "anchor interventions"
  (results). Each reads naturally in context; unifying may be
  over-editing.
- **Figure demotion (Prism's main rec).** Prism suggested moving fig2
  (robustness) to appendix for a cleaner narrative. Disagreed: the
  no-entropy/Thinking failure in fig2 is the kind of honest reporting
  reviewers reward, and moving it to appendix would make it feel
  hidden.

## Specific questions worth a mentor read on

1. **Should we cite Coconut / pause tokens / Wang & Zhou** in §2?
   These are the closest "implicit reasoning" papers we don't currently
   cite. Likely yes, but the framing of *how* they relate to HCDS is
   the mentor's call.

2. **Is the Conclusion section pulling its weight?** It's currently
   ~3 paragraphs of fairly generic re-statement. ICML reviewers
   sometimes prefer a tighter conclusion that ends on the unexpected
   mechanistic finding (Thinking distributes causal load) rather than
   restating the framework.

3. **The Thinking explicit_no_cot edge case.** We currently report it
   in Discussion (line 302) and as a Limitation (line 308 ref). The
   abstract frames it as "evidence of deeply integrated internal CoT".
   Mentor sanity check: is the rhetorical lift from "model ignores
   instruction" → "deeply integrated CoT" a step the reviewer will
   accept, or does it need more support?

4. **The negative anchor-control bar on Thinking + explicit_cot.**
   We frame the −0.275 contrast as "long chains distribute causal
   load." Could alternatively be read as "anchor selection was wrong
   for long-chain models." Mentor sanity check on this framing.

5. **Are 4 figures the right number?** Prism's secondary suggestion
   was a 4-figure version with Mechanistic before Robustness. We left
   the existing order. Mentor's call.

## How to regenerate everything

```sh
python3 scripts/make_figures.py        # all 6 figures (PDF + PNG)
python3 scripts/results_report.py      # every headline number from CSVs
python3 scripts/sanity_check_xlsx.py   # cross-check the team xlsx
```

The numbers in the paper come from these CSV directories, in priority
order:

- `results/runs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/` — n=50 GSM8K, 6-feature
- `results/runs/2026-05-06_2121_strategyqa50_gsm8k50_qwen3-4b_deep-table/` — n=50 StrategyQA
- `results/runs/2026-05-07_0025_gsm8k500_qwen3-4b_deep-table-ext/` — n=500 GSM8K, 5-feature

## Recent commit history (most-relevant)

```
8123c52 Polish pass: tighten Cross-Dataset prose, equation labels, consistency
fe0618d Add Appendix: prompts, hyperparameters, length-matched table
9746d07 Paper Tier-1 + Tier-2 polish per Codex Prism review
98dac9a Strip embedded titles + italic captions from all 6 figures
0fb26d4 Fix RESULT label clipping HCDS title strip
017ae90 Methods pipeline finishing pass + paper-placement doc
70a4239 Methods pipeline polish: body text no longer crowds title strips
3053a55 Polish methods pipeline: section dividers + tighter spacing
9c06c17 Redesign methods pipeline figure: 3-row layout
d8fb037 Paper figures: methods pipeline + 5 statistical figures
```
