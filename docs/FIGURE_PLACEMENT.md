# AJAR — Figure placement guide

Where each `figures/*.pdf` goes in the paper and what its caption should
say. All figures are vector PDFs regenerable from
`scripts/make_figures.py`.

**Design note:** all 6 figures are produced *caption-agnostic* — no
embedded title or italic description. The LaTeX captions below carry
all the prose. This avoids visual duplication when the figure is placed
next to a `\caption{...}` block in the paper.

---

## fig0_methods_pipeline.pdf

**Section:** Methods (preferably the first figure of the section, before
any per-step description)
**Width:** **full-page-width** (`\begin{figure*}` in LaTeX two-column).
The 14×10-inch landscape aspect ratio assumes full width; it will not
read at single-column width.

**Suggested LaTeX:**
```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/fig0_methods_pipeline.pdf}
  \caption{\textbf{HCDS methodology pipeline.}
  Are LLMs Just Acting Reasonable? Hidden CoT Detection in
  Qwen3-4B (Instruct vs Thinking).
  \textit{Setup:} two datasets $\times$ two models $\times$ three
  prompts.
  \textit{Pipeline:} for each $(model, prompt, question)$ cell we run
  four parallel sub-pipelines (baseline + attention probe,
  paraphrase, perturbation, mechanistic) producing a
  six-dimensional feature vector that is z-scored per model.
  \textit{Result:} HCDS contrasts the neutral prompt's distance to
  the no-CoT pole vs the CoT pole; we report bootstrap CIs and
  one-sample t-tests per (model, dataset).}
  \label{fig:methods}
\end{figure*}
```

---

## fig1_cross_dataset_hcds.pdf

**Section:** Results — headline figure, ideally at the top of the
section.
**Width:** Either single-column (`figure`) or full-width (`figure*`)
both work; the chart is 7×4.2 inches and renders cleanly at either size.

**Suggested LaTeX:**
```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/fig1_cross_dataset_hcds.pdf}
  \caption{\textbf{Cross-dataset HCDS replication.} HCDS is positive
  with $p < 0.05$ for both Qwen3-4B variants on both datasets and
  at both scales. Strongest evidence: GSM8K $n{=}500$ Instruct
  ($p = 1.98\mathrm{e}{-141}$). Error bars: 95\% bootstrap CI
  (1000 samples, seed=17). Two-sided t-test p-values labeled
  above each bar.}
  \label{fig:cross-dataset}
\end{figure}
```

**Talking points:** This is the "the result holds" figure. Lead with
GSM8K $n{=}500$ for the headline statistical power, then point to the
StrategyQA bars as cross-dataset replication. Note the 4–7$\times$
gap between Instruct and Thinking magnitudes — Thinking's reasoning
is less prompt-conditional.

---

## fig2_robustness_ablation.pdf

**Section:** Results — robustness subsection (after fig1).
**Width:** Single-column (`figure`) preferred.

**Suggested LaTeX:**
```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/fig2_robustness_ablation.pdf}
  \caption{\textbf{Feature-ablation robustness.} HCDS holds across
  seven of eight feature subsets on GSM8K $n{=}50$. The single
  exception is dropping entropy features on the Thinking model
  (HCDS $= +0.09$, $p = 0.62$, marked with $\times$), which we
  report transparently as evidence that token-level entropy is the
  load-bearing feature for the Thinking signal.}
  \label{fig:ablation}
\end{figure}
```

**Talking points:** The honest no-entropy/Thinking caveat is the most
important narrative element here. Don't bury it.

---

## fig3_length_matched.pdf

**Section:** Results — robustness subsection (with fig2) **OR**
Limitations.
**Width:** Single-column (`figure`).

**Suggested LaTeX:**
```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/fig3_length_matched.pdf}
  \caption{\textbf{Length-matched HCDS by output-length tier.}
  Instruct holds across all length tiers; Thinking is significant
  on short outputs but ambiguous on the medium tier ($p = 0.65$,
  marked $\times$), which is cap-affected. The long Thinking tier
  is not reported because the underlying CSV (\texttt{hcds\_length\_matched.csv})
  did not include a long tier for Thinking.}
  \label{fig:length}
\end{figure}
```

**Talking points:** This figure plus the no-entropy caveat together
form the "Thinking signal is more fragile than Instruct" story. Worth
acknowledging in the limitations.

---

## fig4_anchor_control.pdf

**Section:** Mechanistic analysis subsection.
**Width:** **Full-page-width** (`figure*`) — it has two panels (GSM8K
+ StrategyQA) side-by-side and needs the room.

**Suggested LaTeX:**
```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/fig4_anchor_control.pdf}
  \caption{\textbf{Anchor sensitivity.}
  $\mathrm{anchor\_drop} - \mathrm{control\_drop}$ across three
  prompt conditions on GSM8K and StrategyQA ($n{=}50$ each).
  Positive bars indicate anchors carry a privileged causal load;
  the negative Thinking + \texttt{explicit\_cot} bar on GSM8K
  ($-0.275$) is reported transparently — long reasoning chains
  distribute causal load across many steps rather than
  concentrating in a few attention anchors.}
  \label{fig:anchor}
\end{figure*}
```

**Talking points:** The negative Thinking + explicit_cot bar is a
finding, not a bug. Frame it as a feature of long reasoning chains.

---

## fig5_output_lengths.pdf

**Section:** Discussion **OR** Methods (justifying why no-CoT isn't a
clean baseline for Thinking).
**Width:** **Full-page-width** (`figure*`) — two side-by-side boxplot
panels.

**Suggested LaTeX:**
```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/fig5_output_lengths.pdf}
  \caption{\textbf{Output length distribution by prompt condition
  (GSM8K $n{=}500$).}
  Instruct + \texttt{explicit\_no\_cot} achieves a true answer-only
  regime (median 6 tokens). Thinking + \texttt{explicit\_no\_cot}
  ignores the directive and emits $\approx$575 tokens of reasoning
  anyway — itself evidence that Thinking's reasoning is not
  prompt-conditional.}
  \label{fig:output-length}
\end{figure*}
```

**Talking points:** This is doing double duty — it's both a methods
caveat (no-CoT is confounded for Thinking) and a result (the
reasoning is architecturally baked in). Either section works.

---

## Recommended figure ordering in the paper

1. **fig0** (Methods)
2. **fig1** (Results, headline)
3. **fig2** (Robustness)
4. **fig3** (Length control)
5. **fig4** (Mechanistic)
6. **fig5** (Discussion or end of Methods)

## Cross-references to weave into prose

| Figure | Forward refs from prose |
|---|---|
| fig0 | "...computed as defined in Figure~\ref{fig:methods}." |
| fig1 | "Figure~\ref{fig:cross-dataset} shows positive HCDS for both models on both datasets..." |
| fig2 | "Figure~\ref{fig:ablation} demonstrates the signal is robust to feature choice, with one exception..." |
| fig3 | "Figure~\ref{fig:length} shows the signal holds across output-length tiers for Instruct..." |
| fig4 | "Anchor interventions (Figure~\ref{fig:anchor}) confirm a localized causal structure on Instruct..." |
| fig5 | "Output-length distributions (Figure~\ref{fig:output-length}) reveal that Thinking ignores the no-CoT directive..." |

## Regeneration

All figures are produced by:
```sh
python3 scripts/make_figures.py
```
This reads from the authoritative source CSVs in `results/runs/`, so if
any number ever changes the figures pick it up on rerun. Always
regenerate before pushing changes that affect underlying data.
