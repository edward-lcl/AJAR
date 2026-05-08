# Overleaf find-and-replace patches

Apply each block below in order using Overleaf's find-and-replace
(`Ctrl/Cmd+F` → toggle replace). Pre-checked: every OLD block matches
exactly one place in the canonical `main.tex` from this session.

If a search returns 0 matches, that change has already been applied.
If a search returns 2+ matches, double-check you're in the right
section before replacing — the doc layout is the safety net here.

---

## 0. Global

### 0.1 Figure paths
Find: `prism-uploads/`
Replace: `figures/`
Replace All ✓

### 0.2 Model name
Find: `Qwen3 4B`
Replace: `Qwen3-4B`
Replace All ✓

### 0.3 Full model IDs (Setup section)
Find: `Qwen3-4B Instruct 2507`
Replace: `Qwen3-4B-Instruct-2507`
Replace All ✓

Find: `Qwen3-4B Thinking 2507`
Replace: `Qwen3-4B-Thinking-2507`
Replace All ✓

---

## 1. Title block (around line 60)

### 1.1 Running title

Find:
```latex
\icmltitlerunning{Submission and Formatting Instructions for ICML 2026}
```

Replace:
```latex
\icmltitlerunning{Detecting hidden chain-of-thought in LLMs}
```

---

## 2. Abstract

### 2.1 Soften "confirming"
Find: `confirming the asymmetric prediction that genuine implicit reasoners behave like explicit-CoT models while pattern-matchers do not.`
Replace: `consistent with the asymmetric prediction that genuine implicit reasoners would behave like explicit-CoT models while pattern-matchers would not.`

---

## 3. Introduction (around line 137)

### 3.1 Add StrategyQA + cross-dataset numbers
Find: `We evaluate HCDS on Qwen3-4B (Instruct and Thinking variants) using GSM8K, comparing each model's behavior across the three prompt conditions. HCDS is significantly positive for both models under neutral prompting (GSM8K: +2.254 for Instruct, p = 2.21e-10; +0.479 for Thinking, p = 2.06e-02), indicating that neutral behavior aligns more closely with explicit CoT than with explicit no-CoT.`

Replace: `We evaluate HCDS on Qwen3-4B (Instruct and Thinking variants) using GSM8K and StrategyQA, comparing each model's behavior across the three prompt conditions. HCDS is significantly positive for both models under neutral prompting on both datasets (GSM8K: +2.254 for Instruct, p = 2.21e-10; +0.479 for Thinking, p = 2.06e-02; StrategyQA: +1.871 for Instruct, p = 3.53e-12; +0.597 for Thinking, p = 2.15e-03), indicating that neutral behavior aligns more closely with explicit CoT than with explicit no-CoT.`

### 3.2 Anchor terminology
Find: `Anchor-perturbation interventions further reveal`
Replace: `Anchor-suppression interventions further reveal`

---

## 4. Background (around line 148)

### 4.1 Remove fragile setcounter hack
Find:
```latex
\setcounter{section}{2}
\section{Background}
```

Replace:
```latex
\section{Background}
```

---

## 5. Method (around line 185)

### 5.1 Prompt naming + appendix reference
Find: `For each model $m$ and question $q$, we evaluate three prompt conditions: \textit{explicit\_cot}, which asks the model to reason step by step; \textit{explicit\_no\_cot}, which asks for a direct answer without reasoning; and \textit{neutral}, which provides no instruction about whether reasoning should be shown.`

Replace: `For each model $m$ and question $q$, we evaluate three prompt conditions: \textit{explicit\_cot}, which asks the model to reason step by step; \textit{explicit\_no\_cot}, which asks for a direct answer without reasoning; and \textit{neutral\_strict}, a minimal task-only prompt that gives no instruction about whether reasoning should be shown. Full prompt text is provided in Appendix~\ref{app:prompts}.`

### 5.2 Intuitive HCDS framing (NEW — add a sentence)
Find:
```latex
We then define the per-question Hidden CoT Detection Score (HCDS) as
\[
```

Replace:
```latex
We then define the per-question Hidden CoT Detection Score (HCDS). Intuitively, $\mathrm{HCDS}_q > 0$ means the model's neutral-prompt behavior on question $q$ looks more like its overt-CoT behavior than its overt no-CoT behavior --- i.e., the model is behaving as if it is reasoning under the hood. Formally:
\begin{equation}
```

Then find the closing of the same equation:
Find:
```latex
&- D\!\left(f_{m,q,\mathrm{neutral}}, f_{m,q,\mathrm{cot}}\right).
\end{aligned}
\]
```

Replace:
```latex
&- D\!\left(f_{m,q,\mathrm{neutral}}, f_{m,q,\mathrm{cot}}\right).
\end{aligned}
\label{eq:hcds-q}
\end{equation}
```

### 5.3 Average HCDS equation label
Find:
```latex
\[
\overline{\mathrm{HCDS}} = \frac{1}{N}\sum_{q=1}^{N} \mathrm{HCDS}_q.
\]
```

Replace:
```latex
\begin{equation}
\overline{\mathrm{HCDS}} = \frac{1}{N}\sum_{q=1}^{N} \mathrm{HCDS}_q.
\label{eq:hcds-mean}
\end{equation}
```

---

## 6. Experimental Setup (around line 215)

### 6.1 Prompt naming gloss
Find: `We use the same three prompt conditions for both datasets: \textit{explicit\_cot}, \textit{explicit\_no\_cot}, and \textit{neutral}.`

Replace: `We use the same three prompt conditions for both datasets: \textit{explicit\_cot}, \textit{explicit\_no\_cot}, and \textit{neutral\_strict} (the names used in our codebase and figures; for brevity we refer to them as explicit CoT, explicit no-CoT, and neutral throughout the prose).`

### 6.2 H1 hypothesis (one-sided → two-sided)
Find:
```latex
H_1:\; \mathbb{E}[\mathrm{HCDS}_q] > 0,
```

Replace:
```latex
H_1:\; \mathbb{E}[\mathrm{HCDS}_q] \neq 0,
```

---

## 7. Results — Cross-Dataset Replication

### 7.1 Tighten the numbers-heavy paragraph
Find: `Figure~\ref{fig:cross-dataset} shows that the central HCDS result replicates across GSM8K and StrategyQA and strengthens further when GSM8K is scaled from $n=50$ to $n=500$. On the core GSM8K $n=50$ evaluation with all six feature families, the Qwen3 4B Instruct model achieves a mean HCDS of +2.254, with a 95\% confidence interval of [+1.687, +2.846] and a p-value of 2.21e-10. The Qwen3 4B Thinking model achieves a mean HCDS of +0.479, with a 95\% confidence interval of [+0.068, +0.859] and a p-value of 2.06e-02. On StrategyQA ($n=50$, 6 features), the corresponding values are +1.871 for Instruct and +0.597 for Thinking, with 95\% confidence intervals of [+1.482, +2.282] and [+0.242, +0.950], and p-values of 3.53e-12 and 2.15e-03, respectively. The consistent positive sign across both datasets indicates that hidden-CoT detection is not captured by answer accuracy alone, but by the joint behavioral and mechanistic profile induced by the three prompt conditions.`

Replace: `Figure~\ref{fig:cross-dataset} shows that the central HCDS result replicates across GSM8K and StrategyQA and strengthens further when GSM8K is scaled from $n=50$ to $n=500$. HCDS is significantly positive ($p < 0.05$) for both Qwen3-4B variants on both datasets and at both scales; the strongest evidence is the GSM8K $n=500$ extension, where the Instruct model reaches $\overline{\mathrm{HCDS}} = +2.375$ with $p \approx 2 \times 10^{-141}$ (Eq.~\ref{eq:hcds-mean}). The Thinking signal is smaller in magnitude but consistently positive across all four (model, dataset) combinations. The consistent positive sign indicates that hidden-CoT detection is not captured by answer accuracy alone, but by the joint behavioral and mechanistic profile induced by the three prompt conditions.`

---

## 8. Results section reorder — MANUAL STEP

⚠️ **Move the entire `\subsection{Mechanistic Anchor Analysis}` block (figure environment + paragraph) to come BEFORE `\subsection{Robustness to Feature Choice}`.** New order: 6.1 Cross-Dataset → 6.2 Mechanistic → 6.3 Robustness.

This is structural; find-and-replace cannot do it.

---

## 9. Robustness subsection

### 9.1 Add section label
Find:
```latex
\subsection{Robustness to Feature Choice}

\begin{figure}[t]
```

Replace:
```latex
\subsection{Robustness to Feature Choice}
\label{sec:robustness}

\begin{figure}[t]
```

### 9.2 Reframe entropy ablation as finding (mentor #3)
Find: `Figure~\ref{fig:ablation} demonstrates that the signal is robust to feature choice for Instruct and remains positive for Thinking across all but one ablation. The one transparent failure case is the Thinking model with entropy removed ($\mathrm{HCDS}=+0.09$, $p=0.62$), which indicates that token-level entropy is the load-bearing feature for its weaker signal rather than a dispensable auxiliary statistic. A separate length-matched check further shows that Instruct remains positive across short, medium, and long outputs, whereas Thinking is clearly positive only on short outputs and becomes ambiguous on the cap-affected medium tier. Together, these controls show that HCDS is not explained by raw verbosity alone, while also making clear that the Thinking effect is more fragile than the Instruct effect.`

Replace: `Figure~\ref{fig:ablation} demonstrates that the signal is robust to feature choice for Instruct and remains positive for Thinking across seven of eight ablations. The eighth ablation is itself informative: removing the two entropy features collapses the Thinking signal to $\mathrm{HCDS}=+0.09$ ($p=0.62$), revealing that token-level entropy is the load-bearing feature for the Thinking pathway. The Instruct pathway shows no analogous dependence --- removing entropy leaves Instruct at $\mathrm{HCDS}=+1.72$ ($p=7.4\mathrm{e}{-9}$). The two pathways are distinguishable not just by HCDS magnitude but by which feature carries the signal, which is itself evidence that the two models are doing something qualitatively different under neutral prompting. A separate length-matched check (Table~\ref{tab:length-matched} in Appendix~\ref{app:length-matched}) further shows that Instruct remains positive across short, medium, and long outputs, whereas Thinking is clearly positive only on short outputs and becomes ambiguous on the cap-affected medium tier --- confirming that HCDS is not explained by raw verbosity alone.`

---

## 10. Mechanistic Anchor Analysis paragraph

### 10.1 Acknowledge anchor-scoring methodology limitation (NEW from anchor investigation file)
Find: `Rather than undermining the hidden-CoT interpretation, this negative bar suggests that long reasoning chains in the Thinking model distribute causal dependence across many steps instead of concentrating in a few dominant anchors.`

Replace: `We see two non-mutually-exclusive interpretations of this negative bar. First, long reasoning chains in the Thinking model may distribute causal dependence across many steps instead of concentrating it in a few dominant anchors --- consistent with the Thinking model's larger no-CoT output footprint discussed in Section~\ref{sec:discussion}. Second, our anchor-scoring formula combines forward-attention, answer-attention, and activation-delta proxies, all of which trend toward late-trace ``summary'' or ``answer-formulation'' steps in long traces; suppressing these post-hoc summary steps is less damaging than suppressing earlier mid-reasoning computation that downstream tokens have not yet copied. We report the result transparently and treat the development of an anchor-scoring rule that is robust to long-trace summary bias as future work.`

---

## 11. Discussion

### 11.1 Add label for cross-references
Find:
```latex
\section{Discussion}

These findings support the central claim
```

Replace:
```latex
\section{Discussion}
\label{sec:discussion}

These findings support the central claim
```

### 11.2 Concretize "alternative explanations"
Find: `neutral-prompt behavior more often resembles overt reasoning than direct-answer generation, and this conclusion survives several obvious alternative explanations.`

Replace: `neutral-prompt behavior more often resembles overt reasoning than direct-answer generation, and this conclusion survives the alternative explanations we tested. The signal is not driven by raw verbosity (length-matched check, Appendix~\ref{app:length-matched}), is not collapsed by removing any single feature with the one transparent exception of entropy on Thinking (Section~\ref{sec:robustness}), and replicates on a structurally different commonsense-reasoning benchmark (StrategyQA).`

### 11.3 Update Discussion's figure-list order
Find: `The cross-dataset replication in Figure~\ref{fig:cross-dataset}, the feature-ablation results in Figure~\ref{fig:ablation}, and the mechanistic contrast in Figure~\ref{fig:anchor}`

Replace: `The cross-dataset replication in Figure~\ref{fig:cross-dataset}, the mechanistic contrast in Figure~\ref{fig:anchor}, and the feature-ablation results in Figure~\ref{fig:ablation}`

### 11.4 Emphasize Thinking story (mentor #2 — biggest prose change)
Find: `The model comparison is especially informative. Instruct more nearly obeys \texttt{explicit\_no\_cot} and approaches an answer-only regime, while Thinking continues to emit long reasoning traces even when explicitly told not to, with mean and median no-CoT output lengths of 574.88 and 490.5 tokens, respectively. That makes the no-CoT pole less clean for Thinking, but it is also substantively informative: the model's reasoning behavior appears more deeply integrated and less prompt-conditional. Read together with the diffuse anchor pattern in Figure~\ref{fig:anchor}, the most plausible interpretation is that Thinking relies on hidden reasoning in a more distributed form than Instruct.`

Replace: `The most striking finding is the qualitative difference between the two models' reasoning behavior. Instruct broadly complies with \texttt{explicit\_no\_cot} and approaches an answer-only regime (mean / median 24 / 6 tokens), whereas Thinking continues to emit long reasoning traces even when explicitly told not to (mean / median 574.88 / 490.5 tokens). This is not just a confound for the no-CoT control --- it is a finding in its own right: the Thinking model cannot suppress its reasoning chain on demand. Read together with the entropy-load-bearing pattern in Figure~\ref{fig:ablation} and the diffuse anchor sensitivity in Figure~\ref{fig:anchor}, the picture that emerges is that the Thinking model's internal CoT is qualitatively different from Instruct's: it is harder to turn off, it relies on distributional uncertainty (entropy) rather than localized timing or attention markers, and its causal load is spread across many reasoning steps rather than a few dominant anchors. Together, these three signatures point to hidden reasoning that is more deeply integrated and more prompt-invariant in reasoning-tuned models than in instruction-tuned ones.`

---

## 12. Limitations

### 12.1 Strengthen single-model-family limitation (mentor #4)
Find: `The current study is limited in scope, feature coverage, and evaluation breadth. It evaluates 2 models across 2 datasets, and all models come from the same family, which restricts claims about cross-architecture generality. In addition, the main pipeline uses 1 trial per question under deterministic decoding, so the reported uncertainty reflects variation across questions rather than within-prompt sampling variance. Feature coverage is also uneven in some settings: mechanistic intervention values are available for 503 out of 600 rows, and some prompt conditions admit shorter traces or reduced anchor coverage of only 4 usable examples. These constraints do not invalidate the core findings, but they do narrow the strength of the generalization claim.`

Replace: `The most important limitation is that our qualitative comparison between ``reasoning-tuned'' and ``instruction-tuned'' models rests on a single model family (Qwen3-4B Instruct and Qwen3-4B Thinking). The two checkpoints share architecture, tokenizer, and pre-training corpus and differ only in their instruction- vs reasoning-oriented post-training. This makes the within-family comparison relatively clean, but it also means our claims about how Thinking-class models reason internally cannot be cleanly separated from properties of this specific Qwen3-4B post-training recipe. Replicating the asymmetric pattern --- larger, less prompt-conditional, entropy-driven, diffuse-anchor HCDS in the reasoning-tuned variant --- on at least one additional model family (e.g., DeepSeek R1, Gemma-Reasoning, Llama-3-Reasoning) is the first follow-up we plan. In addition, the main pipeline uses 1 trial per question under deterministic decoding, so the reported uncertainty reflects variation across questions rather than within-prompt sampling variance. Feature coverage is also uneven in some settings: mechanistic intervention values are available for 503 of the 600 (model $\times$ prompt $\times$ question) cells, and the Instruct $\times$ \texttt{explicit\_no\_cot} cell yields traces too short for anchor analysis on all but 4 questions.`

### 12.2 Trim duplicate output-length numbers
Find: `Likewise, the no-CoT control is imperfect for some models, especially when the model continues to produce long reasoning traces despite answer-only instructions, as reflected in the Thinking model's mean and median no-CoT output lengths of 574.88 and 490.5 tokens, respectively.`

Replace: `Likewise, the no-CoT control is imperfect for models that continue to produce long reasoning traces despite answer-only instructions, as discussed for the Thinking model in Section~\ref{sec:discussion}.`

---

## 13. Conclusion (full rewrite)

Find: `This work addresses an important limitation in current research on large language models: the lack of reliable methods for determining whether a model is actually performing multi-step reasoning internally, even when it does not explicitly show its reasoning steps. Existing approaches often rely on written explanations, but these explanations do not always reflect the model's true reasoning process. As a result, it has remained difficult to distinguish genuine internal reasoning from surface-level pattern matching.

To address this problem, this paper introduced a framework that combines linguistic, behavioral, and mechanistic indicators to detect implicit chain-of-thought reasoning. By measuring factors such as response latency, token-level entropy, consistency across paraphrased inputs, and sensitivity to structured perturbations and interventions, the framework tests whether a model's behavior depends on intermediate reasoning steps. The Hidden CoT Detection Score provides a quantitative way to compare neutral-prompt behavior against explicit reasoning and answer-only conditions, allowing hidden reasoning to be studied without requiring the model to reveal its full chain of thought.

Overall, this work provides a more systematic and evidence-based method for studying internal reasoning in language models. This is important for improving interpretability, evaluating reasoning ability more accurately, and better understanding how modern AI systems arrive at their answers. Future work can extend this framework to broader model families, richer diagnostic benchmarks, and more diverse reasoning tasks in order to strengthen its generality and deepen our understanding of latent reasoning in large language models.`

Replace: `We introduced HCDS, a comparative detection score for hidden chain-of-thought reasoning that combines linguistic, behavioral, and mechanistic indicators rather than relying on the faithfulness of any explicit explanation. Rather than asking whether a model's visible reasoning is causally responsible for its answer, HCDS asks whether the model's neutral-prompt behavior is closer to its overt-CoT behavior or to its overt no-CoT behavior in a six-feature joint space. On Qwen3-4B Instruct and Thinking, HCDS was significantly positive on both GSM8K and StrategyQA, with the strongest evidence on the GSM8K $n=500$ extension ($p \approx 2 \times 10^{-141}$).

The most distinctive finding is the qualitative gap between the two model variants. The reasoning-tuned Thinking model cannot be commanded into an answer-only regime (it emits $\sim$575 reasoning tokens even under \texttt{explicit\_no\_cot}), depends on token-level entropy as its load-bearing HCDS feature, and shows diffuse rather than localized anchor sensitivity (Figure~\ref{fig:anchor}). The instruction-tuned Instruct model, by contrast, complies with the no-CoT directive (median 6 tokens), shows entropy-independent HCDS, and concentrates causal load on a small number of anchor steps. Read together, these three signatures point to a single conclusion: \textbf{hidden chain-of-thought in reasoning-tuned models is more deeply integrated, more prompt-invariant, and more diffuse than in instruction-tuned models.} Whether this distinction generalizes beyond the Qwen3-4B family is the most important open question we leave for future work.`

---

## 14. End-of-paper template removal

### 14.1 Replace template Accessibility/Software-Data/Acknowledgements/Impact blocks
Find:
```latex
\section*{Accessibility}

Authors are kindly asked to make their submissions as accessible as possible
for everyone including people with disabilities and sensory or neurological
differences. Tips of how to achieve this and what to pay attention to will be
provided on the conference website \url{http://icml.cc/}.

\section*{Software and Data}

If a paper is accepted, we strongly encourage the publication of software and
data with the camera-ready version of the paper whenever appropriate. This can
be done by including a URL in the camera-ready copy. However, \textbf{do not}
include URLs that reveal your institution or identity in your submission for
review. Instead, provide an anonymous URL or upload the material as
``Supplementary Material'' into the OpenReview reviewing system. Note that
reviewers are not required to look at this material when writing their review.

% Acknowledgements should only appear in the accepted version.
\section*{Acknowledgements}

\textbf{Do not} include acknowledgements in the initial version of the paper
submitted for blind review.

If a paper is accepted, the final camera-ready version can (and usually should)
include acknowledgements.  Such acknowledgements should be placed at the end of
the section, in an unnumbered section that does not count towards the paper
page limit. Typically, this will include thanks to reviewers who gave useful
comments, to colleagues who contributed to the ideas, and to funding agencies
and corporate sponsors that provided financial support.

\section*{Impact Statement}

Authors are \textbf{required} to include a statement of the potential broader
impact of their work, including its ethical aspects and future societal
consequences. This statement should be in an unnumbered section at the end of
the paper (co-located with Acknowledgements -- the two may appear in either
order, but both must be before References), and does not count toward the paper
page limit. In many cases, where the ethical impacts and expected societal
implications are those that are well established when advancing the field of
Machine Learning, substantial discussion is not required, and a simple
statement such as the following will suffice:

``This paper presents work whose goal is to advance the field of Machine
Learning. There are many potential societal consequences of our work, none
which we feel must be specifically highlighted here.''

The above statement can be used verbatim in such cases, but we encourage
authors to think about whether there is content which does warrant further
discussion, as this statement will be apparent if the paper is later flagged
for ethics review.
```

Replace:
```latex
\section*{Impact Statement}

This paper presents work whose goal is to advance the field of Machine
Learning. There are many potential societal consequences of our work, none
which we feel must be specifically highlighted here.
```

---

## 15. Appendix (NEW — paste at end, AFTER \bibliography per ICML convention)

After the `\bibliography{research}` and `\bibliographystyle{icml2026}` lines and BEFORE `\end{document}`, paste:

```latex

\appendix

\section{Prompt Templates}
\label{app:prompts}

All three prompt conditions use the same chat template (system + user
turn) and the same final-answer format ($\boxed{\text{number}}$ for
GSM8K, $\boxed{\text{yes/no}}$ for StrategyQA). The system messages are
fixed; only the user message substitutes the question text.

\paragraph{\texttt{explicit\_cot}.}
\textit{System:}
``Think through this step by step showing your reasoning process then
provide your final answer. Put the final numeric answer in
$\boxed{\,\,}$.''
\textit{User:} ``Question: $\langle q\rangle$''

\paragraph{\texttt{explicit\_no\_cot}.}
\textit{System:}
``Answer-only mode. \texttt{/no\_think}\\
Output exactly one line: $\boxed{\text{number}}$. Do not include
reasoning, explanations, equations, restatements, or units.''
\textit{User:}
``Question: $\langle q\rangle$\\
\texttt{/no\_think}\\
Respond only with $\boxed{\text{number}}$.''

\paragraph{\texttt{neutral\_strict}.}
\textit{System:} ``You are a helpful assistant.''
\textit{User:} ``Question: $\langle q\rangle$''

The \texttt{neutral\_strict} prompt is intentionally minimal --- it
contains no instruction about whether to reason and no answer-format
directive. Answer extraction for this condition falls back to
``last number in text'' when no $\boxed{}$ delimiter is produced.

\section{Decoding and Hyperparameters}
\label{app:hparams}

\begin{table}[h]
\centering
\small
\begin{tabular}{ll}
\toprule
Setting & Value \\
\midrule
Decoding & Greedy ($\mathrm{do\_sample}{=}\mathrm{False}$) \\
Random seed & 17 \\
Trials per cell & 1 \\
Backend & MLX 8-bit (\texttt{Qwen3-4B-*-MLX-8bit}) \\
Max new tokens (baseline) & 1024 \\
Max new tokens (mech baseline) & 1536 \\
Max new tokens (intervention) & 384 \\
Bootstrap resamples & 1000 \\
Bootstrap seed & 17 \\
$t$-test & one-sample, two-sided \\
\bottomrule
\end{tabular}
\caption{Decoding and statistical settings used throughout. All
experiments are deterministic; the question is the unit of statistical
variation.}
\label{tab:hparams}
\end{table}

\section{Length-Matched HCDS}
\label{app:length-matched}

We split GSM8K $n{=}50$ into output-length tertiles (using the
$\textit{explicit\_cot}$ output length as the binning variable) and
recomputed HCDS within each tier. Table~\ref{tab:length-matched} shows
that the Instruct effect is positive and significant in every tier,
while the Thinking effect weakens in the medium tier --- consistent
with our reading that medium-length Thinking outputs are most affected
by the 1024-token cap.

\begin{table}[h]
\centering
\small
\begin{tabular}{llrrrr}
\toprule
Model & Tier & $n$ & mean CoT tok & HCDS & $p$ \\
\midrule
Instruct & short  & 17 & 175.5  & $+1.890$ & $9.4\mathrm{e}{-4}$ \\
Instruct & medium & 16 & 279.8  & $+2.673$ & $1.1\mathrm{e}{-4}$ \\
Instruct & long   & 17 & 472.4  & $+2.224$ & $4.3\mathrm{e}{-4}$ \\
Thinking & short  & 17 & 716.6  & $+1.205$ & $2.1\mathrm{e}{-3}$ \\
Thinking & medium & 33 & 1013.2 & $+0.106$ & $0.65$ \\
\bottomrule
\end{tabular}
\caption{Length-matched HCDS by output-length tier on GSM8K $n{=}50$.
Tier boundaries are tertiles of explicit-CoT output length per model.
The thinking-long tier is omitted because the source CSV
(\texttt{hcds\_length\_matched.csv}) did not include it; the
\texttt{long} bin for Thinking is dominated by cap-truncated runs and
is left to future work.}
\label{tab:length-matched}
\end{table}
```

---

## 16. Figures (binary files — upload separately)

The 6 PDF/PNG figures live in `figures/` in the repo. Upload these to
your Overleaf project's `figures/` folder (or wherever §0.1 points):

- `figures/fig0_methods_pipeline.pdf`
- `figures/fig1_cross_dataset_hcds.pdf`
- `figures/fig2_robustness_ablation.pdf`
- `figures/fig3_length_matched.pdf` *(not referenced inline)*
- `figures/fig4_anchor_control.pdf`
- `figures/fig5_output_lengths.pdf` *(not referenced inline)*

The current versions use the divergent blue→teal palette
(`#02608f` Instruct / `#60988c` Thinking).

---

## 17. Final compile cycle

```sh
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

If undefined references warn on first pass, that's expected — they
clear on the second `pdflatex` after `bibtex` has built the `.bbl`.
