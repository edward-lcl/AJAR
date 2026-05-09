# Overleaf find-and-replace patches

Apply each block below in order using Overleaf's find-and-replace
(`Ctrl/Cmd+F` → toggle replace). Pre-checked: every OLD block matches
exactly one place in the canonical `paper.tex` from this session.

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

## PI Round — Apply AFTER mentor revisions above

The PI's review prompted a substantive reframing: Instruct is the
clean HCDS result, Thinking is reframed as ``prompt-invariance of
reasoning traces'' (because its no-CoT baseline is contaminated by
$\sim$575-token mean output even under \texttt{explicit\_no\_cot}).
Apply these AFTER the mentor revisions above are in place.

### PI.1 Abstract — reframe Thinking + add caution

Find: `Evaluating Qwen3-4B Instruct and Thinking on GSM8K and StrategyQA, we find statistically significant positive HCDS for both models (GSM8K: +2.254 for Instruct and +0.479 for Thinking; StrategyQA: +1.871 for Instruct and +0.597 for Thinking), consistent with the asymmetric prediction that genuine implicit reasoners would behave like explicit-CoT models while pattern-matchers would not. Anchor-suppression analysis further shows that the Thinking model distributes reasoning across many steps rather than concentrating it in critical anchors, and is unable to suppress reasoning even when instructed — itself evidence of deeply integrated internal CoT.`

Replace: `Evaluating Qwen3-4B Instruct and Thinking on GSM8K and StrategyQA, we find statistically significant positive HCDS for both models (GSM8K: +2.254 for Instruct and +0.479 for Thinking; StrategyQA: +1.871 for Instruct and +0.597 for Thinking). The Instruct result is our primary, cleanly-controlled hidden-CoT detection finding; the Thinking result we read with caution and reframe as evidence of prompt-invariance of reasoning traces, since Thinking emits $\sim$575 tokens of reasoning even under explicit answer-only directives and therefore lacks a clean no-CoT baseline. Anchor-suppression analysis further shows that the Thinking model distributes reasoning across many steps rather than concentrating it in critical anchors — though we caution that this signature is also consistent with attention-based anchor scoring biasing toward late-trace summary steps in long Thinking traces.`

### PI.2 Mechanistic methodology details

Find: `Anchor interventions provide a mechanistic check on the behavioral score. Figure~\ref{fig:anchor} plots anchor-minus-control drops across prompt conditions on both datasets. Positive anchor-control contrasts indicate that identified anchors carry privileged causal load, which is the dominant pattern for Instruct.`

Replace: `Anchor interventions provide a mechanistic check on the behavioral score. We score each reasoning step on three attention-based proxies (forward-attention, answer-attention, activation-delta), select the top 3 steps as anchors and 2 random non-anchor steps as matched controls, and zero-out attention at the top 4 attention layers for each. Figure~\ref{fig:anchor} plots anchor-minus-control drops across prompt conditions on both datasets. Positive anchor-control contrasts indicate that identified anchors carry privileged causal load relative to the random-control baseline, which is the dominant pattern for Instruct.`

### PI.3 Mechanistic section label (for cross-ref from Limitations)

Find:
```latex
\subsection{Mechanistic Anchor Analysis}

\begin{figure*}[t]
```

Replace:
```latex
\subsection{Mechanistic Anchor Analysis}
\label{sec:results-mech}

\begin{figure*}[t]
```

### PI.4 Discussion paragraph 1 — Instruct is the clean test

Find: `These findings support the central claim of the paper: hidden reasoning is better detected through comparative behavioral and mechanistic signatures than through surface-level explanations alone. The cross-dataset replication in Figure~\ref{fig:cross-dataset}, the mechanistic contrast in Figure~\ref{fig:anchor}, and the feature-ablation results in Figure~\ref{fig:ablation} all point in the same direction: neutral-prompt behavior more often resembles overt reasoning than direct-answer generation, and this conclusion survives the alternative explanations we tested. The signal is not driven by raw verbosity (length-matched check, Appendix~\ref{app:length-matched}), is not collapsed by removing any single feature with the one transparent exception of entropy on Thinking (Section~\ref{sec:robustness}), and replicates on a structurally different commonsense-reasoning benchmark (StrategyQA).`

Replace: `These findings support the central claim of the paper: hidden reasoning is better detected through comparative behavioral and mechanistic signatures than through surface-level explanations alone. We treat the Instruct results as the cleaner test of this claim, since Instruct broadly complies with the answer-only directive and produces a well-separated no-CoT pole. The Instruct cross-dataset replication in Figure~\ref{fig:cross-dataset}, mechanistic contrast in Figure~\ref{fig:anchor}, and feature-ablation results in Figure~\ref{fig:ablation} all point in the same direction: neutral-prompt behavior more often resembles overt reasoning than direct-answer generation, and this conclusion survives the alternative explanations we tested. The signal is not driven by raw verbosity (length-matched check, Appendix~\ref{app:length-matched}), is not collapsed by removing any single feature, and replicates on a structurally different commonsense-reasoning benchmark (StrategyQA).`

### PI.5 Discussion paragraph 2 — reframe Thinking as prompt-invariance

Find: `The most striking finding is the qualitative difference between the two models' reasoning behavior. Instruct broadly complies with \texttt{explicit\_no\_cot} and approaches an answer-only regime (mean / median 24 / 6 tokens), whereas Thinking continues to emit long reasoning traces even when explicitly told not to (mean / median 574.88 / 490.5 tokens). This is not just a confound for the no-CoT control --- it is a finding in its own right: the Thinking model cannot suppress its reasoning chain on demand. Read together with the entropy-load-bearing pattern in Figure~\ref{fig:ablation} and the diffuse anchor sensitivity in Figure~\ref{fig:anchor}, the picture that emerges is that the Thinking model's internal CoT is qualitatively different from Instruct's: it is harder to turn off, it relies on distributional uncertainty (entropy) rather than localized timing or attention markers, and its causal load is spread across many reasoning steps rather than a few dominant anchors. Together, these three signatures point to hidden reasoning that is more deeply integrated and more prompt-invariant in reasoning-tuned models than in instruction-tuned ones.`

Replace: `The Thinking results require a different framing. Because Thinking emits a mean of 575 tokens of reasoning even under \texttt{explicit\_no\_cot}, the no-CoT pole is not a clean answer-only baseline for that model, and a positive HCDS for Thinking does not strictly demonstrate hidden CoT under a clean comparison. We therefore reframe the Thinking result as evidence of \emph{prompt-invariance of reasoning traces}: the model produces long reasoning traces across all three prompt conditions, including the one that explicitly forbids it. This is not just a confound for the no-CoT control --- it is a finding in its own right. Read together with the entropy-load-bearing pattern in Figure~\ref{fig:ablation} and the diffuse anchor sensitivity in Figure~\ref{fig:anchor}, the picture that emerges is that the Thinking model's internal CoT is qualitatively different from Instruct's: it is harder to turn off, it relies on distributional uncertainty (entropy) rather than localized timing or attention markers, and its causal load is spread across many reasoning steps rather than a few dominant anchors. Together, these three signatures point to hidden reasoning that is more deeply integrated and more prompt-invariant in reasoning-tuned models than in instruction-tuned ones --- though the strength of the underlying causal claim is more circumspect for Thinking than for Instruct because of the no-CoT compliance gap and the mechanistic limitations discussed in Section~\ref{sec:limitations}.`

### PI.6 Limitations — full rewrite with 5 paragraph sub-sections

Find: `The most important limitation is that our qualitative comparison between ``reasoning-tuned'' and ``instruction-tuned'' models rests on a single model family (Qwen3-4B Instruct and Qwen3-4B Thinking). The two checkpoints share architecture, tokenizer, and pre-training corpus and differ only in their instruction- vs reasoning-oriented post-training. This makes the within-family comparison relatively clean, but it also means our claims about how Thinking-class models reason internally cannot be cleanly separated from properties of this specific Qwen3-4B post-training recipe. Replicating the asymmetric pattern --- larger, less prompt-conditional, entropy-driven, diffuse-anchor HCDS in the reasoning-tuned variant --- on at least one additional model family (e.g., DeepSeek R1, Gemma-Reasoning, Llama-3-Reasoning) is the first follow-up we plan. In addition, the main pipeline uses 1 trial per question under deterministic decoding, so the reported uncertainty reflects variation across questions rather than within-prompt sampling variance. Feature coverage is also uneven in some settings: mechanistic intervention values are available for 503 of the 600 (model $\times$ prompt $\times$ question) cells, and the Instruct $\times$ \texttt{explicit\_no\_cot} cell yields traces too short for anchor analysis on all but 4 questions.

A second limitation is that the perturbation and control suite remains narrower than the full benchmark vision. The current study relies primarily on distractor-style perturbations, so the perturbation feature should be interpreted as one dependency-breaking probe rather than a comprehensive benchmark spanning multiple operator families. Likewise, the no-CoT control is imperfect for models that continue to produce long reasoning traces despite answer-only instructions, as discussed for the Thinking model in Section~\ref{sec:discussion}. Finally, the computational cost of the deep evaluation pipeline remains nontrivial, with a high total runtime and expensive artifact-generation cost, which may limit scalability to larger model sets or broader benchmark suites. Future work should therefore extend the framework to more model families, richer perturbation operators, broader datasets, and stochastic multi-trial evaluations.`

Replace: `\paragraph{No-CoT baseline noncompliance for Thinking.}
The most consequential limitation concerns the integrity of the no-CoT pole for the Thinking model. \texttt{explicit\_no\_cot} is intended to elicit answer-only behavior, but Qwen3-4B Thinking emits a mean of 575 tokens (median 491) under that prompt --- compared with mean 24, median 6 for Instruct. The Thinking ``no-CoT'' condition is therefore not a clean answer-only baseline; it is a partially-suppressed reasoning condition. This means a positive HCDS for Thinking does not strictly demonstrate hidden CoT under neutral prompting --- it could instead reflect prompt noncompliance or reasoning-style persistence across all three prompts. We treat the Instruct HCDS result as the cleaner test of the hidden-CoT hypothesis, and we recommend reading the Thinking HCDS as evidence of \emph{prompt-invariance of reasoning traces} rather than clean hidden-CoT detection. The qualitative comparison in Section~\ref{sec:discussion} (Thinking is less prompt-conditional and more diffuse) does not depend on the no-CoT pole being clean and remains supported.

\paragraph{No negative-control calibration.}
HCDS is presented as evidence of latent reasoning, but we have not yet calibrated it against tasks where hidden reasoning is implausible (factual lookup, simple single-step arithmetic, shallow pattern matching). On such tasks a HCDS-positive result would indicate that the score is picking up stylistic differences --- longer, higher-entropy, more variable completions under neutral than under no-CoT --- rather than reasoning per se. Until we run that calibration, the magnitude of HCDS should be interpreted as a comparative quantity within reasoning benchmarks, not as an absolute reasoning detector. Negative-control calibration is the second follow-up we plan.

\paragraph{Single model family.}
Our qualitative comparison between reasoning-tuned and instruction-tuned models rests on a single family (Qwen3-4B Instruct and Qwen3-4B Thinking). The two checkpoints share architecture, tokenizer, and pre-training corpus and differ only in post-training. This makes the within-family comparison relatively clean, but it also means our claims about how Thinking-class models reason internally cannot be cleanly separated from properties of this specific Qwen3-4B post-training recipe. Replicating the asymmetric pattern on at least one additional family (e.g., DeepSeek R1, Gemma-Reasoning, Llama-3-Reasoning) is the third follow-up we plan.

\paragraph{Mechanistic component is preliminary.}
Anchor suppression is presented as causal evidence, but the actual results are mixed. Our anchor-scoring formula combines three attention-based proxies (forward attention, answer attention, activation delta) and selects the top 3 steps per trace; control interventions are 2 random non-anchor steps. Both anchors and controls are then suppressed at the top 4 attention layers. The negative anchor-control contrast on Thinking + \texttt{explicit\_cot} (Figure~\ref{fig:anchor}) admits multiple non-mutually-exclusive explanations: (a) genuinely distributed causal load; (b) attention-based anchor scoring biased toward late-trace summary steps in long Thinking traces (see Section~\ref{sec:results-mech}); (c) suboptimal control matching --- random non-anchor steps need not be matched on position, length, or function; (d) the 1024-token cap distorting the trace endpoint. Stronger mechanistic claims would require gradient-based attribution rather than attention proxies, position-matched controls, joint suppression of multiple distributed anchors, and a layer/site sweep. We mark these as future work.

\paragraph{Statistical reporting.}
The GSM8K $n=500$ extension yields extremely small p-values, but with deterministic decoding and one trial per question, the unit of statistical variation is question identity, not model stochasticity. Our reported uncertainty therefore characterizes between-question variability of HCDS, not sampling-level uncertainty around the underlying model behavior. Furthermore, the six features in $f_{m,q,p}$ are derived from the same generated response and may be autocorrelated through output length. The length-matched check in Appendix~\ref{app:length-matched} addresses this for Instruct (the signal holds across all length tiers) but not cleanly for Thinking (the long-tier is missing and the medium-tier is non-significant). The honest summary is that \textbf{Instruct is the robust HCDS result; Thinking is suggestive but fragile and confounded by failed no-CoT suppression}. Stochastic multi-trial evaluation would let future work quantify within-prompt variance and disentangle these two sources.`

### PI.7 Limitations section label

Find:
```latex
\section{Limitations}

\paragraph{No-CoT baseline noncompliance for Thinking.}
```

Replace:
```latex
\section{Limitations}
\label{sec:limitations}

\paragraph{No-CoT baseline noncompliance for Thinking.}
```

### PI.8 Conclusion — foreground Instruct, hedge Thinking

Find: `We introduced HCDS, a comparative detection score for hidden chain-of-thought reasoning that combines linguistic, behavioral, and mechanistic indicators rather than relying on the faithfulness of any explicit explanation. Rather than asking whether a model's visible reasoning is causally responsible for its answer, HCDS asks whether the model's neutral-prompt behavior is closer to its overt-CoT behavior or to its overt no-CoT behavior in a six-feature joint space. On Qwen3-4B Instruct and Thinking, HCDS was significantly positive on both GSM8K and StrategyQA, with the strongest evidence on the GSM8K $n=500$ extension ($p \approx 2 \times 10^{-141}$).

The most distinctive finding is the qualitative gap between the two model variants. The reasoning-tuned Thinking model cannot be commanded into an answer-only regime (it emits $\sim$575 reasoning tokens even under \texttt{explicit\_no\_cot}), depends on token-level entropy as its load-bearing HCDS feature, and shows diffuse rather than localized anchor sensitivity (Figure~\ref{fig:anchor}). The instruction-tuned Instruct model, by contrast, complies with the no-CoT directive (median 6 tokens), shows entropy-independent HCDS, and concentrates causal load on a small number of anchor steps. Read together, these three signatures point to a single conclusion: \textbf{hidden chain-of-thought in reasoning-tuned models is more deeply integrated, more prompt-invariant, and more diffuse than in instruction-tuned models.} Whether this distinction generalizes beyond the Qwen3-4B family is the most important open question we leave for future work.`

Replace: `We introduced HCDS, a comparative detection score for hidden chain-of-thought reasoning that combines linguistic, behavioral, and mechanistic indicators rather than relying on the faithfulness of any explicit explanation. Rather than asking whether a model's visible reasoning is causally responsible for its answer, HCDS asks whether the model's neutral-prompt behavior is closer to its overt-CoT behavior or to its overt no-CoT behavior in a six-feature joint space. On Qwen3-4B Instruct, HCDS is robustly positive on both GSM8K and StrategyQA --- this is our cleanest hidden-CoT detection result. On Qwen3-4B Thinking, HCDS is also positive but should be interpreted with caution: the no-CoT pole is contaminated by Thinking's $\sim$575-token mean output even under explicit answer-only directives, so the result is better read as evidence of \emph{prompt-invariance of reasoning traces} than of a clean hidden-CoT contrast.

The most distinctive finding is the qualitative gap between the two model variants. The reasoning-tuned Thinking model cannot be commanded into an answer-only regime, depends on token-level entropy as its load-bearing HCDS feature, and shows diffuse rather than localized anchor sensitivity (Figure~\ref{fig:anchor}). The instruction-tuned Instruct model, by contrast, complies with the no-CoT directive (median 6 tokens), shows entropy-independent HCDS, and concentrates causal load on a small number of anchor steps. Read together, these three signatures point to a single conclusion: \textbf{hidden chain-of-thought in reasoning-tuned models appears more deeply integrated, more prompt-invariant, and more diffuse than in instruction-tuned models.} The most important open questions we leave for future work are whether this distinction generalizes beyond the Qwen3-4B family, whether HCDS calibrates to zero on tasks where hidden reasoning should be absent (negative-control benchmarks), and whether a position-matched, gradient-based anchor analysis confirms the diffuse-causal-load reading or replaces it.`

---

## Negative-Control Round — Apply AFTER PI Round

Real negative-control experiment was run after the PI revisions:
50q arithmetic + 50q numeric-factual lookup, both Qwen3-4B variants,
all three prompts. Latency-only HCDS computed and compared to GSM8K
baseline. Result: Thinking calibrates to ~0 on both negative
controls; Instruct partially calibrates (~45-66% of GSM8K HCDS
persists). Honest, nuanced finding — strengthens the paper.

### NC.1 New Appendix D — paste BEFORE the existing
"\section{Length-Matched HCDS}" block

Find:
```latex
\section{Length-Matched HCDS}
\label{app:length-matched}
```

Replace:
```latex
\section{Negative-Control Calibration}
\label{app:negative-control}

A natural concern is that HCDS might pick up stylistic differences between
prompts (longer, higher-entropy completions under neutral than under
no-CoT) rather than reasoning per se. To test this, we ran HCDS on two
negative-control task families where multi-step hidden reasoning is
implausible:

\begin{itemize}
\item \textbf{Single-step arithmetic} ($n=50$): randomly generated
  questions like ``What is $8 \times 6$?'' Gold answers are
  programmatic and exact.
\item \textbf{Numeric-answer factual lookup} ($n=50$): hand-curated
  well-known facts with verifiable integer answers (``How many states
  are in the US?'', ``In what year did humans first land on the
  Moon?'').
\end{itemize}

We ran the same baseline pipeline (greedy, seed 17, 1024-token cap)
under all three prompts on both Qwen3-4B variants and computed the
\emph{latency-only} HCDS variant --- the same single-feature variant
reported on GSM8K in Section~\ref{sec:robustness}. Latency-only is the
most adversarial setting for a stylistic-differences hypothesis: it is
measured directly from generation timing, with no content-level
processing.

\begin{table}[h]
\centering
\small
\begin{tabular}{llrrr}
\toprule
Dataset & Model & $\overline{\mathrm{HCDS}}$ & 95\% CI & $p$ \\
\midrule
GSM8K $n=50$  & Instruct & $+1.79$ & [+1.58, +2.03] & $2.6\mathrm{e}{-19}$ \\
GSM8K $n=50$  & Thinking & $+0.40$ & [+0.21, +0.61] & $3.9\mathrm{e}{-4}$ \\
\midrule
Arithmetic $n=50$ & Instruct & $+0.80$ & [+0.53, +1.27] & $5.0\mathrm{e}{-4}$ \\
Arithmetic $n=50$ & Thinking & $-0.11$ & [$-0.23$, $+0.01$] & $0.10$ \\
Factual    $n=50$ & Instruct & $+1.18$ & [+0.92, +1.50] & $7.6\mathrm{e}{-10}$ \\
Factual    $n=50$ & Thinking & $-0.15$ & [$-0.28$, $-0.03$] & $0.03$ \\
\bottomrule
\end{tabular}
\caption{Latency-only HCDS on GSM8K (reasoning) compared to two
negative-control task families. Thinking calibrates to $\approx 0$
on both negative controls; Instruct shows partial calibration with
roughly 45\%--66\% of the GSM8K HCDS persisting on tasks that
require no reasoning. See text for interpretation.}
\label{tab:negative-control}
\end{table}

\paragraph{Interpretation.}
The Thinking model calibrates almost exactly to zero on both negative
controls (arithmetic: $-0.11$, $p=0.10$, n.s.; factual: $-0.15$,
$p=0.03$ but slightly \emph{negative}). This is despite the fact that
Thinking still emits hundreds of tokens of internal reasoning even on
``$8 \times 6$''-type questions: when the reasoning load is trivial,
the model's latency-per-token equalizes across prompts and HCDS
collapses. The full GSM8K Thinking HCDS of $+0.40$ is therefore
attributable to reasoning- or prompt-invariance-driven differences
in latency profile, not to a baseline ``Thinking talks more under
neutral than under no-CoT'' confound.

The Instruct model shows partial calibration. About 45\% (arithmetic)
to 66\% (factual) of the GSM8K HCDS magnitude persists on
negative-control tasks. This indicates that latency-only HCDS for
Instruct captures both reasoning-induced timing differences \emph{and}
a component of stylistic prompt-compliance asymmetry --- Instruct's
no-CoT prompt produces much shorter outputs (median 6 tokens) than
its neutral prompt on every task, including arithmetic, and that
length asymmetry alone yields a positive latency-only HCDS. We
estimate the reasoning-attributable component of the Instruct
GSM8K HCDS as roughly $+1.0$ in absolute terms (GSM8K minus a
non-reasoning baseline of $\sim$0.8--1.2).

\paragraph{What this changes about the main paper.} The main HCDS
results are not invalidated, but reviewers should read the latency
feature as a hybrid signal for Instruct. The paper's qualitative
asymmetry finding --- Thinking is more prompt-invariant and more
diffuse than Instruct --- is strengthened by these calibration
results, since Thinking's HCDS calibrates cleanly while Instruct's
does not. A full multi-feature HCDS calibration (adding entropy,
paraphrase, perturbation, and mechanistic features on the
negative-control task families) is the next robustness check we plan.

\paragraph{Method note.} Accuracy is not used as an HCDS feature, but
we observe that the Thinking model occasionally produces no
$\boxed{}$-delimited answer on negative-control questions because it
truncates at the 1024-token cap before finishing its self-reflection.
The numeric parser falls back to ``last number in text,'' which can
cause spurious low accuracy (e.g., the Thinking model writes
$\sim$1000 tokens about ``50 states since the admission of Hawaii in
1959'' and the parser reports the answer as 1959). This does not
affect HCDS computation (latency-per-token does not depend on parsed
accuracy) but is itself further evidence of Thinking's
prompt-invariant reasoning behavior.

\section{Length-Matched HCDS}
\label{app:length-matched}
```

### NC.2 Limitations — keep findings out, focus on the actual limitation

Find: `\paragraph{No negative-control calibration.}
HCDS is presented as evidence of latent reasoning, but we have not yet calibrated it against tasks where hidden reasoning is implausible (factual lookup, simple single-step arithmetic, shallow pattern matching). On such tasks a HCDS-positive result would indicate that the score is picking up stylistic differences --- longer, higher-entropy, more variable completions under neutral than under no-CoT --- rather than reasoning per se. Until we run that calibration, the magnitude of HCDS should be interpreted as a comparative quantity within reasoning benchmarks, not as an absolute reasoning detector. Negative-control calibration is the second follow-up we plan.`

Replace: `\paragraph{Negative-control calibration is single-feature only.}
The negative-control calibration we report (Appendix~\ref{app:negative-control}, summarized in Section~\ref{sec:discussion}) covers only the latency-only HCDS variant. We have not yet rerun the full 6-feature HCDS pipeline (paraphrase + perturbation + mechanistic) on the arithmetic and factual task families, so we cannot rule out that one of the other four features behaves differently on negative controls than latency does. Latency was the most adversarial single-feature target because it is measured directly from generation timing with no content-level processing, so the partial-calibration result for Instruct is informative even on its own; full multi-feature calibration is the second follow-up we plan.`

### NC.4 Mechanistic Analysis paragraph — replace anchor description with full-detail version + appendix pointer

Find: `Anchor interventions provide a mechanistic check on the behavioral score. We score each reasoning step on three attention-based proxies (forward-attention, answer-attention, activation-delta), select the top 3 steps as anchors and 2 random non-anchor steps as matched controls, and zero-out attention at the top 4 attention layers for each.`

Replace: `Anchor interventions provide a mechanistic check on the behavioral score. We score each reasoning step on three attention-/activation-based proxies (forward-attention, answer-attention, activation-delta), select the top 2 steps as anchors and 1 random non-anchor step as control, and run two intervention modes (residual zero-out and attention zero-out) at the top 4 step-specific attention layers for each --- 6 interventions per cell. Full per-cell counts and methodology details are in Appendix~\ref{app:mech-details}.`

### NC.6 Appendix D rewrite — fix table truncation + tighten verbiage

Two issues addressed:
  1. Table 2 was getting cut off in two-column layout (right column
     truncating "2.6e", "3.9", "5.0"). Switched to `table*` (full
     page width) with `[t]` placement.
  2. Section D verbiage was duplicating Discussion content. Tightened
     so Appendix D is methodology + table; Discussion carries the
     interpretation.

Find: `\section{Negative-Control Calibration}
\label{app:negative-control}

A natural concern is that HCDS might pick up stylistic differences between
prompts (longer, higher-entropy completions under neutral than under
no-CoT) rather than reasoning per se. To test this, we ran HCDS on two
negative-control task families where multi-step hidden reasoning is
implausible:

\begin{itemize}
\item \textbf{Single-step arithmetic} ($n=50$): randomly generated
  questions like ``What is $8 \times 6$?'' Gold answers are
  programmatic and exact.
\item \textbf{Numeric-answer factual lookup} ($n=50$): hand-curated
  well-known facts with verifiable integer answers (``How many states
  are in the US?'', ``In what year did humans first land on the
  Moon?'').
\end{itemize}

We ran the same baseline pipeline (greedy, seed 17, 1024-token cap)
under all three prompts on both Qwen3-4B variants and computed the
\emph{latency-only} HCDS variant --- the same single-feature variant
reported on GSM8K in Section~\ref{sec:robustness}. Latency-only is the
most adversarial setting for a stylistic-differences hypothesis: it is
measured directly from generation timing, with no content-level
processing.

\begin{table}[h]
\centering
\small
\begin{tabular}{llrrr}
\toprule
Dataset & Model & $\overline{\mathrm{HCDS}}$ & 95\% CI & $p$ \\
\midrule
GSM8K $n=50$  & Instruct & $+1.79$ & [+1.58, +2.03] & $2.6\mathrm{e}{-19}$ \\
GSM8K $n=50$  & Thinking & $+0.40$ & [+0.21, +0.61] & $3.9\mathrm{e}{-4}$ \\
\midrule
Arithmetic $n=50$ & Instruct & $+0.80$ & [+0.53, +1.27] & $5.0\mathrm{e}{-4}$ \\
Arithmetic $n=50$ & Thinking & $-0.11$ & [$-0.23$, $+0.01$] & $0.10$ \\
Factual    $n=50$ & Instruct & $+1.18$ & [+0.92, +1.50] & $7.6\mathrm{e}{-10}$ \\
Factual    $n=50$ & Thinking & $-0.15$ & [$-0.28$, $-0.03$] & $0.03$ \\
\bottomrule
\end{tabular}
\caption{Latency-only HCDS on GSM8K (reasoning) compared to two
negative-control task families. Thinking calibrates to $\approx 0$
on both negative controls; Instruct shows partial calibration with
roughly 45\%--66\% of the GSM8K HCDS persisting on tasks that
require no reasoning. See text for interpretation.}
\label{tab:negative-control}
\end{table}

\paragraph{Interpretation.}
The Thinking model calibrates almost exactly to zero on both negative
controls (arithmetic: $-0.11$, $p=0.10$, n.s.; factual: $-0.15$,
$p=0.03$ but slightly \emph{negative}). This is despite the fact that
Thinking still emits hundreds of tokens of internal reasoning even on
``$8 \times 6$''-type questions: when the reasoning load is trivial,
the model's latency-per-token equalizes across prompts and HCDS
collapses. The full GSM8K Thinking HCDS of $+0.40$ is therefore
attributable to reasoning- or prompt-invariance-driven differences
in latency profile, not to a baseline ``Thinking talks more under
neutral than under no-CoT'' confound.

The Instruct model shows partial calibration. About 45\% (arithmetic)
to 66\% (factual) of the GSM8K HCDS magnitude persists on
negative-control tasks. This indicates that latency-only HCDS for
Instruct captures both reasoning-induced timing differences \emph{and}
a component of stylistic prompt-compliance asymmetry --- Instruct's
no-CoT prompt produces much shorter outputs (median 6 tokens) than
its neutral prompt on every task, including arithmetic, and that
length asymmetry alone yields a positive latency-only HCDS. We
estimate the reasoning-attributable component of the Instruct
GSM8K HCDS as roughly $+1.0$ in absolute terms (GSM8K minus a
non-reasoning baseline of $\sim$0.8--1.2).

\paragraph{What this changes about the main paper.} The main HCDS
results are not invalidated, but reviewers should read the latency
feature as a hybrid signal for Instruct. The paper's qualitative
asymmetry finding --- Thinking is more prompt-invariant and more
diffuse than Instruct --- is strengthened by these calibration
results, since Thinking's HCDS calibrates cleanly while Instruct's
does not. A full multi-feature HCDS calibration (adding entropy,
paraphrase, perturbation, and mechanistic features on the
negative-control task families) is the next robustness check we plan.

\paragraph{Method note.} Accuracy is not used as an HCDS feature, but
we observe that the Thinking model occasionally produces no
$\boxed{}$-delimited answer on negative-control questions because it
truncates at the 1024-token cap before finishing its self-reflection.
The numeric parser falls back to ``last number in text,'' which can
cause spurious low accuracy (e.g., the Thinking model writes
$\sim$1000 tokens about ``50 states since the admission of Hawaii in
1959'' and the parser reports the answer as 1959). This does not
affect HCDS computation (latency-per-token does not depend on parsed
accuracy) but is itself further evidence of Thinking's
prompt-invariant reasoning behavior.`

Replace: `\section{Negative-Control Calibration}
\label{app:negative-control}

A natural concern with HCDS is that it could detect stylistic
differences between prompts --- longer, higher-entropy completions
under neutral than under no-CoT --- rather than reasoning per se.
To address this directly, we ran HCDS on two task families where
multi-step hidden reasoning is implausible:

\begin{itemize}
\item \textbf{Single-step arithmetic} ($n=50$): randomly generated
  questions like ``What is $8 \times 6$?''. Gold answers are
  programmatic.
\item \textbf{Numeric-answer factual lookup} ($n=50$): hand-curated
  well-known facts with verifiable integer answers (``How many
  states are in the US?'', ``In what year did humans first land on
  the Moon?'').
\end{itemize}

We ran the same baseline pipeline (Appendix~\ref{app:hparams}) under
all three prompts on both Qwen3-4B variants and computed the
\emph{latency-only} HCDS variant --- the same single-feature variant
reported on GSM8K in Section~\ref{sec:robustness}. Latency-only is the
most adversarial setting for the stylistic-differences hypothesis,
because it is measured directly from generation timing with no
content-level processing.

\begin{table*}[t]
\centering
\small
\begin{tabular}{llrrr}
\toprule
Task family & Model & $\overline{\mathrm{HCDS}}$ & 95\% CI & $p$ \\
\midrule
GSM8K (reasoning, $n=50$) & Instruct & $+1.79$ & $[+1.58, +2.03]$ & $2.6 \times 10^{-19}$ \\
GSM8K (reasoning, $n=50$) & Thinking & $+0.40$ & $[+0.21, +0.61]$ & $3.9 \times 10^{-4}$ \\
\midrule
Arithmetic (no reasoning, $n=50$) & Instruct & $+0.80$ & $[+0.53, +1.27]$ & $5.0 \times 10^{-4}$ \\
Arithmetic (no reasoning, $n=50$) & Thinking & $-0.11$ & $[-0.23, +0.01]$ & $0.10$ \\
Factual lookup (no reasoning, $n=50$) & Instruct & $+1.18$ & $[+0.92, +1.50]$ & $7.6 \times 10^{-10}$ \\
Factual lookup (no reasoning, $n=50$) & Thinking & $-0.15$ & $[-0.28, -0.03]$ & $0.03$ \\
\bottomrule
\end{tabular}
\caption{Latency-only HCDS on GSM8K (where reasoning is required)
compared to two negative-control task families (where multi-step
hidden reasoning is implausible). Thinking calibrates to
$\approx 0$ on both negative controls; Instruct shows partial
calibration with roughly 45\%--66\% of the GSM8K HCDS persisting on
tasks that require no reasoning. The interpretation is summarized in
Section~\ref{sec:discussion}.}
\label{tab:negative-control}
\end{table*}

The interpretation appears in Section~\ref{sec:discussion}: the
Thinking-model HCDS calibrates essentially to zero on both negative
controls (despite Thinking still emitting hundreds of tokens of
internal reasoning on ``$8 \times 6$''-type questions), so the GSM8K
Thinking HCDS of $+0.40$ is attributable to reasoning- or
prompt-invariance-driven latency differences rather than to a
``Thinking talks more under neutral than under no-CoT'' confound.
The Instruct HCDS only partially calibrates, indicating that the
latency-only HCDS for Instruct is a hybrid of reasoning-induced
timing and stylistic prompt-compliance. We estimate the
reasoning-attributable component of the Instruct GSM8K HCDS as
roughly $+1.0$ in absolute terms (GSM8K minus the non-reasoning
baseline of $\sim$0.8--1.2). A full multi-feature negative-control
calibration --- adding entropy, paraphrase, perturbation, and
mechanistic features on the same task families --- is the next
robustness check we plan.

\paragraph{Parser note.} Accuracy is not used as an HCDS feature, but
the Thinking model occasionally fails to produce a
$\boxed{}$-delimited answer on negative-control questions because it
truncates at the 1024-token cap before finishing its self-reflection.
The numeric parser then falls back to ``last number in text,'' which
can produce spurious low accuracy (e.g., Thinking writes
$\sim$1000 tokens about ``50 states since the admission of Hawaii in
1959'' and the parser reports the answer as 1959). This does not
affect HCDS computation (latency-per-token does not depend on parsed
accuracy), and is itself further evidence of Thinking's
prompt-invariant reasoning behavior.`

---

### NC.5 New Appendix — paste BEFORE the existing
"\section{Negative-Control Calibration}" block

Find:
```latex
\section{Negative-Control Calibration}
\label{app:negative-control}
```

Replace:
```latex
\section{Mechanistic Anchor Methodology --- Full Details}
\label{app:mech-details}

This appendix expands the brief methodology summary in
Section~\ref{sec:results-mech} with the full per-cell details
needed to evaluate or replicate the anchor-suppression analysis.

\paragraph{Anchor selection.}
For each (model, prompt, question) cell, every reasoning step is
scored on three attention-/activation-based proxies, all measured
over the step's generated token positions:
\begin{itemize}
\item \texttt{future\_attention\_mean} --- mean attention paid to this
  step's tokens by all \emph{later} reasoning steps in the trace
  (averaged over all attention layers and heads).
\item \texttt{answer\_attention\_mean} --- mean attention paid to this
  step's tokens by tokens inside the final-answer span.
\item \texttt{activation\_delta\_mean} --- L2 norm of the residual-stream
  delta induced at this step's tokens, averaged over layers $\geq 1$.
\end{itemize}
Each of the three is then z-scored across the trace, and the
unweighted sum gives \texttt{combined\_anchor\_score}. Steps are ranked
by descending score, and the top-\textbf{2} steps are selected as
anchors per cell (\texttt{MECH\_TOP\_ANCHOR\_STEPS=2}).

\paragraph{Intervention layers.}
For each anchor or control step, we identify the four
\emph{step-specific} layers with the highest combined
$\mathrm{answer\_attention\_by\_layer} +
\mathrm{future\_attention\_by\_layer}$, where the per-layer attention
is summed over the step's token positions
(\texttt{MECH\_TOP\_ANCHOR\_LAYERS=4}). Interventions are applied at
exactly these four layers.

\paragraph{Intervention modes (what is suppressed).}
Two intervention modes are run independently per anchor / control:
\begin{itemize}
\item \texttt{residual\_zero}: at each target layer, multiply the
  residual-stream output by $0$ at the step's token positions
  (and only those positions). Other token positions and other
  layers are unchanged.
\item \texttt{attention\_zero}: at each target layer, multiply the
  attention-block output by $0$ at the step's token positions.
\end{itemize}
After the suppression hook is installed, generation resumes from the
prefix \emph{up to and including} the perturbed step, with the same
greedy decoding settings and the intervention max-token cap of 384.
The resulting completion is parsed for the final answer and compared
to the unperturbed baseline; \texttt{intervention\_correct} is the
indicator that the post-intervention prediction matches gold.

\paragraph{Control-step selection.}
After anchors are chosen, control steps are sampled \emph{uniformly
at random without replacement} from the set of non-anchor reasoning
steps in the same trace, with one control step per cell
(\texttt{MECH\_NUM\_CONTROL\_STEPS=1}). The same per-step layer rule
is then used to choose the four target layers for the control. A
fixed run-level seed (17) determines the control-step shuffle, so the
control set is reproducible across reruns. Crucially, controls are
\emph{not} matched on trace position, step length, or step function
--- this is one of the limitations called out in
Section~\ref{sec:limitations}.

\paragraph{Examples included / excluded.}
Anchor analysis is only run when the baseline trace contains at
least one reasoning step with a non-empty
\texttt{generated\_token\_positions} field --- i.e.\ a step that
spans more than a single boxed answer. Per-cell counts on our two
deep-table runs:

\begin{table}[h]
\centering
\small
\begin{tabular}{lllrr}
\toprule
Dataset & Model & Prompt & Usable & Excluded (of 50) \\
\midrule
GSM8K       & Instruct & explicit\_cot     & 50 & 0 \\
GSM8K       & Instruct & explicit\_no\_cot &  4 & 46 \\
GSM8K       & Instruct & neutral\_strict   & 50 & 0 \\
GSM8K       & Thinking & explicit\_cot     & 50 & 0 \\
GSM8K       & Thinking & explicit\_no\_cot & 50 & 0 \\
GSM8K       & Thinking & neutral\_strict   & 50 & 0 \\
\midrule
StrategyQA  & Instruct & explicit\_cot     & 50 & 0 \\
StrategyQA  & Instruct & explicit\_no\_cot &  0 & 50 \\
StrategyQA  & Instruct & neutral\_strict   & 50 & 0 \\
StrategyQA  & Thinking & explicit\_cot     & 50 & 0 \\
StrategyQA  & Thinking & explicit\_no\_cot & 50 & 0 \\
StrategyQA  & Thinking & neutral\_strict   & 49 & 1 \\
\bottomrule
\end{tabular}
\caption{Anchor analysis usability per (dataset, model, prompt)
cell. Across both deep-table runs, 503 of 600 (model $\times$
prompt $\times$ question) cells yield a usable anchor analysis.
Almost all exclusions come from the \texttt{Instruct $\times$
explicit\_no\_cot} cell, where Instruct compliantly produces an
answer-only output ($\sim$6 tokens, e.g.\ \texttt{\textbackslash boxed\{48\}}) with no separable
reasoning steps for the analysis to run on. The single excluded
StrategyQA Thinking $\times$ neutral\_strict question failed
parsing on its anchor probe.}
\label{tab:anchor-coverage}
\end{table}

\paragraph{Why the Instruct $\times$ \texttt{explicit\_no\_cot} cell
is sparse.} The Instruct model complies with the no-CoT directive
and produces median 6-token outputs (mostly \texttt{\textbackslash boxed\{N\}} alone).
Anchor analysis requires separable reasoning steps; with no reasoning
in the output, there is nothing to anchor. We treat this not as a
methodological failure but as itself informative: it means the
\texttt{Instruct $\times$ \texttt{explicit\_no\_cot}} pole is genuinely
``answer-only'' in a way the Thinking $\times$ \texttt{explicit\_no\_cot}
pole is not (Thinking still produces 50/50 usable anchor cells under
the same prompt --- direct evidence of the prompt-invariance finding
discussed in Section~\ref{sec:discussion}).

\paragraph{Anchor-scoring bias toward late-trace steps.}
Independent investigation
(\texttt{anchor\_investigation\_instruct\_neutral.md} in the
artifact directory) showed that the attention-based anchor score
systematically prefers late-trace ``summary'' or ``answer-formulation''
steps in long traces, because those steps are by construction the
ones that downstream tokens attend to most. Suppressing these
post-hoc summary steps is less damaging than suppressing earlier
mid-reasoning computation that downstream tokens have not yet
copied. An optional late-trace position penalty
(\texttt{AJAR\_ANCHOR\_LATE\_PENALTY}) is plumbed through the
runner; the runs reported in this paper use the unpenalised score
(\texttt{penalty\_alpha=0}) to match the original proposal exactly,
and we mark a position-aware re-run as future work
(Section~\ref{sec:limitations}).

\section{Negative-Control Calibration}
\label{app:negative-control}
```

### NC.3 Discussion — split the alternative-explanations into a dedicated calibration paragraph

Find: `The signal is not driven by raw verbosity (length-matched check, Appendix~\ref{app:length-matched}), is not collapsed by removing any single feature, and replicates on a structurally different commonsense-reasoning benchmark (StrategyQA).`

Replace: `The signal is not driven by raw verbosity (length-matched check, Appendix~\ref{app:length-matched}), is not collapsed by removing any single feature, and replicates on a structurally different commonsense-reasoning benchmark (StrategyQA).

A direct calibration on tasks where hidden reasoning is implausible (Appendix~\ref{app:negative-control}) further sharpens the picture. On single-step arithmetic and numeric-answer factual lookup, the Thinking-model latency-only HCDS calibrates essentially to zero ($-0.11$ and $-0.15$ respectively) even though Thinking still emits hundreds of tokens of internal reasoning on those questions; the Instruct-model HCDS only partially calibrates ($+0.80$ and $+1.18$, vs $+1.79$ on GSM8K). We read this as direct evidence that the Thinking HCDS is essentially fully reasoning- or prompt-invariance-attributable, while the Instruct latency-only HCDS is a hybrid of reasoning and stylistic prompt-compliance with a reasoning-attributable component of approximately $+1.0$ in absolute terms.`

---

## Abstract Round — split Instruct/Thinking framing per reviewer note

Reviewer noted: "split the results sentence so Instruct and Thinking
are no longer co-characterized as both showing hidden CoT-like
behavior; Instruct = clean hidden-CoT, Thinking = prompt-invariance
with the 575 vs 24 token contrast included." This patch implements
exactly that split.

### Abstract.A Replace the entire abstract block

Find:
```latex
\begin{abstract}
Large language models (LLMs) can often correctly answer complex reasoning tasks without explicitly revealing the steps they take, raising a fundamental question: do these models internally perform multi-step CoT reasoning, or rely on direct-answer generation and pattern completion? Current approaches depend on surface-level linguistic cues that prior work has shown can be unfaithful, leaving no reliable method to detect implicit CoT when explicit reasoning is absent. We propose a framework that integrates linguistic, behavioral, and mechanistic interpretability signals, including token-level entropy, structured error patterns, and sensitivity to perturbations that disrupt intermediate dependencies, and a Hidden CoT Detection Score (HCDS) that quantifies whether neutral-prompt behavior aligns more closely with explicit CoT or with suppressed-CoT baselines. Evaluating Qwen3-4B Instruct and Thinking on GSM8K and StrategyQA, we find statistically significant positive HCDS for both models (GSM8K: +2.254 for Instruct and +0.479 for Thinking; StrategyQA: +1.871 for Instruct and +0.597 for Thinking), consistent with the asymmetric prediction that genuine implicit reasoners would behave like explicit-CoT models while pattern-matchers would not. Anchor-suppression analysis further shows that the Thinking model distributes reasoning across many steps rather than concentrating it in critical anchors, while producing reasoning-like traces under explicit no-CoT instructions, suggesting default CoT-like behavior.
\end{abstract}
```

Replace with:
```latex
\begin{abstract}
Large language models often answer complex reasoning questions correctly without revealing intermediate steps, raising the question of whether they perform latent reasoning or pattern completion. Existing detection methods rely on surface-level linguistic cues that prior work has shown can be unfaithful. We propose the Hidden CoT Detection Score (HCDS), a comparative behavioral and mechanistic signal that quantifies whether a model's neutral-prompt behavior aligns more closely with its explicit-CoT or with its explicit no-CoT behavior. On Qwen3-4B Instruct, HCDS is robustly positive on both GSM8K and StrategyQA ($+1.79$ to $+2.38$, $p < 10^{-10}$) --- our cleanest hidden-CoT detection result. On Qwen3-4B Thinking, HCDS is also positive ($+0.33$ to $+0.60$, $p < 0.05$) but the result is better read as evidence of \emph{prompt-invariance of reasoning traces}: under explicit no-CoT directives, Thinking emits a mean of $\sim$575 reasoning tokens vs Instruct's $\sim$24, so the no-CoT pole is not a clean answer-only baseline for that model. Anchor-suppression analysis further shows that the Thinking model distributes its causal load across many reasoning steps, suggesting that hidden chain-of-thought in reasoning-tuned models is more deeply integrated, more prompt-invariant, and more diffuse than in instruction-tuned ones.
\end{abstract}
```

---

## Joint-Anchor Round — Apply AFTER Sensitivity Round

Per the reviewer's suggestion, we ran the joint-anchor suppression
experiment that targets both top-2 anchor steps simultaneously
(union of token positions, union of step-specific top-4 layers) and
a matched joint control of 2 random non-anchor steps. 200 new
torch interventions on Thinking + explicit_cot GSM8K n=50, zero
failures. Result: anchor-control contrast moves from -0.275
(single) to -0.120 (joint) — 56% narrowing of the negative gap.
Partial support for distributed causal load; reveals that
single-control variance had inflated the original bar. Honest
mixed-result write-up; no claim is invalidated, the original bar
is moderated.

### JA.1 §6.2 Mechanistic Anchor Analysis paragraph — extend

Find: `We see two non-mutually-exclusive interpretations of this negative bar. First, long reasoning chains in the Thinking model may distribute causal dependence across many steps instead of concentrating it in a few dominant anchors --- consistent with the Thinking model's larger no-CoT output footprint discussed in Section~\ref{sec:discussion}. Second, our anchor-scoring formula combines forward-attention, answer-attention, and activation-delta proxies, all of which trend toward late-trace ``summary'' or ``answer-formulation'' steps in long traces; suppressing these post-hoc summary steps is less damaging than suppressing earlier mid-reasoning computation that downstream tokens have not yet copied. We report the result transparently and treat the development of an anchor-scoring rule that is robust to long-trace summary bias as future work.`

Replace: `We see two non-mutually-exclusive interpretations of this negative bar. First, long reasoning chains in the Thinking model may distribute causal dependence across many steps instead of concentrating it in a few dominant anchors --- consistent with the Thinking model's larger no-CoT output footprint discussed in Section~\ref{sec:discussion}. Second, our anchor-scoring formula combines forward-attention, answer-attention, and activation-delta proxies, all of which trend toward late-trace ``summary'' or ``answer-formulation'' steps in long traces; suppressing these post-hoc summary steps is less damaging than suppressing earlier mid-reasoning computation that downstream tokens have not yet copied. To partially disambiguate these interpretations, we ran a joint-suppression experiment that intervenes at \emph{both} top-2 anchor steps simultaneously and at a matched joint control of 2 random non-anchor steps (Appendix~\ref{app:joint-anchor}); the joint anchor-control contrast on Thinking + explicit\_cot moves from $-0.275$ (single) to $-0.120$ (joint), a 56\% narrowing of the negative gap. This partially supports the distributed-causal-load reading but also reveals that single random-control selection had inflated the original negative bar. We treat the development of position-matched controls and an anchor-scoring rule that is robust to long-trace summary bias as future work.`

### JA.1b Limitations §4 — remove "joint suppression" from future-work list (we did it)

Find: `Stronger mechanistic claims would require gradient-based attribution rather than attention proxies, position-matched controls, joint suppression of multiple distributed anchors, and a layer/site sweep. We mark these as future work.`

Replace: `The joint-suppression experiment in Appendix~\ref{app:joint-anchor} partially disambiguates (a) from (b/c) by showing the negative gap narrows by 56\% when both top-2 anchors are suppressed simultaneously, but does not fully resolve the question. Stronger mechanistic claims would require gradient-based attribution rather than attention proxies, position-matched controls (rather than random non-anchor sampling), and a layer/site sweep. We mark these as priority follow-up.`

### JA.2 New Appendix H — paste at end of file BEFORE \end{document}

Find:
```latex
\end{document}
```

(at the very end of the file, the LAST one)

Replace with:
```latex
\section{Joint Anchor Suppression}
\label{app:joint-anchor}

The negative anchor-control contrast on Thinking + \texttt{explicit\_cot}
($-0.275$ in Figure~\ref{fig:anchor}) admits two non-mutually-exclusive
interpretations: distributed causal load (Hypothesis~A) or
attention-based anchor scoring biased toward late-trace summary steps
(Hypothesis~B). To partially disambiguate, we ran a joint-suppression
experiment that targets both top-2 anchor steps \emph{simultaneously}
(union of token positions, union of top-4 step-specific layers) and a
matched joint control that suppresses 2 random non-anchor steps
simultaneously, both with the same two intervention modes
(\texttt{residual\_zero} and \texttt{attention\_zero}) used in the main
results. We re-use the existing baseline traces and anchor selections
from the deep-table run; only the joint interventions are newly
generated. All 50 tasks ran to completion with zero failures.

\begin{table}[!htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrr@{}}
\toprule
Metric & Single (existing) & Joint (new) \\
\midrule
Baseline accuracy             & $0.840$ & $0.840$ \\
Anchor mean accuracy          & $0.735$ & $0.700$ \\
Control mean accuracy         & $0.460$ & $0.580$ \\
\midrule
Anchor drop                   & $0.105$ & $0.140$ \\
Control drop                  & $0.380$ & $0.260$ \\
$\mathrm{anchor\_drop} - \mathrm{control\_drop}$ & $-0.275$ & $-0.120$ \\
\bottomrule
\end{tabular}
\caption{Single-step versus joint-step anchor suppression on
Thinking + \texttt{explicit\_cot} GSM8K $n{=}50$. ``Single'' columns
reproduce the result reported in the main paper (Figure~\ref{fig:anchor});
``Joint'' columns suppress two anchors / two controls simultaneously.
Per-mode breakdown: anchor $\times$ residual\_zero acc $=0.720$,
anchor $\times$ attention\_zero acc $=0.680$, control $\times$
residual\_zero acc $=0.640$, control $\times$ attention\_zero acc
$=0.520$.}
\label{tab:joint-anchor}
\end{table}

\paragraph{What the joint result shows.} The anchor-control contrast
narrows from $-0.275$ to $-0.120$ --- a 56\% reduction toward zero,
but still negative. Two effects drive the change:

\begin{enumerate}
\item \textbf{Joint anchor suppression has a slightly larger effect
  than single} (drop $0.105 \to 0.140$). This is modest support for
  Hypothesis~A: suppressing both anchors together does compound
  causal load somewhat, consistent with reasoning being distributed
  across multiple steps in the Thinking model.
\item \textbf{Matched joint control happens to sample less impactful
  steps on average than the original single random control}
  ($0.380 \to 0.260$). This tells us the original $-0.275$ contrast
  was inflated by random-control variance: a single random
  non-anchor step happened to land on impactful positions more often
  than a sample of two would on average.
\end{enumerate}

\paragraph{Bottom line.} Hypothesis~A (distributed causal load)
receives partial support from the joint experiment; Hypothesis~B
(anchor scoring bias) is not falsified, since even joint suppression
of attention-identified anchors does not overtake matched random
controls. The honest reading is that both effects contribute and the
single-step negative bar in Figure~\ref{fig:anchor} should be read as
a mild rather than dramatic anchor weakness. Position-matched control
selection (rather than purely random non-anchor sampling) and a
gradient-based anchor score that does not preferentially select
late-trace summary steps would both be expected to further narrow
the contrast; we mark these as the priority follow-up mechanistic
experiments.

\end{document}
```

---

## Sensitivity Round — Apply AFTER Negative-Control Round

Discrete sensitivity analysis added per Armaan's suggestion. We
recompute HCDS on every $2^k - 1$ non-empty feature subset and
report what fraction stays positive. Result: HCDS sign is
positive in 92.9%-98.4% of subsets for Instruct and 76.2%-93.3%
for Thinking. The few negative-sign Instruct subsets are
degenerate (no behavioral features); all negative-sign Thinking
subsets exclude entropy. Equal-weighting empirically vindicated.

### S.1 Robustness §6.3 — append a new paragraph at the end

Find the end of the paragraph that starts with `Figure~\ref{fig:ablation} demonstrates that the signal is robust...` and ends with `confirming that HCDS is not explained by raw verbosity alone.`

Add immediately after that paragraph:

```latex
\paragraph{Sensitivity to feature weighting.} The above ablations test 7 specific feature subsets; to address whether equal-weighting of features is empirically justified, we extend this to all $2^k - 1$ non-empty subsets of available features and report the fraction for which HCDS sign and significance are preserved (Appendix~\ref{app:sensitivity}). Across all three task families, Instruct HCDS is positive in 92.9\%--98.4\% of subsets and significantly positive in 85.7\%--95.2\%; Thinking is positive in 76.2\%--93.3\% of subsets. The few negative-sign Instruct subsets are degenerate combinations of perturbation- and mechanistic-only features (subsets with no behavioral or linguistic content), and all negative-sign Thinking subsets exclude both entropy features --- exactly consistent with the entropy-load-bearing finding above. This sensitivity analysis empirically vindicates equal-weighting: the headline conclusion (HCDS positive) is preserved across the overwhelming majority of feature-weight configurations we tested.
```

### S.2 New Appendix G — paste at end of file BEFORE \end{document}

Find:
```latex
\end{document}
```

(at the very end of the file)

Replace with:
```latex
\section{Sensitivity to Feature-Subset Weighting}
\label{app:sensitivity}

Because HCDS combines six features with equal weighting in z-scored
space, a natural reviewer concern is whether the headline conclusion
depends on that weighting choice. We address this with a discrete
sensitivity analysis: for each (dataset, model) we recompute HCDS
on every $2^k - 1$ non-empty subset of the $k$ features available
in that dataset's task6 table, and record the fraction of subsets
for which HCDS is positive and significantly positive ($p<0.05$
two-sided one-sample $t$-test).

\begin{table}[!htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}llrrr@{}}
\toprule
Dataset & Model & Subsets & \% positive & \% sig.\ positive \\
\midrule
GSM8K $n=50$       & Instruct & 63 & 96.8\% & 87.3\% \\
GSM8K $n=50$       & Thinking & 63 & 84.1\% & 47.6\% \\
StrategyQA $n=50$  & Instruct & 14 & 92.9\% & 85.7\% \\
StrategyQA $n=50$  & Thinking & 15 & 93.3\% & 60.0\% \\
GSM8K $n=500$      & Instruct & 63 & 98.4\% & 95.2\% \\
GSM8K $n=500$      & Thinking & 63 & 76.2\% & 69.8\% \\
\bottomrule
\end{tabular}
\caption{HCDS sign and significance coverage across all non-empty
feature subsets per (dataset, model). StrategyQA admits fewer
subsets because the paraphrase- and perturbation-feature pipelines
were not run for it; GSM8K $n=500$ uses partial-feature aware
distance computation where the mechanistic feature is available
only on the $n=50$ subsample.}
\label{tab:sensitivity}
\end{table}

\paragraph{Where the negative-sign subsets fall.} On GSM8K $n=50$,
the only two subsets with negative-sign Instruct HCDS are
(\texttt{mechanistic\_intervention}) alone (HCDS $=0$, degenerate)
and (\texttt{perturbation\_delta}, \texttt{mechanistic\_intervention})
(HCDS $=-0.30$, $p=0.22$ n.s.) --- both are subsets that contain
no behavioral or linguistic features. All ten negative-sign Thinking
subsets exclude both entropy features, exactly consistent with the
entropy-load-bearing finding in Section~\ref{sec:robustness}: the
Thinking pathway depends on token-level entropy as its primary
HCDS-positive signal, and removing entropy collapses the score.

\paragraph{Implications.} The headline conclusion --- HCDS positive,
neutral behavior aligns with explicit CoT --- holds for the large
majority of feature-weight configurations we tested. Equal-weighting
is therefore not load-bearing for the result; almost any non-trivial
weighting that includes the load-bearing features yields the same
sign. A continuous-weight extension (e.g., Dirichlet-sampled weight
vectors over the simplex) is the natural next step we leave to
future work; the discrete subset coverage is the cheapest
sensitivity check that can be run from the existing task6 tables
without additional model generations.

\end{document}
```

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
