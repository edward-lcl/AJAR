# Appendix D — Negative-Control Calibration

(LaTeX draft. Numbers will be filled in once
`scripts/compute_negative_control_hcds.py` produces
`results/runs/negative_control/hcds_summary_latency_only.csv`.)

```latex
\section{Negative-Control Calibration}
\label{app:negative-control}

A reviewer concern raised during PI review was that HCDS could be
picking up stylistic differences between prompts (longer, higher-
entropy, more variable completions under neutral than under no-CoT)
rather than reasoning per se. To address this, we computed HCDS on
two negative-control task families where multi-step hidden reasoning
is implausible:

\begin{itemize}
\item \textbf{Single-step arithmetic} ($n=50$): randomly generated
  questions of the form ``What is $a$ \texttt{+} $b$?'',
  ``$a$ \texttt{-} $b$?'', ``$a$ $\times$ $b$?'' with $a, b$ drawn
  from small integers. Gold answers are programmatic and exact.
\item \textbf{Numeric-answer factual lookup} ($n=50$): hand-curated
  well-known facts with verifiable integer answers (``How many
  states are in the US?'', ``In what year did humans first land on
  the Moon?'', etc.).
\end{itemize}

For both negative-control families we ran baseline generation under
the same three prompts, the same models, and the same decoding
settings as the main study, and computed the \emph{latency-only}
HCDS variant — the same single-feature variant we report on GSM8K
in Section~\ref{sec:robustness}. Latency-only is the most
adversarial setting for a stylistic-differences hypothesis: if HCDS
is measuring nothing more than ``neutral generates more / longer
text than no-CoT'' then latency-only HCDS should remain positive on
arithmetic and factual lookup, where no hidden reasoning is needed.

\begin{table}[h]
\centering
\small
\begin{tabular}{llrrr}
\toprule
Dataset & Model & $\overline{\mathrm{HCDS}}$ & 95\% CI & $p$ \\
\midrule
GSM8K  $n=50$        & Instruct & $+1.79$ & [+1.58, +2.03] & $2.6\mathrm{e}{-19}$ \\
GSM8K  $n=50$        & Thinking & $+0.40$ & [+0.21, +0.61] & $3.9\mathrm{e}{-4}$ \\
\midrule
Arithmetic $n=50$    & Instruct & TODO & TODO & TODO \\
Arithmetic $n=50$    & Thinking & TODO & TODO & TODO \\
Factual    $n=50$    & Instruct & TODO & TODO & TODO \\
Factual    $n=50$    & Thinking & TODO & TODO & TODO \\
\bottomrule
\end{tabular}
\caption{Latency-only HCDS on GSM8K (reasoning) vs the two
negative-control task families. Reasoning vs non-reasoning gap is
INTERPRETATION TODO once results land.}
\label{tab:negative-control}
\end{table}

\textbf{Interpretation.} TODO — write after seeing actual numbers.
Three possible outcomes:

\begin{itemize}
\item \emph{If negative-control HCDS $\approx 0$ for Instruct on both
families:} The latency feature is detecting reasoning-induced
timing differences, not stylistic length. This supports the main
HCDS interpretation.
\item \emph{If negative-control HCDS is positive but markedly
smaller than GSM8K:} Latency-only HCDS captures both reasoning and
stylistic effects, and the gap between reasoning and
non-reasoning tasks is itself the calibration signal.
\item \emph{If negative-control HCDS is comparable to GSM8K:} The
latency-only HCDS is largely a stylistic-difference detector. The
multi-feature HCDS in the main paper would still be informative
(since adding entropy/paraphrase/perturbation/mech provides
non-stylistic signals), but the latency feature alone should be
deemphasized.
\end{itemize}

\textbf{Method note.} We use latency-only HCDS specifically because
it is the most adversarial single-feature variant for the stylistic
hypothesis: it is measured directly from generation timing, with no
content-level processing. A full 6-feature HCDS comparison would
require also building paraphrase fixtures and running mechanistic
intervention slices on the negative-control questions; we mark this
as future work.
```
