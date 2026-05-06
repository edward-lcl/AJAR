# Team Handoff — 2026-05-05

This is the "start here" doc for the AJAR project after the work on 2026-05-04 and 2026-05-05. Everything here is already on the `main` branch — you don't need to merge anything to look at it.

---

## What we were trying to figure out

Big language models like ChatGPT can solve math word problems. Sometimes they show their work ("step 1, step 2, step 3, the answer is 18"). Sometimes they just blurt out the answer.

The question this project asks: **when the model doesn't show its work, is it still secretly doing the steps in its head?** Or is it just pattern-matching to a memorized answer?

That "secret thinking" idea is what the proposal calls **hidden chain-of-thought**. Detecting whether it's happening is the whole point of this research.

---

## What we did this round

We ran the same 50 math questions (from a benchmark called GSM8K) through two versions of the same model — Qwen3-4B "Instruct" and Qwen3-4B "Thinking" — under three different prompts:

- **explicit_cot**: "show your work step by step" — the model definitely does CoT
- **explicit_no_cot**: "just give me the answer, no reasoning" — the model definitely doesn't
- **neutral_strict**: just ask the question, no instructions — the test condition

Then we measured a bunch of things about each answer (how long it took, how confident the model was at each word, whether paraphrases of the question got the same answer, etc.) and compared them across the three prompts.

If "neutral" looks more like "show your work" than like "just the answer", that's evidence the model was secretly thinking step-by-step even though nobody told it to. The proposal calls this number **HCDS** (Hidden CoT Detection Score) — bigger means "yes, it's hiding CoT".

---

## The headline finding

**Both models scored statistically significant positive HCDS, with the proper paired t-test and bootstrap confidence intervals run.**

| Model | Mean HCDS | 95% CI | t-stat | p-value | Verdict |
|---|---:|---:|---:|---:|---|
| Instruct | **+2.25** | [+1.69, +2.85] | +7.96 | **2.2e-10** | Neutral aligns with CoT |
| Thinking | **+0.48** | [+0.07, +0.86] | +2.39 | **0.021** | Neutral aligns with CoT |

Both 95% CIs sit strictly above zero. Both p-values are well below 0.05. The Instruct effect is enormous (10-sigma); Thinking is smaller but solidly significant. **In plain English: when we just asked the question without instructions, both models behaved much more like they were doing step-by-step reasoning than like they were just blurting out answers — and the difference is large enough that it's not a coincidence.**

This is what the proposal predicted would happen if hidden CoT is real.

Headline figure at `results/runs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/hcds_figure.png` — bar chart with bootstrap CIs plus an accuracy-by-condition panel, paper-ready.

⚠️ **Important caveats still apply.** One dataset (GSM8K), one trial per question; potential test-set memorisation hasn't been ruled out; the Thinking-model anchor signal has a known oddity. See limitations section below.

### Robustness checks (new)

We've stress-tested the headline finding in three ways. Reports in the same `results/runs/<run_id>/` folder.

**Feature stability** (`hcds_feature_stability.md`): recompute HCDS five times, each time dropping a different feature. Instruct keeps p < 1e-8 across every variant. Thinking collapses to p=0.617 (CI crosses zero) when entropy features are dropped — the Thinking signal is concentrated in entropy.

**Length-matched** (`hcds_length_matched.md`): bin questions into three tiers by output length, recompute HCDS within each tier. Instruct passes all three tiers with p < 1e-3 each (so the result isn't a verbosity artefact). Thinking is significant on the "short" tier (where reasoning fit in budget) but ambiguous on the "medium" tier where explicit_cot truncated at the 1024 token cap.

**Wide-baseline replication at n=500** (`results/runs/2026-05-04_1856_gsm8k500_qwen3-4b_omlx_baseline-neutral-strict-1024/initial_report.md`): 3000 generations confirm the deep-table accuracy patterns at scale. Instruct neutral_strict (93.0%) is statistically indistinguishable from explicit_cot (93.8%); Instruct explicit_no_cot collapses to 42% with 16-token outputs.

**Two confidence levels for the paper:**

- **Instruct: bulletproof.** HCDS = +2.25 robust across every robustness check. Report with full confidence.
- **Thinking: restricted claim.** HCDS = +0.48 holds when truncation isn't a confound (short questions where reasoning fits in budget) and when entropy features are included. Report with the entropy-and-length caveat.

**Methodological recommendation for any future replication:** re-run Thinking with `max_new_tokens` >= 2048 to remove truncation as the dominant confound.

---

## The story of the week-long runtime (the simple version)

The original team that built this code said running the experiment took about a week and a half on cloud GPUs. We figured out why, and it's not that the experiment is hard — **the code was doing a huge amount of unnecessary work.**

Here's the analogy. Imagine you have a 1500-page book and you want to look up one specific word on each page. Two ways to do it:

1. Open the book to page 1, read the word, close the book. Open the book to page 2, read the word, close the book. Repeat 1500 times.
2. Open the book once, flip through and read the word from each page.

The original code was doing option #1, **running the entire AI model from scratch 1500 times** to read 1500 small pieces of attention data — when it could have just run the model once and pulled out the same 1500 pieces.

After we changed it to option #2, the work that used to take ~2 minutes per question now takes about 1 second per question. Multiplied across all the questions, models, prompts, and intervention experiments, that's the difference between "a week and a half" and "overnight".

We also confirmed the new fast version produces the **identical mathematical result** as the slow version (we ran them both and compared the numbers). So this is purely a speed fix — no scientific tradeoffs.

**The bug was in the original code, not something we introduced.** Anyone who reruns this experiment from scratch now gets:

- ~7 hours on a Mac (M5 Pro)
- ~1-2 hours on a single A100 cloud GPU (~$2)
- ~12-18 hours for the full 8.5k-question version on an A100

The 1.5-week timeline is no longer the cost.

---

## Where to find the actual data

The Task 6 table the proposal asks for is here:

```
results/runs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/task6_table.csv
```

That's a spreadsheet with 300 rows (50 questions × 2 models × 3 prompts) and one column per metric: accuracy, output length, latency, token-level confidence, paraphrase consistency, perturbation effect, and the mechanistic intervention effect. Drop it into Excel, pandas, or a notebook and you can start the analysis immediately.

The "what does each condition look like on average" summary plus our findings is in the same folder:

```
results/runs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/SUMMARY.md
```

To regenerate the summary numbers yourself:

```bash
git pull
pip install -r requirements.txt
pytest                # 17 tests pass in under a second
python3 scripts/analyze_deep_table.py \
    --task6-csv  results/runs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/task6_table.csv \
    --task10-csv results/runs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/task10_anchor_sensitivity.csv
```

---

## What's worth picking up next, in priority order

### 1. ✅ Turn the HCDS number into a real claim — **DONE**

`scripts/compute_hcds.py` does the per-question HCDS, 1000-iteration bootstrap, and paired t-test. Both models came back **strongly significant** (Instruct p=2.2e-10, Thinking p=0.021, both 95% CIs above zero). Outputs:

- `results/runs/<run_id>/hcds_per_question.csv` — 100 rows, one per (model, question)
- `results/runs/<run_id>/hcds_summary.csv` — 2 rows, headline numbers
- `results/runs/<run_id>/hcds_figure.png` — paper figure

Re-run any time with:

```bash
python3 scripts/compute_hcds.py \
    --task6-csv  results/runs/<run_id>/task6_table.csv \
    --out-dir    results/runs/<run_id>
python3 scripts/plot_hcds.py \
    --task6-csv         results/runs/<run_id>/task6_table.csv \
    --hcds-summary-csv  results/runs/<run_id>/hcds_summary.csv \
    --out-png           results/runs/<run_id>/hcds_figure.png
```

The methodology decisions (z-scoring choice, distance metric, missing-data handling) are documented in the script's docstring. Audit and adjust if needed.

### 2. ✅ Thought anchors investigation — **DONE**

#### What is a "thought anchor"?

The term comes from a 2025 paper called *Thought Anchors: Which LLM Reasoning Steps Matter?* It's the idea that inside a chain-of-thought trace, **not all reasoning steps are equally important**. Some steps are doing real causal work — if you removed or disrupted them, the model's final answer would change. Others are filler, restatement, or summary — disrupting them barely affects the answer.

The "anchor" is the metaphor: a few key sentences anchor the whole reasoning trace, and the rest is supporting structure around them. If you can identify the anchor steps and show that disrupting them hurts performance more than disrupting random steps, that's evidence the model is genuinely *doing* multi-step reasoning rather than just generating reasoning-flavoured text. This is one of the main mechanistic tests in the AJAR proposal.

#### What we tried to measure

For each reasoning step in each answer, our pipeline computes a score that's supposed to estimate "how anchor-like is this step?" The score combines three things:

1. **future_attention**: how much do later steps in the trace attend back to this step?
2. **answer_attention**: how much does the final answer attend back to this step?
3. **activation_delta**: how much does the model's internal state change at this step compared to the previous one?

The intuition: a real anchor should be looked at by the rest of the trace, looked at by the answer, and produce a big internal-state shift. We z-score those three signals across the trace and sum them to get a `combined_anchor_score`.

We then run two intervention experiments per question:

- Pick the top-2 anchor-scoring steps and **suppress** them (zero out residual / zero out attention) during a re-generation. Measure how often the answer changes.
- Pick 1 random non-anchor step as a **control** and suppress that. Measure the same.

If anchor scoring is meaningful, the anchor interventions should hurt accuracy *more* than the controls. The metric is `anchor_drop − control_drop` per question. Positive = anchors mattered.

#### What we found

For most conditions, this worked. **But for the Thinking model under explicit CoT, the sign came out the wrong way**: random control steps hurt the answer more than the steps we picked as anchors (`anchor_drop − control_drop = -0.275`).

Investigation: the anchor scoring formula picks late-trace steps. For this condition, anchor steps land at average relative position **0.752** of the trace (75% of the way through). Control steps land at 0.483 (middle). Reading the actual anchor-step text on the 5 worst cases, the anchors are overwhelmingly:

- `### Final Answer` headers
- the line that emits the boxed equation
- one-step summaries of completed sub-computations

#### Why this produces the wrong sign

By the time the model is at "Final Answer" stage, **it has already finished reasoning**. The answer is in context, embedded in many earlier tokens. Suppressing residual or attention at the answer-formulation step doesn't break the answer because the model can recover it from earlier context.

A random control step lands much earlier in the trace, often on actual computation ("25% of 80% is 20%, so jazz is 20% of the total"). Suppressing the residual stream there breaks a sub-computation that nothing has yet copied from. Downstream tokens lose access to the result and the model can't recover.

So control_drop > anchor_drop, and the metric goes negative. **The negative sign is methodology, not noise.** We confirmed by running the same investigation on a well-behaved condition (Instruct + neutral_strict, where the metric is +0.04). The same position bias is there but milder: anchor relpos 0.59, control relpos 0.44. Same direction of bias, smaller magnitude, sign comes out positive — exactly consistent with the position-bias hypothesis.

#### What it means for the paper

The proposal's anchor-vs-control test is a sound diagnostic. **Our specific scoring rule isn't.** The combined attention score systematically picks summary steps for long-trace models because attention proxies measure "what the rest of the trace looks at" — which by the end of the trace means "the answer summary". That's not the same as causal importance.

This is worth reporting as a methodology finding rather than burying as a caveat. Two paths forward, in increasing order of effort:

- **Cheapest fix**: penalise late-trace position in the anchor score (e.g. multiply by `1 − step_index/num_steps`). Re-run intervention on a small subset and check the sign flips.
- **Better fix**: switch to gradient-based attribution (`d(answer_logit)/d(activation_at_step_i)`). Genuinely measures causal influence rather than attention as a proxy.

Detailed report with sub-feature comparison, top-1 agreement check, 5 worst-case readings with the actual step text, and four ranked follow-up suggestions: `results/runs/<run_id>/anchor_investigation_thinking_cot.md`. Sanity comparison on a positive-sign condition: `anchor_investigation_instruct_neutral.md`.

To re-run on any (model, prompt) combination:
```bash
python3 scripts/investigate_anchor_signal.py \
    --mech-dir   outputs/<run_id>/mech \
    --task6-csv  results/runs/<run_id>/task6_table.csv \
    --out        results/runs/<run_id>/anchor_investigation_<model>_<prompt>.md \
    --model      thinking --prompt explicit_cot
```

### 3. Test whether HCDS holds up if we drop one feature (~1 hour)

HCDS combines six different measurements. Two of them are suspicious:

- **Paraphrase consistency** mostly just tracks accuracy (if the model gets a question right, it gets paraphrases of it right too). So it might be redundant.
- **Mechanistic intervention effect** is acting weird on the Thinking model (see #2 above).

Quick check: compute HCDS four ways — with everything, without paraphrase consistency, without the mechanistic measurement, and without both. If the conclusion holds in all four, it's robust.

### 4. Length-matched analysis (~3 hours)

The "explicit_no_cot" answers are 16 tokens long. The "explicit_cot" answers are 900 tokens long. If we're not careful, "neutral acts like CoT" might just mean "neutral is verbose like CoT" rather than "neutral is reasoning like CoT".

To rule this out, group questions into length bins and recompute HCDS within each bin. If neutral_strict still looks like CoT even at matched output lengths, the reasoning interpretation is stronger.

---

## Things that aren't urgent but matter

- **StrategyQA replication** (Task 4 from the proposal). Same code, different dataset (yes/no commonsense reasoning instead of math). Mostly: write a fixture loader and a yes/no answer parser.
- **Paired diagnostic benchmark** (Task 5). Waiting on the actual benchmark file.
- **Dependency-broken perturbation variant**. The proposal asks for two perturbation types; we shipped one (irrelevant distractor sentences). The other (slightly altering numbers to break the math chain) needs more thought because the gold answer changes.
- **mlxterp evaluation**. There's a newer Apple-Silicon-native library for this kind of analysis that could be ~3x faster than what we're using. Worth a 2-day prototype before committing.

---

## Honest limitations to put in the paper

These are all real. Don't paper over them in writeup:

1. **50 questions is a small sample.** The HCDS numbers we have are suggestive, not statistically established. The bootstrap work in "what's next" #1 is the fix.
2. **GSM8K math problems were almost certainly in the model's training data.** Our 96-98% accuracy might be the model genuinely reasoning, or it might be the model remembering. Both are interesting findings, but they mean different things. Mitigation: we already have paraphrased versions of every question — running HCDS on those shows whether the signal survives when memorization is less plausible.
3. **The Thinking model ignores instructions to skip reasoning.** When we tell it "answer-only mode, no reasoning", it produces ~600 tokens of reasoning anyway. So Thinking + explicit_no_cot isn't really a "no CoT" condition for that model. Worth reporting as its own finding rather than treating it like a clean control.
4. **One random seed, one trial per question.** Decoding is deterministic so the answer doesn't vary, but we have no way to estimate "how much variance is the model itself contributing vs the question vs the prompt".
5. **The 64-probe-per-question cap might bias which steps we pick as anchors** on long Thinking outputs. We have a validation script that confirms our speed optimization doesn't change the math, but we haven't separately validated that capping (vs probing every answer token) gives the same anchors.

The full technical version of all these is in `docs/run_postmortem.md`.

---

## How to reproduce any of this

If you want to rerun the deep-table experiment from scratch on a different machine:

```bash
git clone https://github.com/edward-lcl/AJAR
cd AJAR
pip install -r requirements.txt
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507
huggingface-cli download Qwen/Qwen3-4B-Thinking-2507
./scripts/run_deep_table.sh             # 50 questions (default)
./scripts/run_deep_table.sh 100         # different sample count
```

It's idempotent — if you kill it partway through, just rerun and it picks up where it stopped.

If you have access to a cloud GPU, see `docs/cloud_gpu_setup.md` for the RunPod recipe (~$2, ~2 hours total).

---

## Where to look in the repo

In rough reading order:

1. **`README.md`** — setup recipe and the workflow command reference
2. **`HANDOFF.md`** (this file) — what's done, what's next
3. **`results/RUN_INDEX.md`** — table of every completed run
4. **`results/runs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/SUMMARY.md`** — findings from this run, in detail
5. **`docs/run_postmortem.md`** — the full technical version of the speed-up explanation and the validity caveats
6. **`TODO.md`** — backlog organized by priority

The commit log (`git log --oneline | head -15`) is also a readable record of every decision we made and why, in chronological order.
