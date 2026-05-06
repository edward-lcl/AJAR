# Team Handoff — 2026-05-05

This is the "start here" doc for the AJAR project after the work on
2026-05-04 and 2026-05-05. Everything here is already on the `main`
branch — you don't need to merge anything to look at it.

---

## What we were trying to figure out

Big language models like ChatGPT can solve math word problems. Sometimes
they show their work ("step 1, step 2, step 3, the answer is 18").
Sometimes they just blurt out the answer.

The question this project asks: **when the model doesn't show its work,
is it still secretly doing the steps in its head?** Or is it just
pattern-matching to a memorized answer?

That "secret thinking" idea is what the proposal calls **hidden chain-of-thought**.
Detecting whether it's happening is the whole point of this research.

---

## What we did this round

We ran the same 50 math questions (from a benchmark called GSM8K) through
two versions of the same model — Qwen3-4B "Instruct" and Qwen3-4B
"Thinking" — under three different prompts:

- **explicit_cot**: "show your work step by step" — the model definitely does CoT
- **explicit_no_cot**: "just give me the answer, no reasoning" — the model definitely doesn't
- **neutral_strict**: just ask the question, no instructions — the test condition

Then we measured a bunch of things about each answer (how long it took,
how confident the model was at each word, whether paraphrases of the
question got the same answer, etc.) and compared them across the three
prompts.

If "neutral" looks more like "show your work" than like "just the answer",
that's evidence the model was secretly thinking step-by-step even though
nobody told it to. The proposal calls this number **HCDS** (Hidden CoT
Detection Score) — bigger means "yes, it's hiding CoT".

---

## The headline finding

**Both models scored positive on HCDS.**

- Instruct model: HCDS = +1.24
- Thinking model: HCDS = +1.03

In plain English: when we just asked the question without instructions,
both models behaved much more like they were doing step-by-step reasoning
than like they were just blurting out answers. Same direction, both
models, every measurement we tracked. **This is what the proposal predicted
would happen if hidden CoT is real.**

⚠️ **Important caveat**: 50 questions is a small sample. To turn this
from "looks like a real signal" into "this is real with X% confidence",
we need to run some statistics (bootstrap and t-tests). That's the next
job. See "what to do next" below.

---

## The story of the week-long runtime (the simple version)

The original team that built this code said running the experiment took
about a week and a half on cloud GPUs. We figured out why, and it's not
that the experiment is hard — **the code was doing a huge amount of
unnecessary work.**

Here's the analogy. Imagine you have a 1500-page book and you want to
look up one specific word on each page. Two ways to do it:

1. Open the book to page 1, read the word, close the book. Open the book
   to page 2, read the word, close the book. Repeat 1500 times.
2. Open the book once, flip through and read the word from each page.

The original code was doing option #1, **running the entire AI model from
scratch 1500 times** to read 1500 small pieces of attention data — when
it could have just run the model once and pulled out the same 1500 pieces.

After we changed it to option #2, the work that used to take ~2 minutes
per question now takes about 1 second per question. Multiplied across
all the questions, models, prompts, and intervention experiments, that's
the difference between "a week and a half" and "overnight".

We also confirmed the new fast version produces the **identical
mathematical result** as the slow version (we ran them both and compared
the numbers). So this is purely a speed fix — no scientific tradeoffs.

**The bug was in the original code, not something we introduced.** Anyone
who reruns this experiment from scratch now gets:

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

That's a spreadsheet with 300 rows (50 questions × 2 models × 3 prompts)
and one column per metric: accuracy, output length, latency, token-level
confidence, paraphrase consistency, perturbation effect, and the
mechanistic intervention effect. Drop it into Excel, pandas, or a
notebook and you can start the analysis immediately.

The "what does each condition look like on average" summary plus our
findings is in the same folder:

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

### 1. Turn the HCDS number into a real claim (~3 hours)

Right now we have one number per model. To say "this result is real and
not a coincidence", we need to:

- Compute HCDS *per question* instead of just averaging across all of them
- Resample the questions 1000 times with replacement (bootstrap) to see
  how stable the number is
- Run a paired t-test to get an actual p-value

If HCDS is reliably positive across resamples and the p-value is below
0.05, **that's the headline result for the paper**. If it's not, we
need more data.

Who picks this up: anyone comfortable with pandas + scipy. Suggested
file to write: `scripts/compute_hcds.py`.

### 2. Investigate one weird finding (~2 hours)

We tried to identify the "most important reasoning steps" in each answer
(we call these "anchors") and then test whether suppressing them hurts
the model's answer more than suppressing random steps would.

For most conditions, this worked as expected. **But for the Thinking
model under explicit_cot, the opposite happened**: messing with random
steps hurt the model *more* than messing with the steps we picked as
important. That's the opposite of what should happen if our anchor-picking
is correct.

Three things to try:
- Look at our anchor-picking formula and see if one part of it
  (attention to future words, attention from the answer, or activation
  changes) is misleading
- Re-run with a bigger token budget for the intervention follow-ups —
  maybe they were getting cut off too early to reach a final answer
- Pick 5 of the bad cases and read what's actually happening

### 3. Test whether HCDS holds up if we drop one feature (~1 hour)

HCDS combines six different measurements. Two of them are suspicious:

- **Paraphrase consistency** mostly just tracks accuracy (if the model
  gets a question right, it gets paraphrases of it right too). So it
  might be redundant.
- **Mechanistic intervention effect** is acting weird on the Thinking
  model (see #2 above).

Quick check: compute HCDS four ways — with everything, without paraphrase
consistency, without the mechanistic measurement, and without both. If
the conclusion holds in all four, it's robust.

### 4. Length-matched analysis (~3 hours)

The "explicit_no_cot" answers are 16 tokens long. The "explicit_cot"
answers are 900 tokens long. If we're not careful, "neutral acts like
CoT" might just mean "neutral is verbose like CoT" rather than "neutral
is reasoning like CoT".

To rule this out, group questions into length bins and recompute HCDS
within each bin. If neutral_strict still looks like CoT even at matched
output lengths, the reasoning interpretation is stronger.

---

## Things that aren't urgent but matter

- **StrategyQA replication** (Task 4 from the proposal). Same code,
  different dataset (yes/no commonsense reasoning instead of math).
  Mostly: write a fixture loader and a yes/no answer parser.
- **Paired diagnostic benchmark** (Task 5). Waiting on the actual
  benchmark file.
- **Dependency-broken perturbation variant**. The proposal asks for two
  perturbation types; we shipped one (irrelevant distractor sentences).
  The other (slightly altering numbers to break the math chain) needs
  more thought because the gold answer changes.
- **mlxterp evaluation**. There's a newer Apple-Silicon-native library
  for this kind of analysis that could be ~3x faster than what we're
  using. Worth a 2-day prototype before committing.

---

## Honest limitations to put in the paper

These are all real. Don't paper over them in writeup:

1. **50 questions is a small sample.** The HCDS numbers we have are
   suggestive, not statistically established. The bootstrap work in
   "what's next" #1 is the fix.
2. **GSM8K math problems were almost certainly in the model's training
   data.** Our 96-98% accuracy might be the model genuinely reasoning,
   or it might be the model remembering. Both are interesting findings,
   but they mean different things. Mitigation: we already have
   paraphrased versions of every question — running HCDS on those
   shows whether the signal survives when memorization is less
   plausible.
3. **The Thinking model ignores instructions to skip reasoning.** When
   we tell it "answer-only mode, no reasoning", it produces ~600 tokens
   of reasoning anyway. So Thinking + explicit_no_cot isn't really a
   "no CoT" condition for that model. Worth reporting as its own
   finding rather than treating it like a clean control.
4. **One random seed, one trial per question.** Decoding is
   deterministic so the answer doesn't vary, but we have no way to
   estimate "how much variance is the model itself contributing vs the
   question vs the prompt".
5. **The 64-probe-per-question cap might bias which steps we pick as
   anchors** on long Thinking outputs. We have a validation script
   that confirms our speed optimization doesn't change the math, but
   we haven't separately validated that capping (vs probing every
   answer token) gives the same anchors.

The full technical version of all these is in `docs/run_postmortem.md`.

---

## How to reproduce any of this

If you want to rerun the deep-table experiment from scratch on a
different machine:

```bash
git clone https://github.com/edward-lcl/AJAR
cd AJAR
pip install -r requirements.txt
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507
huggingface-cli download Qwen/Qwen3-4B-Thinking-2507
./scripts/run_deep_table.sh             # 50 questions (default)
./scripts/run_deep_table.sh 100         # different sample count
```

It's idempotent — if you kill it partway through, just rerun and it
picks up where it stopped.

If you have access to a cloud GPU, see `docs/cloud_gpu_setup.md` for the
RunPod recipe (~$2, ~2 hours total).

---

## Where to look in the repo

In rough reading order:

1. **`README.md`** — setup recipe and the workflow command reference
2. **`HANDOFF.md`** (this file) — what's done, what's next
3. **`results/RUN_INDEX.md`** — table of every completed run
4. **`results/runs/2026-05-04_2232_gsm8k50_qwen3-4b_deep-table/SUMMARY.md`** — findings from this run, in detail
5. **`docs/run_postmortem.md`** — the full technical version of the
   speed-up explanation and the validity caveats
6. **`TODO.md`** — backlog organized by priority

The commit log (`git log --oneline | head -15`) is also a readable
record of every decision we made and why, in chronological order.
