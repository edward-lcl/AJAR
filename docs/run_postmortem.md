# Run postmortem and validity notes

Written 2026-05-05 after the 2026-05-04_2232 deep-table run completed
successfully. Two sections: where the historical "1.5-week" runtime came
from (and what we changed), and the scientific validity questions that
remain unresolved on the data we now have.

## Where the 1.5-week runtime came from

The original team reported that running this experiment to scale on cloud
GPUs (A100s, multi-GPU) took ~1.5 weeks. Investigating the initial repo
commit `7edcb34` shows the reason: **`probe_attention_vectors` was doing
one full model forward pass per probe position.**

```python
# From the original repo, scripts/run_qwen3_gsm8k_mi.py:
for probe in probes:
    pos = int(probe["full_token_position"])
    prefix = full_ids[: pos + 1].unsqueeze(0).to(device)
    outputs = model(input_ids=prefix, output_attentions=True, use_cache=False)
```

For a typical Thinking baseline with a 1500-token output, the probe plan
contains roughly one probe per reasoning step plus one per answer token —
on the order of 1500 probe positions. So each baseline triggered ~1500
fresh model forward passes just to read the attention vector at one
position from each. Causal masking guarantees the result is identical to
slicing a single full-sequence forward at those positions, so all of
that work was redundant.

Wall-time impact:
- A100 80GB: ~10-15 ms per forward pass on a 1500-token sequence
  → ~15-22 s of probing per Thinking baseline → ~3.5 hours per (model,
  prompt) for 50 questions → days for the full intervention sweep at
  the team's original scale (8.5k questions, 4 modes, multiple anchors).
- M5 Pro MPS: ~150-200 ms per forward pass on a 1500-token sequence
  → ~3-5 minutes of probing per Thinking baseline → projected ~5 days
  for a 50-question deep run before the optimisation.

After the optimisation, on this machine, the same probe phase takes
~1 second per long Thinking baseline (one full-sequence forward whose
attention output is indexed at the probe positions). The final
deep-table run completed all 1050 baselines+interventions in 7h43m.

### Was the bug in the original code?

Yes. `git show 7edcb34:scripts/run_qwen3_gsm8k_mi.py` shows the same
per-probe forward loop as our pre-optimisation code. The team's
reported 1.5-week runtime was running this version.

### Why didn't we hit it harder than they did?

We did, in two ways:

1. **MPS makes per-forward overhead worse than CUDA.** Each fresh
   `model(input_ids=prefix)` call has device-side overhead that scales
   with sequence length on MPS more steeply than on A100. So the same
   bug cost us ~5 days where it cost the team ~1.5 weeks across
   multi-GPU CUDA, which is roughly proportional to the speed ratio
   adjusted for GPU count.

2. **MPS unified memory amplified the OOM risk.** The fix introduces
   one giant numpy array of shape `(num_probes, num_layers, num_heads,
   seq_len)`. On the A100 + 80 GB CUDA path, materialising this works
   fine. On 48 GB unified memory with the model + activations + raw
   attentions also resident, we hit a 30+ GB peak and macOS killed the
   worker. We then capped `num_probes` at 64 to keep the array under
   1 GB even on long outputs (`AJAR_MAX_ANSWER_PROBES`, default 64).

### Other inefficiencies fixed in this fork

- **Intervention budget tied to baseline budget.** Every intervention
  generated up to `AJAR_MAX_NEW_TOKENS` tokens, even though the
  intervention only needs to reach the answer (~few hundred tokens
  past the modification point). New
  `AJAR_INTERVENTION_MAX_NEW_TOKENS=384` cuts intervention wall time
  4-6x on Thinking-class outputs.
- **Cross-process queue interleaving caused 30+ model swaps in a
  1.5-hour window** in the killed run. Splitting the orchestrator into
  one runner invocation per model leaves at most one swap per model
  lifetime.
- **Resume blew away aggregate JSONLs** when nothing new ran (the
  in-memory `baseline_records` was empty and overwrote the prior
  300-row file). Now the aggregate JSONL is rebuilt from per-sample
  files at end-of-run instead of from in-memory state.

### Practical implication for the team

Reproducing this experiment from scratch now costs roughly:

| Path | Time | Cost |
|---|---|---|
| Local M5 Pro Mac | ~7-8 hours overnight | $0 |
| Single A100 80GB on RunPod | ~1-2 hours | ~$1-3 |
| Full 8.5k-sample sweep on a single A100 | ~12-18 hours | ~$15-25 |

The 1.5-week timeline is no longer the relevant cost.

## Scientific validity caveats

The headline `HCDS = +1.24` (Instruct) and `HCDS = +1.03` (Thinking) are
real signals, but the following concerns should land in the methods
section before we present them as evidence of latent CoT.

### 1. n=50 per condition is too small for significance claims

The descriptive HCDS numbers from `analyze_deep_table.py` have **no
confidence intervals**. With 50 questions, swapping out 10 questions
could plausibly flip the result. Required:
- Per-question HCDS (z-score features per question, not per condition,
  then compute distance per question)
- Bootstrap CIs over 1000 resamples
- Paired t-test of `D(neutral, no_cot) - D(neutral, cot)` against zero
  per (model, dataset)

This is Task 8 in the proposal and the most important next step.

### 2. The Thinking model's anchor signal is the wrong sign

`task10_anchor_sensitivity.csv` shows `anchor_drop − control_drop`
of -0.275 for Thinking + explicit_cot and roughly 0 for the other
Thinking conditions. The proposal predicts anchor_drop > control_drop;
we see the opposite pattern. Possible explanations:

- The anchor scoring (`z(future_attn) + z(answer_attn) +
  z(activation_delta)`) selects steps that are *visually salient* in
  attention but not *causally necessary* for the answer.
- The 384-token intervention budget truncates Thinking continuations
  before they reach a final answer, biasing the comparison.
- Thinking-model traces are highly redundant and recover from any
  local intervention; random control steps land later in the trace
  where there's less recovery room.

Recommended sub-experiment (added to TODO.md): re-score anchors using
each of the three sub-features alone, measure which one actually
predicts interventionable steps, and see whether raising the
intervention budget to 768 changes the sign on Thinking + explicit_cot.

### 3. The `MAX_ANSWER_PROBES=64` cap could bias anchor selection

The optimisation samples 64 evenly-spaced positions across the answer
span instead of probing every answer token. The validation script
`scripts/validate_probe_optimization.py` confirms that on a 213-token
sample, the *single-forward optimisation itself* is bit-identical to
the prefix-forward path (anchor ranking unchanged). But that
validation kept all probes; it does not test the 64-position cap.

Required: re-run anchor selection on a few samples with the cap
disabled (`AJAR_MAX_ANSWER_PROBES=99999`), compare the top-3 anchor
ranks against the capped run, report the fraction of samples where
top-3 differs. If <5% of samples differ, the cap is safe; if more,
we need a smarter sampling strategy or a streaming aggregation that
avoids the giant numpy array entirely.

### 4. GSM8K test split is plausibly in Qwen3 training data

Qwen3-4B-Instruct hits 96-98% on GSM8K test under our cot/neutral
conditions. Either the model is genuinely capable of grade-school
arithmetic, or it has memorised some fraction of the test split. The
HCDS argument hinges on the model "doing reasoning" — but if it is
memorising, what HCDS measures is "memorisation that mimics CoT
behavioural traces" rather than "latent reasoning." Both are
interesting findings, but they are different findings.

Mitigations to consider for the paper:
- Compare HCDS on novel paraphrases (already produced in our
  paraphrase fixture) where memorisation is less likely to hit. If
  HCDS holds on paraphrases, the signal is more robust.
- Use a held-out generation (e.g. MATH or MGSM) where contamination
  is less established.
- Acknowledge the contamination concern explicitly in limitations.

### 5. Single random seed

`AJAR_SEED=17`. Affects:
- Control-step selection (which random non-anchor steps get probed)
- Distractor template choice in perturbation fixtures (which of 8
  distractors lands on each question)

If we ran with seed 18 or 19, the control-step accuracies and
perturbation Δ accuracies would shift slightly. Single-seed numbers
have no within-condition variance estimate.

Cheap fix: re-run only the intervention phase under 3 seeds (not all
baselines, just the random control selection), report variance.
Roughly 30 min per seed at the new speed.

### 6. Single trial per (sample, condition)

Decoding is deterministic (`temperature=0`), so all results are from
one trial per question. The proposal mentions "multiple trials" for
variance reduction. We have none, so we cannot estimate the variance
contributed by the model itself (vs the question and the prompt).

This is fine for deterministic decoding but means we cannot
disentangle "this question is hard" from "this prompt is hard."
Recommended: 3 trials at `temperature=0.7` on a 25-question subset to
estimate model-level variance, even if we keep the headline numbers
deterministic.

### 7. Paraphrase consistency tracks accuracy too closely

In every condition, paraphrase consistency and accuracy are within
0.05 of each other. So in HCDS feature space, paraphrase consistency
and accuracy are approximately collinear and don't add independent
information. Worth verifying with a correlation matrix; if confirmed,
report HCDS with and without paraphrase consistency in the feature
vector and check the conclusion is stable.

### 8. The numbers_preserved=true filter on paraphrases drops 22% of paraphrases

132 of 600 paraphrase rows were filtered. Fine if drops are
uncorrelated with question difficulty, but our (LLM-generated)
paraphrases probably fail the number-preservation check more often
on harder/longer questions. We should report the filtered-out rate
per condition and check that paraphrase consistency is not biased
by question difficulty.

### 9. We have not run length-matched analysis (Task 11)

Output token count varies wildly: instruct/explicit_no_cot averages
16 tokens, thinking/explicit_cot averages 912. Latency-per-token is
the only feature we standardise on length. Other features (entropy
mean, paraphrase consistency, anchor sensitivity) are not
length-controlled. Tasks 7-8 will need a length-matched control to
distinguish "neutral_strict reasons like CoT" from "neutral_strict
is verbose like CoT."

## What's actually solid

Despite the caveats above:

- **The optimisation correctness is bit-validated.** Anchor ranking is
  identical between the legacy per-probe path and the new single-forward
  path on at least one real sample. We have a reproducible script
  (`validate_probe_optimization.py`).
- **The pipeline ran 1050 items with 0 failures.** No silent corruption
  in the data files (per-sample baseline.json, hidden_summary.npz,
  intervention json all consistent).
- **The wide oMLX baseline (3000 generations) is unaffected by any of
  the MI-related optimisations.** Accuracy/latency/output_length
  numbers from that run are independent of the validity questions
  about the MI slice.
- **The HCDS direction is consistent across both models** (positive in
  both). That's a non-trivial replication signal even at this scale.

## What I'd hand to a reviewer

If this were submitted today:

> The paper demonstrates a positive HCDS signal (Neutral closer to CoT
> than to NoCoT) on Qwen3-4B Instruct and Thinking variants, descriptive
> only at n=50, with a counterintuitive negative anchor sensitivity on
> Thinking + explicit_cot that we hypothesise reflects either anchor-
> scoring saturation or intervention-budget truncation. We acknowledge
> potential GSM8K contamination of the test set and report HCDS computed
> with and without paraphrase-consistency to confirm the signal does not
> depend on a single feature. Bootstrap CIs and paired tests across
> questions are provided in the appendix.

That's the level of caveat-honesty needed before this is paper-ready.
The headline result is real; the supporting evidence needs more work.

# n=500 extension run (2026-05-07): operational lessons

The n=500 5-feature HCDS extension (`run_n500_extension.sh`) ran on
M5 Pro / 48 GB unified RAM. Two operational failures and several
optimisation opportunities surfaced.

## What killed the original overnight run

**Not sleep — out-of-memory.** The 2026-05-06 overnight launcher died
mid-stage with the laptop still on. Root cause: oMLX server held both
Qwen3-4B variants warm (~12 GB combined for 8-bit MLX), browser tabs
+ Slack + Spotify + Cotypist + Perplexity + Copilot LSPs ate another
~5 GB, and unified memory ran out. macOS started thrashing swap, the
runner stalled, eventually the system ate the process. The earlier
"sleep-related crash" hypothesis was wrong; caffeinate would not have
saved this run.

Evidence the day after on the resumed run, with the same app set
loaded: swap used = 18.0 GB / 18.4 GB total (basically saturated),
free pages ~1.1 GB. Throughput dropped from ~2-3 s/gen (expected) to
~5 s/gen — a direct signature of paging.

## Pre-flight checklist for memory-bound runs

Before launching anything multi-hour on this hardware:

1. **Quit GUI apps that aren't needed**: Spotify, Perplexity, Cotypist,
   any extra browser windows. Easy 1 GB.
2. **Close unused browser tabs**, especially video/Slack/Notion tabs.
   Firefox can hit 2-3 GB across plugin-containers.
3. **Decide whether Copilot LSP stays.** GitHub Copilot in Zed runs
   two `node` processes that together take ~750 MB. If you're not
   actively coding during the run, disable Copilot or quit Zed.
4. **Run `caffeinate -is -w <pid>`** bound to the orchestrator PID, so
   it auto-cleans up when the run finishes.
5. **AC power.** `caffeinate -s` only prevents sleep when plugged in.
6. **Confirm oMLX is serving the right model set.** Both Qwen3-4B
   variants warm = ~12 GB. Sequential phases that load one variant at
   a time would cut peak by ~6 GB.

A simple wrapper script could enforce 1-5 automatically.

## Compute lessons from stage 3

Total stage 3 time: ~18 hours wallclock for 15,000 generations
(9000 paraphrase + 6000 perturbation), vs the 5-7 hr I estimated.
Why the 2-3× error:

- **Thinking model dominates wallclock.** Long traces (sometimes
  >30 s for a single generation) anchor the average around 5 s/gen,
  not the 2-3 s/gen Instruct gets in isolation.
- **OMLX_CONCURRENCY=4 only buys ~2× speedup, not 4×.** oMLX serves
  requests with significant per-request serialisation. Worth
  benchmarking 2 vs 4 vs 8 on this hardware before committing
  hours to a run.
- **Per-phase warm-up loss.** Each phase (paraphrase, perturbation)
  loads models cold. ~2-3 min × phases × models = 10-20 min wasted
  per stage 3 launch.

## Compute optimisations to land before next n=500 run

Sorted by impact-per-hour-of-engineering.

1. **Skip "original" rows in the paraphrase fixture.** Currently
   `build_paraphrases.py` emits 1500 rows per 500 questions
   (1 original + 2 paraphrases). The "original" rows re-evaluate
   questions we already have canonical baselines for — 3000 redundant
   generations × ~5 s = ~4 hours of compute. The aggregator can join
   paraphrase variants against the canonical baseline. Implementation:
   either drop `variant_kind == "original"` from the fixture, or have
   `build_task6_table.py` resolve originals from the baseline dir.

2. **Sequential model loading.** Currently oMLX holds both Qwen3-4B
   variants warm throughout stage 3. Running all instruct work first,
   unloading, then loading thinking, keeps peak weights at ~6 GB
   instead of ~12 GB. On 48 GB hardware with browser+chat apps
   loaded, this is the difference between thrashing swap and not.
   Implementation: split each phase into model-keyed sub-runs at the
   orchestrator level, with explicit unload between.

3. **Per-phase resume already works** — keep this. Per-sample dirs
   plus `AJAR_RESUME=1` saved the run when the laptop crashed at
   midnight: zero useful work was lost.

4. **NUM_PARAPHRASES=1 for fast turnarounds.** Halves paraphrase cost
   (~5 hours saved) at the cost of a noisier paraphrase-consistency
   signal. Worth it for prelim/iteration; not for the final paper
   table.

5. **Caffeinate inside the launcher.** Add to
   `run_overnight_pre_icml.sh` and `run_n500_extension.sh`:
   ```bash
   caffeinate -is -w $$ &
   trap 'kill %1 2>/dev/null' EXIT
   ```

## Hygiene: per-sample JSON field names

Per-sample `baseline.json` uses `correct` (not `is_correct`),
`prediction_numeric` (not `predicted_numeric`). Caused a false alarm
during pre-sleep audit. Either standardise to one set of names or
add a `SCHEMA.md` next to the runner. Low priority but easy.

## What we got right

- **Per-sample-dir + idempotent resume.** Saved hours when the
  laptop OOM'd mid-run.
- **n=500 behavioural + n=50 mech split.** Compute economy plus
  scientifically defensible — the paper's defensible position is
  this two-cut framing, not a chase for full 6-feature at n=500.
- **Aggregator design is format-tolerant.** `build_task6_table.py`
  silently drops missing features, so a partial run still produces
  a defensible (smaller) HCDS. No special-case code needed for the
  resume case.

## 2026-06-13 — Gemma cross-family + Thinking pole repair (3 bugs)
- **Gemma3 fp16 = garbage.** torch AJAR_DTYPE=auto picks fp16 on MPS; Gemma3 activations
  overflow fp16 -> NaN logits -> 1024 tokens that decode to "" (num_entropy_tokens=0,
  mean_entropy_generated=None). The aggregate silently fell back to 3-feature. **Always run
  Gemma torch with AJAR_DTYPE=float32** (verified: gen_len=154, entropy=0.068). Qwen is
  fp16-stable; Gemma is not.
- **Full Thinking torch OOMs** on long cot/neutral (1536-tok eager attention, worker -9).
  Scope no-CoT pole repair to explicit_no_cot only (short force-closed outputs).
- **pgrep-on-script-name self-deadlock.** A waiter whose command line contains the script
  name X makes `pgrep -f X` in another script match the waiter forever. Wait on exact PIDs
  (`kill -0 $PID`) or on the python runner, never on script-name patterns that self-match.
