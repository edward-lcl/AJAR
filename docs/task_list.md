**Tasks**

 Case study figure examples:[https://arxiv.org/pdf/2409.01497](https://arxiv.org/pdf/2409.01497)

**Technical Tasks:**

* **Task 1 Due Sunday: Run Qwen3 4B Thinking 2507, Qwen3 4B Instruct 2507 on Dataset GSM8K,** under the three prompting strategies on 8.5k questions evaluate on the metrics mentioned in the proposal (Armaan and Jasmine) 

**Explicit CoT**

**Explicit No-CoT**

**Neutral**

**Rerun Qwen3 4B Thinking 2507, Qwen3 4B Instruct 2507 on Dataset StrategyQA under** three prompting strategies: [@Ryan Le](https://fall2025algov-v8h6164.slack.com/team/U09R552C57S) & [@Abdullah Sultan](https://fall2025algov-v8h6164.slack.com/team/U09LAPZU2H5)

**Neutral (rerun)**

\-**Task 2:** add casual testing via perturbations (details in proposal) and run the same 2 models with the 3 prompting techniques on the GSM8K dataset

 

**\-Task 3**: run mechanistic interpretability

 

**\-Task 4:** rinse and repeat tasks 1-3 on the StrategyQA dataset

 

**\-Task 5:** Rinse and repeat on the paired diagnostic benchmark

 

**Evaluation:** Follow this plan to run these experiments and record what LLMs behave under non-CoT and CoT. After doing these analyses, also give visual evidence like where those implicit CoT might happen inside of LLMs mechanistically based on those metrics.

 

**Task 6:** 

For each:

* Model  
* Dataset  
* Prompt condition  
* Question  
* Trial

Create a structured table with:

* Accuracy  
* Latency per token  
* Output length  
* Token entropy (mean \+ slope)  
* Paraphrase consistency  
* Perturbation Δ accuracy  
* Mechanistic intervention Δ accuracy (if open model)

 

**Task 7: Compute Hidden CoT Detection Score** 

For each dataset \+ model:

1. Compute feature vector per condition:  
   * Mean latency scaling  
   * Mean entropy pattern  
   * Mean perturbation sensitivity  
   * Consistency score  
   * Mechanistic drop (if available)  
2. Measure distance:

Distance(Neutral, Explicit CoT)  
Distance(Neutral, No-CoT)

Then define:

HCDS \= D(Neutral, NoCoT) − D(Neutral, CoT)

If HCDS \> 0 consistently → Neutral behaves more like CoT.

Deliverable:  
→ HCDS per model per dataset.

   
**Task 8: Statistical Testing**

Now test:

Is HCDS significantly \> 0?

Use:

* Bootstrap confidence intervals   
* Or paired t-tests across questions

Report:

* Mean  
* 95% CI  
* p-value

Without this, it’s descriptive only.

   
**Task 9: Perturbation Fragility Analysis**

After running perturbations.

Now quantify:

For each condition:

Δ Accuracy \= Original − Perturbed

Plot:

* Depth vs Accuracy Drop  
* Distractor vs Accuracy Drop

   
**Task 10: Mechanistic Validation**  
From the interpretability runs:

Measure:

Performance Drop after activation suppression

Compare across:

* Explicit CoT  
* Neutral  
* No-CoT

   
 **Task 11: Control Analyses**

Rule out:

* Longer outputs causing latency  
* Entropy just reflecting randomness  
* GPU noise affecting timing

Add:

Length-matched analysis  
Entropy-only baseline  
Randomized token baseline

   
**Task 12: Cross-Dataset Aggregation**

combine:

* GSM8K  
* StrategyQA  
* Paired benchmark

Check:

Is hidden CoT detection consistent across tasks?

Report:

* Mean HCDS across datasets  
* Variance

 

