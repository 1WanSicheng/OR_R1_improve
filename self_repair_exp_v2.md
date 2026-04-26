# Experiment Plan V2: Improve Self-Repair Data Before Training

## Goal

Re-run the self-repair idea from first principles:

**If the previous training did not improve Pass@1, first improve the quality of repair supervision before doing any more training.**

The central question for V2 is:

**Can we convert more failed one-shot outputs into high-quality repair trajectories that contain real corrective information?**

Only if the answer is yes should we continue to training.

---

## First-Principles Diagnosis From V1

The previous round failed mainly because the training signal was too weak.

Observed issues:

1. Too many samples were labeled as `no_code`, so the pipeline was learning formatting failure more than modeling failure.
2. Diagnosis routing was too coarse, so `runtime_error`, `missing_variable`, and `infeasible` often received generic repair instructions.
3. A single repair template was used for very different failure modes.
4. SFT filtering was too strict, so the final repair dataset was nearly empty.

Therefore, V2 should optimize the supervision pipeline in this order:

1. improve first-pass output structure
2. improve code extraction and execution robustness
3. improve failure-type-specific repair instructions
4. improve repair data acceptance rules
5. only then test training

---

## Stage 1: Build Better Repair Data

### 1.1 Strong-format first pass

Replace the weak first-pass prompt with a strict response format:

- mathematical model sections
- explicit `coptpy` implementation
- fenced Python code block
- requirement that the final code defines and solves `model`

Purpose:

- reduce false `no_code`
- convert formatting failures into executable samples

### 1.2 Robust code extraction

Use a more tolerant extraction pipeline:

- accept ```python fences
- accept generic ``` fences
- recover code from a `## Python` section when the fence is missing
- recover code from the first detected `import` / `from` block if needed

Purpose:

- prevent otherwise usable samples from being discarded

### 1.3 Route failures into different repair modes

Instead of one repair prompt, use three modes:

- `format repair`: for `no_code`
- `execution repair`: for `missing_variable`, `runtime_error`, `syntax_error`
- `modeling repair`: for `infeasible`, `unbounded`, `wrong_answer`

Purpose:

- make the repair instruction match the real failure source

### 1.4 Improve diagnosis

Use explicit state-aware diagnosis:

- `no_code` -> formatting/non-compliance
- `missing_variable` -> imports, undefined symbols, bad indexing
- `runtime_error` -> API misuse, tensor/indexing misuse, invalid COPT constraint builder
- `infeasible` -> constraint conflict / sign / bound issue
- `unbounded` -> missing bound/resource constraint
- `wrong_answer` -> objective direction, integrality, missing key constraints, numeric mismatch

Purpose:

- turn generic repair into targeted repair

### 1.5 Tiered SFT acceptance

Do not keep only fully correct repairs.

Admit repaired samples when they achieve one of the following:

- `no_code -> executable`
- `runtime_error/missing_variable -> optimal`
- `infeasible/unbounded -> optimal or feasible`
- `wrong_answer -> numerically closer and optimal`

Also record a quality tier for each sample.

Purpose:

- increase sample count while preserving ranking by usefulness

---

## Stage 2: Validate Repair Usefulness Again

Compare V2 coarse repair vs V2 diagnosis-guided repair on the same failed subset.

Metrics:

- code recovery rate
- executable rate
- optimal rate
- correct rate
- improved rate
- number of accepted SFT samples

Decision rule:

- continue to training only if V2 diagnosis-guided repair is materially better than V1 and better than V2 coarse repair
- otherwise stop and report that supervision quality is still insufficient

---

## Stage 3: Train Only If The Data Is Better

If Stage 2 succeeds:

- train a small LoRA SFT branch from the same SFT base
- use only V2 repair data
- keep training short for fast validation

The purpose is not to claim final performance, but to test whether better repair supervision starts to transfer to first-pass behavior.

---

## Stage 4: Held-Out Validation

Evaluate first-pass Pass@1 on:

- ComplexOR
- IndustryOR

Optionally compare repair-mode metrics on:

- ComplexOR full
- IndustryOR first 20

---

## Interpretation

### Case A
Repair data quality improves and first-pass improves.

Conclusion:

- repair supervision is now informative enough to help capability

### Case B
Repair data quality improves but first-pass does not.

Conclusion:

- the repair policy improved, but transfer to one-shot solving is still weak

### Case C
Repair data quality still does not improve.

Conclusion:

- the bottleneck remains in supervision construction, not training

---

## Minimum Success Criterion For V2

To consider V2 successful enough to continue:

1. `no_code` must drop substantially
2. accepted repair-SFT samples must increase from near-zero to a meaningful count
3. diagnosis-guided repair must beat coarse repair on execution/optimal/correct metrics

Only then is further training justified.
