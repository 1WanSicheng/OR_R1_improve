# Experiment Plan: Can Self-Repair Training Improve OR-R1?

## Goal
Answer one final question:

**Can self-repair training improve model capability, rather than only acting as a second attempt at inference time?**

To support this claim, the experiment must separate:
1. **repair usefulness at inference time**
2. **training benefit after learning from repair data**
3. **true capability gain on first-pass generation**

---

## Stage 1: Build diagnosis-guided repair data

### 1.1 Collect failed first-pass samples
Run the current OR-R1 model on training problems and collect failed cases.

For each case, save:
- problem text
- first-pass generated text/code
- solver status
- predicted answer
- gold answer if available

### 1.2 Label error types
Assign each failed case one simple error type:
- runtime/API error
- infeasible
- unbounded
- objective direction error
- missing integrality
- missing key constraint
- other

### 1.3 Create structured diagnosis
For each labeled error, attach a short diagnosis block:
- failure type
- likely cause
- repair instruction

### 1.4 Generate repaired outputs
For each failed case, run one-round diagnosis-guided repair and save:
- repaired text/code
- repaired solver status
- repaired answer
- whether the original error was fixed
- whether the final answer became correct

### Stage 1 output
Create a repair dataset of triples:

`(problem, failed_output, diagnosis) -> repaired_output`

This dataset is the basis for later training.

---

## Stage 2: Verify diagnosis-guided repair is useful

Before training, first check whether diagnosis-guided repair is actually better than coarse repair.

### Compare two settings
- **Coarse repair**: only generic feedback such as runtime error / infeasible / wrong answer
- **Diagnosis-guided repair**: structured diagnosis added

### Evaluate
Compare on the same failed sample set:

- error fix rate
- infeasible -> feasible rate
- unbounded -> bounded rate
- final correct rate after repair

### Decision rule
Only continue if diagnosis-guided repair is clearly better than coarse repair.

Otherwise, stop.  
If diagnosis itself does not improve repair quality, then training on it is unlikely to help.

---

## Stage 3: Train a self-repair model

Now test whether learning from repair data improves the model itself.

### Training setting
Start from the same SFT checkpoint and create two training branches:

- **Baseline branch**: keep original OR-R1 training
- **Self-repair branch**: add repair data for supervised fine-tuning

### Self-repair training data format
Use examples like:

- input:
  - problem
  - failed first-pass output
  - diagnosis
- target:
  - repaired formulation/code

### Training objective
Teach the model to map diagnosed failures to targeted correction.

At this stage, use **SFT first**.  
Do not start with RL.

---

## Stage 4: Test whether training improves first-pass capability

This is the key stage.

The final claim is not whether repair works.  
The final claim is whether **self-repair training improves model capability**.

### Evaluate both branches on held-out benchmarks
Use the same evaluation protocol for:
- baseline-trained model
- self-repair-trained model

### Main metrics
- Pass@1
- average solving accuracy
- valid code rate
- feasibility rate
- infeasible rate
- unbounded rate

### Core requirement
The model must be evaluated in **first-pass generation mode**, not repair mode.

This is necessary because the question is:
- does self-repair training improve the base model itself?

not:
- can the model succeed after an extra repair round?

---

## Stage 5: Add a repair-mode evaluation

Besides first-pass evaluation, also compare both models in repair mode.

### Why
This helps answer two different questions:

1. **Capability gain**: does self-repair training improve Pass@1?
2. **Workflow gain**: does self-repair training improve repair success when repair is allowed?

### Metrics in repair mode
- final correct rate after one repair
- error fix rate
- infeasible -> feasible rate
- unbounded -> bounded rate

---

## Stage 6: Final comparison table

The final report should include both first-pass and repair-mode results.

| Model | Pass@1 | Avg Acc | Valid Code | Feasible | Infeasible | Unbounded | Repair Final Correct | Repair Fix Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline OR-R1 |  |  |  |  |  |  |  |  |
| OR-R1 + self-repair training |  |  |  |  |  |  |  |  |

---

## Final interpretation rules

### Case A: Pass@1 improves
Conclusion:
- self-repair training improves underlying model capability

### Case B: Pass@1 unchanged, but repair-mode metrics improve
Conclusion:
- self-repair training improves repair behavior, but not first-pass capability

### Case C: neither improves
Conclusion:
- current self-repair training does not provide useful gains

### Case D: first-pass drops, repair improves
Conclusion:
- self-repair training may bias the model toward correction behavior without improving direct formulation ability

---

## Minimal success criterion

To claim that self-repair training improves model capability, the following must hold:

1. diagnosis-guided repair is better than coarse repair
2. self-repair-trained model is better than baseline on **first-pass Pass@1**
3. the gain is consistent on held-out benchmarks

If only repair-mode performance improves, then the conclusion should be weaker:

**self-repair training improves recovery ability, but not base one-shot capability**

---

## Recommended order

1. Build repair dataset
2. Verify diagnosis-guided repair is useful
3. Train self-repair SFT model
4. Evaluate first-pass capability
5. Evaluate repair-mode capability
6. Decide whether self-repair is a true capability improvement or only a workflow improvement