# Structure-Aware Reward Experiment for OR-R1

## Project Objective

This project extends OR-R1 by adding a structure-aware reward into the existing TGRPO training pipeline.

The central hypothesis is:

OR-R1 already rewards output format correctness, code executability, and outcome agreement, but it does not explicitly reward whether the generated optimization model is structurally correct. Adding structure-aware reward should improve one-shot formulation reliability, reduce structural modeling errors, improve feasibility, and improve Pass@1.

This project should preserve the original OR-R1 pipeline as much as possible, and only add one new reward family plus its verifier.

---

## Baseline Assumption

Assume the existing OR-R1 training pipeline already contains the following reward components:

- `r_fmt`: format reward
- `r_code`: valid code reward
- `r_vote`: majority-voting reward

The new total reward should be:

`r_total = λ_fmt * r_fmt + λ_code * r_code + λ_vote * r_vote + λ_struct * r_struct`

Where `r_struct` is the new structure-aware reward.

Do not replace the OR-R1 reward system. Only augment it.

---

## Main Idea

The new reward should evaluate whether the generated optimization model is structurally correct, not just whether:

- the output format is correct,
- the code executes,
- or the final numerical answer matches other samples.

The structure-aware reward should be decomposed as:

`r_struct = α_obj * r_obj + α_var * r_var + α_con * r_con + α_align * r_align`

Where:

- `r_obj`: objective polarity correctness
- `r_var`: variable semantics correctness
- `r_con`: constraint coverage correctness
- `r_align`: parameter-entity alignment correctness

The first implementation should be simple, rule-based, and easy to debug. Do not start with a learned reward model.

---

## Reward Component Definitions

### 1. Objective Reward

Goal: check whether the generated model uses the correct optimization direction.

Examples:
- "minimize total cost" should map to minimization
- "maximize total profit" should map to maximization

Minimal implementation:
- return `1.0` if objective direction is correct
- return `0.0` otherwise

Optional softer implementation:
- `1.0` if objective exists and direction is correct
- `0.3` if objective exists but direction is wrong
- `0.0` if objective is missing or unparsable

Store this score as `r_obj`.

---

### 2. Variable Reward

Goal: check whether decision variables are correctly defined.

Important aspects:
- variable existence
- variable type: binary / integer / continuous
- variable indexing dimension: e.g. `x[i]`, `x[i,j]`, `x[i,j,t]`
- optional domain constraints such as nonnegative

Typical failure modes:
- binary variable generated as continuous
- missing time index
- wrong index structure
- missing decision variable entirely

Minimal implementation:
compute average match over variable slots. Each variable slot can include:
- semantic role
- type
- index arity

Suggested scoring:
- full match = `1.0`
- partial match = `0.5`
- mismatch = `0.0`

Store this score as `r_var`.

---

### 3. Constraint Reward

Goal: check whether key constraints mentioned in the problem are present in the generated model.

This is the most important structure-aware reward.

At the first stage, do not require exact symbolic equivalence. Only check key constraint category coverage.

Constraint categories may include:
- capacity
- demand satisfaction
- flow balance
- assignment uniqueness
- budget
- inventory conservation
- non-negativity
- integrality
- resource limit
- time consistency

Minimal implementation:
`r_con = (# covered key constraints) / (# gold key constraints)`

Example:
- if the gold problem contains 4 key constraints
- and the generated model covers 3 of them
- then `r_con = 0.75`

Store this score as `r_con`.

---

### 4. Alignment Reward

Goal: check whether parameters and entities are mapped correctly.

Examples:
- supply belongs to factories, not customers
- demand belongs to customers
- cost should be indexed by `(factory, customer)`
- inventory should align with time-indexed state variables if applicable

Typical failure modes:
- swapped indices
- assigning demand to the wrong entity
- cost/revenue confusion
- wrong entity-parameter binding

Minimal implementation:
`r_align = (# correctly mapped relations) / (# total gold relations)`

Store this score as `r_align`.

---

## Intermediate Schema Representation

To compute structure-aware reward, normalize both the problem statement and the generated model into the same schema representation.

Use a schema like the following:

```yaml
ProblemSchema:
  objective: minimize_cost
  entities:
    - factories
    - customers
  variables:
    - name: ship
      indices: [factory, customer]
      type: continuous
      domain: nonnegative
  constraints:
    - supply_capacity(factory)
    - demand_satisfaction(customer)
  parameters:
    - supply[factory]
    - demand[customer]
    - cost[factory, customer]