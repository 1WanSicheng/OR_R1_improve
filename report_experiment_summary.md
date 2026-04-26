# OR-R1 Reproduction and Reward-Design Trial Summary

## 1. Objective

We first reproduced the main OR-R1 pipeline on the current machine, then explored whether adding a structure-aware reward could improve final `pass@1`, especially on harder formulation benchmarks.

The reproduction target was:

`Qwen3-8B -> SFT -> TGRPO -> merge -> evaluation`

Because the machine is `4 x RTX 4090 24G`, the RL stage used a memory-adapted setting rather than the exact paper-scale configuration.

## 2. What We Tried

### 2.1 Original OR-R1 pipeline

We reproduced the original training flow with:
- `SFT` on the 3K instruction data
- `TGRPO` on the local 4090-friendly configuration
- merged final model
- full `pass@1` evaluation on 8 benchmarks

### 2.2 New structure-aware reward

Without modifying the original training code, we added a new reward family:

`r_total = r_fmt + r_code + r_vote + lambda_struct * r_struct`

where:
- `r_struct = r_obj + r_var + r_con + r_align`
- `r_obj`: objective direction
- `r_var`: variable type / indexing
- `r_con`: key constraint coverage
- `r_align`: parameter-entity alignment

We ran two main variants:
- `struct 20-step`: first structure-reward version
- `struct v2 20-step`: reduced structure reward weight and larger LoRA capacity

We also kept a `baseline 20-step` run for comparison.

## 3. Main Results

### 3.1 Original OR-R1 reproduced result

Full 8-dataset `pass@1`:

| Dataset | pass@1 |
|---|---:|
| NL4OPT | 0.9391 |
| MAMO_EasyLP | 0.9034 |
| MAMO_ComplexLP | 0.4171 |
| IndustryOR | 0.2800 |
| NLP4LP | 0.8512 |
| ComplexOR | 0.4444 |
| OptiBench | 0.6380 |
| ICMLTEST | 0.8634 |

Average `pass@1`: `0.6671`

### 3.2 New attempts on key hard sets

| Method | IndustryOR | ComplexOR |
|---|---:|---:|
| Original OR-R1 | 0.2800 | 0.4444 |
| baseline 20-step | 0.3000 | 0.3889 |
| struct 20-step | 0.2900 | 0.3889 |
| struct v2 20-step | 0.2600 | 0.4444 |

## 4. Preliminary Conclusion

The original OR-R1 pipeline was successfully reproduced on this machine and remains the strongest overall result among all tested settings.

Our structure-aware reward trials did not produce a stable overall improvement:
- `baseline 20-step` slightly improved `IndustryOR`, but hurt `ComplexOR`
- `struct 20-step` did not improve either target set
- `struct v2 20-step` recovered `ComplexOR` to the original OR-R1 level, but reduced `IndustryOR`

So far, the new reward design changes the behavior of the model, but it has not yet translated into a robust `pass@1` gain.

## 5. Analysis

The current evidence suggests three points:

1. The structure-aware reward is not useless, but its benefit is dataset-dependent.  
   It seems to help some structured formulation cases, but it does not generalize consistently across benchmarks.

2. Reward-shape changes alone are not enough.  
   Even when training-side reward becomes higher, the final `pass@1` may stay flat or even drop, which means the reward is not yet aligned tightly enough with end-task success.

3. There is a clear trade-off between benchmarks.  
   The newer variants improved or recovered `ComplexOR`, but this came with weaker `IndustryOR`, suggesting the current reward terms or weights may over-bias certain formulation patterns.

## 6. Next Step

The next reasonable direction is not to keep changing many knobs at once. A cleaner next step is:

- keep the better LoRA setting fixed
- tune only `lambda_struct` and the most important structure term weights
- evaluate on the same target sets (`IndustryOR`, `ComplexOR`, optionally `MAMO_ComplexLP`)

This will make it easier to identify whether the bottleneck is reward weight, reward definition, or dataset mismatch.
