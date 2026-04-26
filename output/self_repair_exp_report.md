# Self-Repair Experiment Report

## 1. 实验目标

本次实验参照 `self_repair_exp.md`，验证一个核心问题：把 OR-R1 的单次输出改成“失败诊断 -> targeted repair -> 学习修复轨迹”后，是否能提升模型本身的 first-pass Pass@1，而不仅是多一次推理机会。

本次没有改动原始训练/评测源码，新增了独立实验脚本和 wrapper：

- `eval/self_repair_pipeline.py`
- `01_self_repair_sft_lora.sh`
- `01_sft_train_self_repair.py`

主要日志保存在 `output/self_repair_exp_logs/`，模型和中间产物保存在 `output/self_repair_exp_train100/`、`output/self_repair_sft_lora_train100/`、`output/full_self_repair_sft_lora_train100/`。

## 2. 做了什么

先用原始 OR-R1 localfull 模型在训练集前 99 条上做 first-pass 生成，收集失败样本，并对 32 条失败样本分别跑两种 repair：

- coarse repair：只给通用反馈，如 runtime error、infeasible、wrong answer。
- diagnosis-guided repair：加入结构化诊断，如 missing constraint、infeasible、runtime/API error、missing variable 等，并要求 targeted repair。

随后把 diagnosis repair 中真正修好或数值更接近答案的轨迹转成 SFT 数据。实际只得到 1 条可训练样本。为了完整闭环，仍然训练了一个最小 LoRA self-repair 分支，20 step，从 `output/sft_qwen3_8b_dir_3Ksample_1epoch` 出发，最后合并为 `output/full_self_repair_sft_lora_train100`。

训练过程中遇到两个问题并已按日志修复：

- DeepSpeed + LoRA + gradient checkpointing 报 `element 0 of tensors does not require grad`，新增 `01_sft_train_self_repair.py` 并在 PEFT 模型上启用 `enable_input_require_grads()`。
- 关闭 checkpointing 后显存 OOM，因此最终保留 checkpointing 并使用新增训练入口完成训练。

## 3. 结果

### Repair 数据构建

| Setting | Samples | Pass@1 | Optimal | Feasible | Improved |
|---|---:|---:|---:|---:|---:|
| Initial on train subset | 99 | 0.1616 | 0.3838 | 0.3838 | - |
| Coarse repair on failed subset | 32 | 0.0000 | 0.0625 | 0.0625 | 0.0000 |
| Diagnosis repair on failed subset | 32 | 0.0313 | 0.0938 | 0.0938 | 0.0313 |

diagnosis-guided repair 比 coarse repair 略好，但只多修对 1/32，SFT 样本也只有 1 条，数据质量和规模都不足以支撑强结论。

### First-Pass Pass@1

| Model | ComplexOR | IndustryOR |
|---|---:|---:|
| Original OR-R1 localfull | 0.4444 | 0.2800 |
| OR-R1 + self-repair LoRA | 0.4444 | 0.2700 |

first-pass 结果没有提升：ComplexOR 持平，IndustryOR 略降。

### Repair-Mode

| Model | Dataset | Initial Pass@1 | Repaired Pass@1 | Repair Improved |
|---|---|---:|---:|---:|
| Original OR-R1 localfull | ComplexOR full 18 | 0.4444 | 0.4444 | 0.0000 |
| OR-R1 + self-repair LoRA | ComplexOR full 18 | 0.4444 | 0.4444 | 0.0000 |
| Original OR-R1 localfull | IndustryOR first 20 | 0.7000 | 0.7000 | 0.0000 |
| OR-R1 + self-repair LoRA | IndustryOR first 20 | 0.7000 | 0.7000 | 0.0000 |

repair-mode 也没有提升，说明当前 self-repair 数据和训练方式没有带来可观的恢复能力增益。

## 4. 初步结论

本轮实验对应 `self_repair_exp.md` 中的 Case C：first-pass 没有提升，repair-mode 也没有提升。当前 self-repair 尝试不能证明模型能力提升。

主要原因不是 LoRA 训练失败，而是上游 repair 轨迹本身太弱：32 条失败样本里 diagnosis-guided repair 只成功 1 条，且大量失败仍停留在 no_code、infeasible、runtime error。这说明如果 self-repair 环节不能真正定位并修复“漏约束、变量定义错误、目标方向、可行性建模”等问题，后续 SFT 很难提升 pass@1。

下一步如果继续推进，建议先扩大并改进 repair 数据构建，而不是继续加训练步数：优先让诊断阶段产生更多“失败 -> 可行/正确”的高质量轨迹，再做 SFT 或 RL。
