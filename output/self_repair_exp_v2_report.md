# Self-Repair Experiment V2 Report

## 1. 目标

本轮按第一性原理重排了 self-repair 方案，先优化上游监督质量，再判断是否值得训练。

V2 的核心假设是：

- 如果上游 repair 数据真的有信息增益，应该先看到：
  - `no_code` 明显下降
  - repair 后的 executable / optimal / correct 指标提升
  - 可用于 SFT 的轨迹数量明显增加

只有这三点成立，训练才有意义。

## 2. 本轮新增内容

本轮没有改动原始源码，只新增独立文件：

- `self_repair_exp_v2.md`
- `eval/self_repair_pipeline_v2.py`
- `01_self_repair_sft_lora_v2.sh`

主要改动：

1. stronger first-pass prompt，强制输出固定 section 和 fenced python code
2. 更宽松的代码提取：支持 `python_fence`、普通 fence、`section_recovery`
3. repair 分流为 `format` / `execution` / `modeling`
4. SFT 接收规则改为 tiered acceptance，而不是只收最终答对样本

日志：

- smoke: `output/self_repair_exp_logs/stage0_v2_smoke.log`
- 正式 100 条实验: `output/self_repair_exp_logs/stage1_2_v2_pipeline.log`

## 3. 结果

### 3.1 相对 V1 的主要变化

| Metric | V1 | V2 |
|---|---:|---:|
| Initial pass@1 | 0.1616 | 0.1616 |
| Initial optimal rate | 0.3838 | 0.3030 |
| Initial `no_code` count | 41 | 14 |
| Initial executable rate | 无该指标 | 0.8586 |
| Repair failed sample count | 32 | 48 |
| Diagnosis repair pass@1 | 0.0313 | 0.0000 |
| Diagnosis repair optimal rate | 0.0938 | 0.1042 |
| SFT examples | 1 | 0 |

### 3.2 V2 关键指标

| Setting | Samples | Pass@1 | Optimal | Executable |
|---|---:|---:|---:|---:|
| Initial | 99 | 0.1616 | 0.3030 | 0.8586 |
| Coarse repair | 48 | 0.0000 | 0.1042 | 0.7292 |
| Diagnosis repair | 48 | 0.0000 | 0.1042 | 0.7292 |

额外统计：

- `initial_extraction_modes`
  - `python_fence`: 44
  - `section_recovery`: 41
  - `none`: 14
- `diagnosis_route_counts`
  - `execution`: 27
  - `format`: 12
  - `modeling`: 9

### 3.3 失败模式变化

V2 的最大收益是把大量纯格式失败变成了“至少可提取/可执行的代码”：

- V1 的主失败是 `no_code`
- V2 把 `no_code` 从 41 降到 14

但新的主失败变成了：

- `syntax_error`
- `runtime_error`

在 48 条 repair 子集里，diagnosis repair 的最终状态分布是：

- `syntax_error`: 22
- `no_code`: 13
- `optimal`: 5
- `runtime_error`: 4
- `infeasible`: 2
- `non_optimal`: 2

route 级别上：

- `format` route 仍然主要落回 `no_code`
- `execution` route 主要落到 `syntax_error`
- `modeling` route 才有少量 `optimal`

## 4. 结论

本轮 V2 只解决了第一层问题：

- **格式入口显著改善**

但没有解决第二层更关键的问题：

- **模型虽然更愿意输出代码了，但代码正确性没有改善，反而大量转移成 syntax/runtime failure**

因此本轮没有形成任何可用于训练的高质量 repair 轨迹：

- `self_repair_sft_examples = 0`

基于这个结果，**本轮不继续训练 LoRA**。原因很直接：

1. diagnosis repair 没有优于 coarse repair
2. 没有产生可训练样本
3. 继续训练只会把噪声放大，不会提升 pass@1

## 5. 初步分析

这次实验说明，当前 bottleneck 已经从“有没有代码”转移到了“代码能不能写对”。

更具体地说：

1. 先前的问题主要是输出格式不稳定
2. V2 已经把这个问题压下去了
3. 但 repair 过程仍然倾向于重写长篇文本和长代码
4. 对 execution route 来说，真正需要的是局部代码修补，而不是整段重写

所以接下来的重点不应该是继续 SFT，而应该是继续把 repair supervision 拆细：

- 对 `format` route：强制只输出代码块，不要再写长篇数学解释
- 对 `execution` route：改成 patch-style repair，只修 import / 索引 / API / 变量定义
- 对 `modeling` route：保留完整重写

当前最重要的结论不是“self-repair 无效”，而是：

**现在的 repair still behaves like noisy regeneration, not targeted correction.**

在这一点没改掉之前，训练不会有稳定收益。
