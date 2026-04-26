# OR-R1 两个增强方案说明

## 1. 文档目的

这份文档把我们在 OR-R1 上做过的两个增强方向统一说明清楚：

1. `structure reward`
2. `self-repair`

重点不是再复述结果，而是把下面几件事讲清楚：

- 两个方案各自的原理
- 它们分别插在 OR-R1 的哪一层
- 真实训练过程到底怎么跑
- 训练和实验中实际遇到了什么问题

相关已有总结可参考：

- [report_experiment_summary.md](report_experiment_summary.md)
- [self_repair_exp_report.md](output/self_repair_exp_report.md)
- [self_repair_exp_v2_report.md](output/self_repair_exp_v2_report.md)
- [or_r1_flow_trace_report.md](output/or_r1_flow_trace_20260417/or_r1_flow_trace_report.md)

## 2. OR-R1 原始训练流程

原始 OR-R1 主流程可以压缩成：

```text
SFT -> GRPO/TGRPO -> merge -> evaluation
```

### 2.1 SFT

SFT 读取的是带监督的 `prompt/completion` 数据，典型数据在：

- [OR-Instruct-Data-3K](datasets/OR-Instruct-Data-3K)

训练时：

- `prompt` 是题目
- `completion` 是高质量数学建模文本 + `coptpy` 代码
- 训练时把 `prompt + completion` 拼成一条序列
- `prompt` 部分 label 被 mask 成 `-100`
- loss 只打在 `completion` token 上

也就是说，SFT 学的是：

```text
problem -> full solution
```

### 2.2 GRPO

GRPO 读取的是 `question/answer` 数据，典型数据在：

- [train_100.jsonl](datasets/trainset/train_100.jsonl)

这里没有 gold completion。流程是：

1. 把 `question` 包成 prompt
2. 模型一次采样多个 completion
3. 每个 completion 做格式检查、代码执行、答案提取
4. 基于 reward 更新模型

原始 reward 在：

- [02_grpo_train.py](02_grpo_train.py)

原始 reward 组成是：

```text
r_total = r_fmt + r_code + r_vote
```

其中：

- `r_fmt`：输出格式是否完整
- `r_code`：代码是否可执行、是否能得到 `model.objval`
- `r_vote`：同一题的多个采样结果中，是否与组内多数答案一致

这套机制的一个重要现实问题是：如果 completion 经常只写出格式壳子、不能形成可执行代码，训练侧 reward 就会退化成主要看格式分。这点在：

- [or_r1_flow_trace_report.md](output/or_r1_flow_trace_20260417/or_r1_flow_trace_report.md)

里已经有真实 trace。

## 3. 方案一：Structure Reward

### 3.1 原理

这个方案的出发点是：原始 OR-R1 的 reward 更看重

- 格式对不对
- 代码能不能跑
- 多个采样是否一致

但它不直接奖励“数学建模结构本身是否合理”。所以我们补了一个新的结构奖励项：

```text
r_total = r_fmt + r_code + r_vote + lambda_struct * r_struct
```

实现文件：

- [structure_reward.py](structure_reward.py)
- [02_grpo_train_struct.py](02_grpo_train_struct.py)

### 3.2 它加在训练流程的哪里

只加在 **GRPO/RL** 这一层，没有改原始 SFT 数据和原始 SFT 脚本。

也就是说，当前真实流程是：

```text
原始 SFT -> 带 structure reward 的 GRPO -> merge -> eval
```

而不是：

```text
改过的 SFT -> 改过的 GRPO
```

这样做的原因是控制变量。先只改 reward，便于判断 reward 设计本身是否有效。

更具体地说，`structure reward` 不是单独训练一个 verifier，也不是额外训练一个 schema 预测器，而是直接插进现有 `reward_with_reference(...)` 这条 reward 链里。

原始 OR-R1 的 GRPO 一条样本的训练流转是：

```text
question
-> prompt
-> model sample K 个 completions
-> 每个 completion 执行 reward 验证
-> 得到每个 completion 的 total reward
-> GRPO 根据 reward 更新模型参数
```

加入 structure reward 后，变化只在“reward 验证”这一步：

```text
question
-> prompt
-> model sample K 个 completions
-> 每个 completion:
     r_fmt
     r_code
     r_vote
     r_struct
-> total reward = r_fmt + r_code + r_vote + lambda_struct * r_struct
-> GRPO 根据新的 total reward 更新模型参数
```

所以它“教模型学习”的方式不是监督学习，而是：

- 如果某个 completion 的结构更像合理建模，`r_struct` 更高
- 该 completion 在 RL 中就更容易被提高概率
- 反过来，结构明显不合理的 completion 概率会被压低

也就是说，它是通过 **改变 completion 级别的 reward**，间接改变模型的生成偏好。

### 3.2.1 structure reward 的训练数据是怎么构建的

这部分最容易误解成“是不是要再造一份新的 SFT 数据”。当前版本不是。

当前训练仍然直接使用原始 OR-R1 的 RL 数据文件，例如：

- `datasets/trainset/train_100.jsonl`

这类数据一条样本只有：

- `question`
- `answer`

其中：

- `question` 用来构造成 GRPO 的输入 prompt
- `answer` 主要用于记录和离线评估，不是直接提供标准 completion

所以 structure reward 不是通过“新增 labeled completion”来训练，而是通过原始 RL 采样流转来训练：

```text
question
-> 按 OR-R1 模板包装成 prompt
-> 当前策略采样 K 个 completions
-> 每个 completion 分别算原 reward 和 r_struct
-> 合成 total reward
-> GRPO 用这一组 reward 回传更新策略
```

因此，所谓“构建训练”的真实含义是：

- 不改数据集字段
- 不改监督标签
- 只改 completion 生成后的 reward 计算链

从第一性原理看，structure reward 训练的是：

```text
哪些 completion 更值得保留
```

而不是：

```text
标准答案文本应该长什么样
```

### 3.2.2 一条真实样本在训练里怎么流转

一条 GRPO 样本的真实流转，可以压成下面这条链：

```text
dataset row(question, answer)
-> build prompt
-> sample completion_1 ... completion_K
-> 对每个 completion:
     抽代码
     执行代码
     提取 prediction_answer
     计算 r_fmt / r_code / r_vote / r_struct
-> 得到一组 total rewards
-> GRPO 根据组内相对高低更新模型
```

这里 structure reward 只参与最后这一段：

```text
completion text -> completion_schema
question text -> problem_schema
schema match -> r_struct
```

它不参与代码执行，不参与求解器返回，也不替代原始的 `r_fmt / r_code / r_vote`。

### 3.3 新 reward 怎么计算

`r_struct` 不是由另一个 LLM 生成的，也不是再让模型跑一次题。它是基于规则的 schema matching。

流程是：

1. 从题目文本抽一个 `problem_schema`
2. 从 completion 文本抽一个 `completion_schema`
3. 比较两边的结构一致性

schema 提取本身不是模型推理，而是正则和关键词规则。主要字段包括：

- `objective`
- `variable_type`
- `variable_arity`
- `constraints`
- `alignments`

再拆成四个子分数：

- `r_obj`：目标方向是否一致
- `r_var`：变量类型和维度是否一致
- `r_con`：关键约束类别是否覆盖
- `r_align`：参数和实体关系是否基本对齐

默认约束项更重，所以：

```text
r_struct = weighted_avg(r_obj, r_var, r_con, r_align)
```

这里最容易误解的一点是：`problem_schema` 和 `completion_schema` 都不是“再跑一次模型”得到的。

- `problem_schema`：直接从题目文本抽
- `completion_schema`：直接从模型输出文本抽

两边都是规则抽取，不涉及额外采样。当前实现里，schema 提取主要靠：

- 正则匹配目标方向
- 关键词匹配变量类型
- 变量索引模式匹配维度
- 关键词集合匹配约束类别
- 正则/关键词匹配参数与实体关系

所以可以把它理解成：

```text
text -> heuristic schema parser -> schema object
```

不是：

```text
text -> LLM judge -> schema
```

### 3.3.1 它是怎么 verify 的

当前 `r_struct` 的 verify 是 **强规则字段匹配**，不是 learned reward model，也不是符号级数学证明。

验证逻辑可以理解成：

1. 题目里能不能看出这是 `maximize` 还是 `minimize`
2. 题目里能不能看出变量更像 `binary/integer/continuous`
3. 题目里提到的关键结构，比如 `capacity / demand / budget / flow balance`，completion 里有没有出现
4. completion 的变量维度、约束类别、目标方向，和题目里抽出来的 schema 是否大体一致

因此，它判断的是：

```text
completion 在结构层面是否“像对的”
```

而不是：

```text
completion 在数学上是否“严格正确”
```

这也是它和原始 OR-R1 最大的差别：

- 原始 OR-R1 更偏格式、执行和组内一致性
- structure reward 新增的是结构一致性

但这也解释了它的局限：

- `problem_schema` 自身可能抽错
- `completion_schema` 自身也可能抽不全
- 所以 `r_struct` 是启发式信号，不是绝对真值

### 3.3.2 reward 在代码里是怎么落地验证的

当前实现里，`r_struct` 的验证不是一次性黑盒打分，而是“先抽字段，再逐项比对”。

可以把它理解成 4 步：

1. `extract_problem_schema(question)`
   从题目文本里抽：
   - 目标方向
   - 变量类型/维度
   - 约束类别
   - 参数与实体关系

2. `extract_completion_schema(completion)`
   从模型输出里抽：
   - 有没有写出目标方向
   - 有没有出现二元/整数/连续变量表述
   - 有没有出现 capacity / demand / flow / assignment 这类结构
   - 有没有相应的参数和对象关系

3. 分项比较
   - `objective match`
   - `variable match`
   - `constraint overlap`
   - `alignment overlap`

4. 聚合成 `r_struct`
   按权重做加权平均，其中约束项默认更重

因此它的“verify”更接近：

```text
schema-level consistency check
```

不是：

```text
solver-backed correctness check
```

这一点必须和原始 OR-R1 分清楚：

- `r_code` 的 verify 是真执行代码
- `r_vote` 的 verify 是组内答案投票
- `r_struct` 的 verify 是文本结构规则匹配

所以三者的证据强度并不一样：

- `r_code` 属于执行证据
- `r_vote` 属于一致性证据
- `r_struct` 属于结构启发式证据

### 3.4 它和原始 OR-R1 的区别

原始 OR-R1 更偏：

- 格式正确性
- 代码可执行性
- 组内答案一致性

structure reward 新增的是：

- 建模结构看起来是否像对的

所以它补的是结构偏好，而不是标准答案监督。

### 3.5 真实实验中遇到的问题

1. `problem_schema` 和 `completion_schema` 都是规则抽取，不是 oracle  
它们可能抽错，所以 `r_struct` 本身是 noisy signal，不是严格 verifier。

2. reward 变了，不等于 final `pass@1` 一定提高  
structure reward 能改变模型行为，但未必能稳定转化成 end-task 提升。

3. benchmark 间有 trade-off  
在我们的实验里，一些设定能改善某个 hard set，但会伤另一个。

### 3.6 当前结论

当前版本说明：

- structure-aware reward 是一个有方向感的增强
- 但当前规则设计还不够稳
- 还没有带来稳定、可复现的 `pass@1` 提升

## 4. 方案二：Self-Repair

### 4.1 原理

这个方案的核心假设不是“多修一次就更好”，而是：

```text
如果模型能从失败输出和错误信号中学会修正，
这种修正能力可能反过来提升 first-pass 表现
```

原始 OR-R1 的普通 SFT 学的是：

```text
problem -> solution
```

self-repair 想额外学的是：

```text
problem + failed_output + diagnosis -> repaired_solution
```

### 4.2 它加在训练流程的哪里

当前版本不是在线 RL self-repair，也不是每轮训练后自动反思。

它是一个 **离线数据构建 -> 再做 SFT** 的流程：

```text
已有模型 first-pass
-> 收集失败样本
-> 诊断
-> 生成 repair
-> 执行筛选
-> 写成新的 self-repair SFT 数据
-> 训练一个新的 self-repair SFT / LoRA 分支
```

也就是说：

- self-repair 当前主要插在 **SFT 数据构建层**
- 不是直接插进原始 GRPO reward 里

从“模型怎么学习”这个角度看，self-repair 当前版本做的不是：

```text
每一轮 RL 后在线反思
```

而是：

```text
先离线造出一批 repair 监督样本
再把这些样本当成新的 SFT 数据做一次监督训练
```

所以它给模型学习的方式，是 **改 SFT 训练样本的输入输出映射**，而不是改原始 GRPO reward。

### 4.3 repair 是怎么生成的

repair 不是人工写的，而是模型二次生成的。

流程如下：

1. 用 base model 先答一遍

```text
problem -> initial_response
```

2. 执行 `initial_response` 中的代码

得到：

- `initial_execution_state`
- `initial_execution_best_solution`
- `stdout/stderr`

3. 基于失败结果做 diagnosis

例如识别成：

- `format/no_code`
- `execution/runtime_error`
- `execution/missing_variable`
- `solver/infeasible`
- `solver/unbounded`

同时附上：

- `diagnostic_tags`
- `likely_cause`
- `repair_instruction`

4. 把这些信息拼成 repair prompt，再让模型生成一版修复后的完整答案

```text
problem + failed_output + diagnosis + feedback
-> repaired_solution
```

5. 再执行 `repaired_solution`

只有 repair 后的结果真的更好，才会进入 SFT 数据集。

这里的 `repaired_solution` 不是人工写的标准答案，也不是从参考答案反推出来的，而是 **同一个 base model 在 repair prompt 条件下二次生成的结果**。

所以真实流程是：

```text
第一次生成：
problem -> failed_output

第二次生成：
problem + failed_output + diagnosis + feedback
-> repaired_solution
```

也就是说，repair 本身也是模型生成的，只不过它的输入条件更丰富。

### 4.3.1 repair 是怎么判断“值不值得学”的

不是所有 repair 都会进入训练集。当前流程会对 repair 结果再执行、再筛选。

第一版的筛选逻辑更严格，大致是：

- repair 后直接答对，收
- 或者 repair 后比 first-pass 更接近答案，且 solver 状态是 `optimal`，收

第二版改成了分层接收：

- `tier1_code_recovery`
  `no_code -> executable`
- `tier2_execution_fix`
  `runtime_error / missing_variable / syntax_error -> optimal`
- `tier3_model_fix`
  `infeasible / unbounded / non_optimal -> optimal`
- `tier4_correct_answer`
  最终数值直接答对

所以 self-repair 不只是“生成一版 repair”，而是：

```text
生成 repair
-> 执行 repair
-> 根据状态和数值改善决定是否入库
```

### 4.4 self-repair 的 SFT 样本长什么样

第一性原则上，它不是把原题再训练一遍，而是写成新的 `prompt/completion`：

- `prompt`
  - `problem`
  - `failed first-pass output`
  - `diagnosis`
  - `repair_instruction`
  - V2 里还会加 `quality_tier`

- `completion`
  - 修复后的完整数学建模文本
  - 修复后的完整 `coptpy` 代码

也就是说，输出不是“补丁建议”，也不是“只改几行 diff”，而是：

```text
full repaired solution
```

这里“给模型学习”的关键点在于，self-repair 改了 SFT 的输入语义，也改了监督目标的语义：

原始 SFT 学的是：

```text
problem -> solution
```

self-repair SFT 学的是：

```text
problem + failed_output + diagnosis -> repaired_solution
```

也就是说，模型不再只是学“怎么解题”，而是额外学一层：

```text
看到这种失败和这种错误信号时，应该如何修改
```

### 4.4.1 self-repair 样本里到底包含什么

当前 self-repair 的 `prompt` 不是只有题目，而是显式包含：

- 原题
- first-pass 失败输出
- `failure_type`
- `diagnostic_tags`
- `likely_cause`
- `repair_instruction`
- V2 中还可能有 `quality_tier`

`completion` 则是修复后的完整答案。

所以当前 self-repair SFT 的本质是：

```text
把“失败轨迹 + 诊断 + 修复后答案”写成监督样本
```

而不是：

```text
只把原题和最终正确答案重新训练一遍
```

### 4.5 第一版和第二版

第一版实现：

- [self_repair_pipeline.py](eval/self_repair_pipeline.py)

主要逻辑：

- first-pass 失败样本
- coarse repair
- diagnosis-guided repair
- 只有 repair 明显更好时才收进 SFT

问题是样本太少，最终只得到 1 条可训练样本。

第二版实现：

- [self_repair_pipeline_v2.py](eval/self_repair_pipeline_v2.py)

V2 做了四个关键增强：

1. stronger first-pass prompt
2. 更宽松的代码提取
3. repair route 分流为 `format / execution / modeling`
4. SFT 接收规则改成 tiered acceptance

V2 的收益是把大量 `no_code` 压下去了，但新的主失败变成了 `syntax_error / runtime_error`，最终还是没形成足够高质量的 repair SFT 样本。

### 4.6 真实训练中遇到的问题

1. 上游 repair 数据质量太差  
这是最核心问题。repair 如果大多还是错的，后面的 SFT 只是在学习噪声。

2. 大量失败其实是格式失败，而不是深层建模失败  
如果 code extraction 太严格，很多样本会被误归到 `no_code`。

3. diagnosis 和真实失败类型不总是对齐  
如果 diagnosis 错了，repair prompt 就不够针对。

4. execution route 更适合 patch-style 修补，不适合整段重写  
当前模型倾向于整段重写长代码，容易从 `no_code` 转移成 `syntax_error/runtime_error`。

5. LoRA + DeepSpeed + gradient checkpointing 的工程问题  
实验中出现过：

- `element 0 of tensors does not require grad`
- 关闭 checkpointing 后 OOM

最终通过新增训练入口解决：

- [01_sft_train_self_repair.py](01_sft_train_self_repair.py)

6. 更根本的问题是“repair 数据本身太弱”  
如果大部分 repair 结果没有明显优于 first-pass，那么即使 SFT 脚本本身能正常跑，训练学到的也仍然是噪声而不是稳定修复规律。

### 4.7 当前结论

当前 self-repair 的主要结论不是“这个想法逻辑不通”，而是：

- 现在的 repair 更像 noisy regeneration
- 还不是稳定的 targeted correction

在 repair 本身没有先变强之前，把这些轨迹拿去训 SFT，不会稳定提升 `pass@1`。

## 5. 两个方案在训练层面的根本区别

可以把它们并排理解：

### 5.1 Structure Reward

- 插在：GRPO/RL
- 作用：改 reward
- 目标：让模型偏向结构上更像正确建模的 completion
- 训练信号：sequence-level reward shaping

### 5.2 Self-Repair

- 插在：SFT 数据构建
- 作用：改训练样本形式
- 目标：让模型学会“看到失败 + 错因后如何修”
- 训练信号：supervised repair trajectories

所以：

- structure reward 更像“改偏好”
- self-repair 更像“补一种新能力映射”

## 6. 真实训练流程里的关键观察

### 6.1 原始 OR-R1

从真实 trace 看，原始训练里的 GRPO 很容易遇到：

- prompt 很长
- completion 不一定完整
- reward 退化成主要吃格式分

说明原始 RL 信号本身就不总是稳定。

### 6.2 Structure Reward

因为它是叠加在原始 reward 上，所以如果底层 completion 质量就不高，`r_struct` 往往只能提供一个弱补充信号。

### 6.3 Self-Repair

如果 first-pass 失败样本大多不是“可修的小错”，而是“大段缺代码 / 大段乱写”，那 repair 很难稳定产出高质量轨迹。此时 self-repair SFT 就没有足够的监督价值。

## 7. 当前阶段最合理的理解

这两个方向都不是无效方向，但都还没有走到“可稳定提升最终结果”的阶段。

更准确的理解是：

1. `structure reward`
   已经证明可以改变模型行为，但 reward 和最终成功率的对齐还不够紧。

2. `self-repair`
   已经证明“失败后再生成一版”这件事本身不难做，但 repair 数据质量还不足以变成有效训练监督。

## 8. 推荐阅读顺序

如果你后面还要继续看代码和实验，建议按这个顺序：

1. [or_r1_flow_trace_report.md](output/or_r1_flow_trace_20260417/or_r1_flow_trace_report.md)
   先看原始 OR-R1 真实训练流转

2. [report_experiment_summary.md](report_experiment_summary.md)
   再看 structure reward 的实验总结

3. [self_repair_exp_report.md](output/self_repair_exp_report.md)
   看 self-repair 第一版

4. [self_repair_exp_v2_report.md](output/self_repair_exp_v2_report.md)
   看 self-repair 第二版和瓶颈转移

5. [or_r1_small_batch_experiment.ipynb](notebooks/or_r1_small_batch_experiment.ipynb)
   自己跑小批量，逐步对照理解
