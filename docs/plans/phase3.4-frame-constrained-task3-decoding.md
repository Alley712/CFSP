# Phase 3.4：框架约束的 Task 3 角色解码方案

> 日期：2026-07-29  
> 基线：E7/E8（RoBERTa-wwm-ext + Task3 框架特征注入）  
> 目标：不依赖 FGM、EMA、学习率调度等训练技巧，利用 CFSP 任务结构本身提升 Task 3 指标

---

## 一、背景与动机

当前项目已经完成以下关键实验：

| 实验 | 方案 | A 榜总分 | B 榜总分 | 结论 |
|------|------|:------:|:------:|------|
| E3/E4 | BERT baseline 复现 | 69.68 | 68.80 | 成功复现官方 baseline |
| E5/E6 | RoBERTa-wwm-ext | 70.25 | 69.30 | 预训练模型升级有效 |
| E7/E8 | RoBERTa + Task3 框架特征注入 | 70.77 | 69.46 | Task3 框架信息有效 |
| E9/E10/E11 | AEDA 数据增强 | 70.23 / 69.86 | 69.35 | 数据增强退化，放弃 |

原计划继续进行训练技巧调优，例如 FGM 全开、EMA、NoisyTune 变体、学习率调度等。但这些方法偏通用，不能直接解决 CFSP 当前最突出的结构性问题：

**Task 3 的角色预测没有充分利用框架与角色之间的合法从属关系。**

根据数据统计：

- 框架数：713
- Task 3 全局唯一角色名：1009
- frame_info.json 中 FE 条目总数：30472
- 每个框架平均 FE 数：约 42.7

这意味着 Task 3 当前面对的是一个非常大的全局标签空间。即使 Phase 3.2 已经加入 frame embedding，模型仍然是在 1009 个角色中打分，只是额外获得了一个 soft bias。新方案进一步把 `frame_info.json` 中的结构知识用于解码阶段，减少不合法角色的竞争。

---

## 二、核心方案

方案名称：

**框架约束的 Task 3 角色解码：Frame-Constrained Role Decoding**

核心思想：

1. Task 1 不只输出 top-1 框架，而是输出 top-k 框架及其分数。
2. Task 3 对每个 Task 2 预测出的 span，在多个候选框架下分别计算角色 logits。
3. 利用 `frame_info.json` 构建 `frame -> legal FE roles` 映射。
4. 对每个候选框架，只允许该框架下合法角色参与竞争。
5. 融合 Task 1 框架分数和 Task 3 角色分数，得到最终角色预测。

整体流程：

```text
句子 + 目标词
    ↓
Task 1 输出 top-k 框架及分数
    ↓
Task 2 输出候选论元 span
    ↓
Task 3 在每个候选框架下预测角色
    ↓
frame_info.json 合法角色约束
    ↓
融合框架分数与角色分数
    ↓
输出 [sentence_id, span_start, span_end, role_name]
```

---

## 三、为什么不直接使用 Top-1 硬约束

最直接的做法是：Task 1 预测哪个框架，就只允许 Task 3 输出该框架下的角色。

但这个方案风险较高。当前 Task 1 准确率约为 71%-72%，如果 top-1 框架预测错误，正确角色可能会被合法角色 mask 直接屏蔽，导致 Task 3 无法恢复。

因此推荐使用 **Top-K 框架边际化**：

- top-1 框架正确时，可以像硬约束一样缩小角色空间；
- top-1 框架错误但 top-k 中包含正确框架时，Task 3 仍有机会恢复；
- 可以通过 Task 1 置信度控制每个框架对最终角色的影响。

建议先尝试：

| 参数 | 候选值 |
|------|--------|
| K | 3, 5 |
| alpha | 0.5, 1.0, 2.0 |
| 非法角色 mask | -10 或 -inf |

---

## 四、解码公式

对每个候选论元 span，最终角色分数可以定义为：

```text
score(role) =
  logsumexp over frame_k:
    task3_logit(role | frame_k)
    + alpha * task1_log_prob(frame_k)
    + legal_mask(frame_k, role)
```

其中：

- `task3_logit(role | frame_k)`：Task 3 在候选框架 `frame_k` 条件下给出的角色分数。
- `task1_log_prob(frame_k)`：Task 1 对候选框架的置信度。
- `alpha`：框架置信度权重。
- `legal_mask(frame_k, role)`：如果该 role 属于该 frame，取 0；否则取 `-inf` 或一个很小的负数。

最终输出：

```text
role* = argmax_role score(role)
```

如果实现上暂时不方便做 `logsumexp`，可以先用更简单的 max 融合：

```text
score(role) =
  max over frame_k:
    task3_logit(role | frame_k)
    + alpha * task1_log_prob(frame_k)
    + legal_mask(frame_k, role)
```

max 融合更容易实现，适合作为第一版。

---

## 五、实施路径

### E12a：Top-1 框架硬约束

目标：快速验证合法角色约束是否有效。

改动范围：

- 修改 `predict_task3.py`
- 读取 Task 1 输出的 top-1 frame
- 从 `frame_info.json` 构建合法角色集合
- 对不属于该 frame 的角色 logits 做 mask

优点：

- 不需要重新训练
- 实现成本最低
- 可以快速判断约束是否提升 Precision

风险：

- 受 Task 1 top-1 错误影响较大
- 可能提升 Precision，但降低 Recall

### E12b：Top-K 框架边际化

目标：作为主实验方案，在引入约束的同时缓解 Task 1 级联错误。

改动范围：

- 修改 `predict_task1.py`，额外输出每条样本 top-k 框架及分数
- 修改 `predict_task3.py`，读取 top-k 框架结果
- 对每个 span 构造多个 frame_id 条件输入
- 使用合法角色 mask 和融合公式输出最终角色

推荐配置：

```text
K = 3
alpha = 1.0
illegal_mask = -10
fusion = max
```

如果 E12b 有提升，再尝试：

```text
K = 5
alpha = 0.5 / 2.0
fusion = logsumexp
illegal_mask = -inf
```

### E12c：训练期加入框架角色 Mask

目标：让训练和推理阶段的约束一致。

改动范围：

- 修改 `model_task3.py` 或 `train_task3.py`
- 训练时根据 gold frame 对非法角色 logits 加 mask
- 使用 masked cross entropy 训练 Task 3

优点：

- 模型学习目标更贴合推理约束
- 理论上比只改解码更充分

风险：

- 需要重新训练 Task 3
- 如果预测阶段 Task 1 框架错误，模型可能更依赖 frame，从而放大级联误差

因此建议先做 E12a/E12b，再决定是否进入 E12c。

---

## 六、与现有 E7/E8 的关系

E7/E8 已经在 Task 3 中加入：

```text
FrameEmbedding(713 -> 256) + Linear(256 -> 1009)
```

并将 frame bias 加到 Task 3 角色 logits 上。

新方案不是替代 E7/E8，而是在其基础上继续利用框架信息：

| 方案 | 框架信息使用方式 | 约束强度 |
|------|------------------|----------|
| E7/E8 | frame embedding 产生 soft bias | 软约束 |
| E12a | top-1 frame 合法角色 mask | 强约束 |
| E12b | top-k frame 合法角色 mask + 分数融合 | 中强约束 |
| E12c | 训练期 + 推理期均加入合法角色约束 | 最强 |

推荐优先做 E12b，因为它在收益和风险之间最平衡。

---

## 七、预期收益

保守估计：

| 场景 | Task3 F1 提升 | 总分提升 | 说明 |
|------|:-----------:|:------:|------|
| 乐观 | +2.0 ~ +2.5 | +0.8 ~ +1.0 | top-k 覆盖率高，非法角色竞争明显减少 |
| 中性 | +0.8 ~ +1.5 | +0.32 ~ +0.6 | 约束提升 Precision，Recall 基本持平 |
| 悲观 | -0.3 ~ +0.5 | -0.12 ~ +0.2 | Task1 错误较多，mask 误伤正确角色 |

判断该方案是否值得继续推进的关键指标：

1. Task 1 top-k 框架覆盖率是否明显高于 top-1。
2. Task 3 Precision 是否因非法角色过滤而提升。
3. Task 3 Recall 是否没有明显下降。
4. A/B 榜提升是否一致。

---

## 八、验证实验

建议先在 dev 集或本地可控数据上做以下验证：

### 1. Task 1 Top-K 覆盖率

统计 gold frame 是否出现在 top-k 预测中：

```text
top1_acc
top3_acc
top5_acc
```

如果 top-3/top-5 覆盖率明显高于 top-1，说明 E12b 有实际空间。

### 2. 合法角色命中率

在 dev 集上统计 gold role 是否属于：

```text
Task1 top-1 frame 的合法 FE 集
Task1 top-3 frames 的合法 FE 并集
Task1 top-5 frames 的合法 FE 并集
```

该指标可以直接评估 mask 误伤风险。

### 3. Task 3 对照提交

至少形成以下对照：

| 实验 | 说明 |
|------|------|
| E7/E8 | 当前最佳基线 |
| E12a | top-1 hard mask |
| E12b-k3 | top-3 mask + max 融合 |
| E12b-k5 | top-5 mask + max/logsumexp 融合 |

---

## 九、风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| Task 1 top-1 错误导致正确角色被屏蔽 | Task3 Recall 下降 | 使用 top-k，不依赖 top-1 |
| 某些 FE 名称跨框架共享，mask 收益有限 | 提升不明显 | 融合 Task3 原始分数，避免完全依赖框架 |
| top-k 推理增加耗时 | 预测更慢 | K 先取 3；Task3 只在预测阶段增加少量 batch |
| `frame_info.json` 中 FE 名称和 Task3 标签映射不一致 | mask 错误 | 统一从同一个 `frame_info.json` 构建 role2idx 和 frame2roles |
| A 榜提升但 B 榜不稳定 | 泛化风险 | A/B 分别提交，对照 E7/E8 判断跨榜一致性 |

---

## 十、推荐结论

推荐将 Phase 3.4 从通用训练技巧调优调整为：

**E12：Top-K 框架约束的 Task 3 角色解码。**

优先级：

1. 先做 E12a，快速确认 top-1 合法角色 mask 是否有正向信号。
2. 主做 E12b，用 top-3 或 top-5 框架边际化降低级联误差。
3. 若 E12b 有稳定提升，再考虑 E12c，把框架角色约束加入 Task 3 训练期。

该方案更贴合 CFSP 任务本身，也更适合作为报告中的方法创新点：它不是简单堆训练技巧，而是显式利用 FrameNet 中“框架—框架元素”的结构知识，对 Task 3 的标签空间进行语义约束。
