# Phase 4：数据驱动的优化方向分析

> 日期：2026-07-30
> 状态：种子已固定（`--seed` 参数），训练可复现
> 基线：E12/E13（~70.03 A榜 / 69.43 B榜），E7（70.77）已不可复现
> 截止：2026-08-01 12:00

---

## 一、从已完成的实验反推瓶颈

### 1.1 实验路径回顾

```
Phase 3.1 RoBERTa升级     → E5 70.25 (+0.57)  ✅ 有效
Phase 3.2 框架特征注入     → E7 70.77 (+0.52)  ✅ 有效，Task3 +1.31
Phase 3.3 AEDA数据增强     → E9-E11 全退化     ❌ 中文标点破坏span精度
Phase 3.4 框架约束解码     → E12b +0.02        ❌ 82.7%错误是合法但选错
Phase 3.5 NoisyTune消融   → E14 -0.77 vs E7    ❌ NT是有效正则化，关掉反而降
```

### 1.2 五条关键数据线索

**线索一 — E13 的放大效应**

```
Task2 NT OFF → Task2 F1 +0.07 → Task3 F1 +0.46  (Task3 模型权重未变)
```

Task2 质量的微小提升被 Task3 放大了 **6.6 倍**。Task3 的完全匹配评测（span + role 都对才算对）意味着：Task2 边界差一个字，Task3 角色就算对也全错。Task2 recall 仅 85.51%（dev），**14.5% 的金标论元从未被 Task3 看到**。

**线索二 — 训练/推理的输入分布不匹配**

```
Task3 训练时输入 = gold span（100% 正确，来自 cfn_spans）
Task3 推理时输入 = Task2 预测 span（precision 80.61%，19% 假阳性）
```

Task3 从未在训练中见过「假论元」。Task2 产出的假阳性 span 对 Task3 来说是 OOD 输入，但模型仍然会给它们分配高分角色——CrossEntropy 从未教过模型「拒绝」。这直接贡献了 Task3 precision 只有 53.58%。

**线索三 — 三个任务的训练强度不对等**

| | Task 1 | Task 2 | Task 3 |
|--|:--:|:--:|:--:|
| Epochs | 10 | 5 | 10 |
| FGM | ❌ 注释 | ✅ 启用 | ❌ 注释 |
| NoisyTune | ✅ | ✅ | ✅ |
| Scheduler | Linear | Linear | Linear |
| 训练 eval best epoch | 4/10 | 3/5 | 5/10 |

FGM 的代码已经写好了，就在 `train_task1.py` 和 `train_task3.py` 里，只是被注释掉。Task 2 已经验证了 FGM 在 PyTorch 1.13 环境下有效。所有任务用的都是 linear warmup + linear decay，而 cosine annealing 是已知更好的调度策略。Task 1/3 在 epoch 4-5 已达 best，后面在 plateau 上空跑。

**线索四 — Task3 训练 78.86% vs 推理 56.81%，差距 22pp**

| 场景 | Task3 指标 | 说明 |
|------|:--:|------|
| 训练 eval（gold span, gold frame）| 78.86% accuracy | 模型能区分角色 |
| dev pipeline（Task2 span, Task1 frame）| 56.81% F1 | 级联误差吃掉 22pp |

这 22pp 的 gap 来自三个源头：
1. Task2 提供的 span 有边界误差（precision 80.61%, recall 85.51%）
2. Task1 提供的 frame 有 27.3% 错误率
3. 完全匹配评测（exact match）比 accuracy 更严格

**线索五 — Top-3 覆盖率 87.43% vs E12b 零收益**

```
Task1 top-1 frame accuracy:  72.70%
Task1 top-3 frame coverage:  87.43%  (+14.73pp)
Task1 top-5 frame coverage:  90.39%
```

E12b 用了 top-3 但只涨了 0.02。不是 top-K 没用，而是 **K 次独立 forward + max 融合**的方式太粗暴——K 个框架各自独立投票，互相不知道对方的存在。正确的融合应该让 Task3 在**一次 forward 中感知所有候选框架**，但 E12b 没做到。

---

## 二、优化方向

### 方向 1：训练系统化提升（改动小，一次重训完成）

**FGM 全开 + Cosine 调度 + 延长 epoch**

| 改动 | 文件:行号 | 操作 |
|------|:---:|------|
| Task 1 FGM | `train_task1.py:151-155` | 取消注释 |
| Task 3 FGM | `train_task3.py:154-160` | 取消注释 |
| Cosine annealing | 三个 `train_taskX.py` 的 scheduler | 替换 warmup_linear |
| Task 1 epochs | `params.py` 默认值 | 10→15 |
| Task 3 epochs | `params.py` 默认值 | 10→15 |

FGM 是已知有效的对抗训练方法（已在 Task 2 验证），参数扰动范围 epsilon=1.0 作用于 embedding 层。Cosine annealing 相比 linear decay，在训练后期给模型更多「重新探索」的机会，特别适合 Task 1/3 这种 epoch 4-5 已达 best 但后续 plateau 的情况。

**预期收益**：Task1 +0.5~1.0 Acc, Task3 +0.5~1.5 F1 → 总分 +0.5~1.5

---

---

### 方向 2：Task3 引入「非论元」类别（改动中等，单独重训 Task3）

核心思路：让 Task3 学会拒绝。

```
当前标签空间: 1009 个正类（所有可能的 FE 名称）
新增标签:     第 1010 类 = "None"（不是论元）

训练时:
  正样本 = gold span + gold role（和现在一样）
  负样本 = 从句子中随机抽取的假 span + "None" 标签
  每个 batch 混入 1-2 个负样本

推理时:
  Task3 预测角色
  如果 top-1 = "None" 或 confidence < 阈值 → 丢弃该 span
```

**改动范围**：
- `dataset_task3.py`：新增负采样逻辑
- `model_task3.py`：num_labels = 1010（改 init 的 config.num_labels + 1）
- `predict_task3.py`：过滤 "None" 预测

**为什么值得做**：直接针对 Task3 precision=53.58% 的核心原因——假阳性 span 被分配了不该有的角色。如果把 Task2 的 19% 假阳性过滤掉一半，Task3 precision 可提升 8-10pp。

**预期收益**：Task3 F1 +1.0~2.0 → 总分 +0.4~0.8

---

### 方向 3：多 seed Ensemble（最稳，可并行训练）

种子已固定（`--seed` 参数），训练可复现。用 2-3 个不同 seed 分别训练三任务，预测时投票/平均：

```
Task 1: soft voting（多个 checkpoint 的 logits 取平均 → argmax）
Task 2: span 投票（每个预测 span 统计被几个模型预测到，> 半数保留）
Task 3: soft voting（同 Task 1）
```

三个 seed 可以**并行训练**（先后跑或三张卡）。单 seed 训练约 10h（三任务串行），并行后总时间约 10-12h。

**预期收益**：+0.3~0.8 总分

---

### 方向 4：Task3 的 span 表征增强（架构改动，留给 PPT 做 future work）

当前 Task3 只取了 `logits[i, :, span_start, span_end]` 这一个格子的分数——只用了 span 的两个端点。但 GlobalPointer 产出了完整的 `(num_labels, seq_len, seq_len)` 矩阵。

可以不只是取 `(start, end)` 这一个格子，而是：
- 对整个 span 内部所有 token 做 soft attention 聚合
- 或者取对角线上的若干点（span 内部的子结构）

这个方向实现成本较高，但作为报告中的「未完成的优化方向」是很好的素材。

---

## 三、已否决方向

### 否决 1：Task3 训练时注入框架噪声

**来源**：原方向 2。当前 Task3 训练用 gold frame（100% 正确），推理用 Task1 预测（72% 正确）。训练/推理存在 frame 质量 gap。假设在训练时以 15% 概率将 frame_ids 替换为随机值，迫使模型不完全依赖 frame_id。

**否决依据**：AEDA 实验（Phase 3.3）提供了直接反证。

```
AEDA 的本质 = 往训练数据注入随机噪声，期望模型变得更鲁棒

E9 (AEDA + 半epoch):  70.23  vs E7 70.77  → -0.54
E11 (AEDA + 全epoch): 69.86  vs E7 70.77  → -0.91
```

AEDA 在任何 epoch 设置下均退化。实验记录结论：

> AEDA 对 CFSP 任务无效。中文标点噪声破坏了 span 定位精度，且分类任务已充分收敛，不需要额外正则化。

frame 噪声和 AEDA 是同一种思路——往训练信号里注入随机性，指望模型学会忽略噪声。但 AEDA 的结果说明：**GlobalPointer + CrossEntropy 架构对这种扰动不是变得更鲁棒，而是直接被带偏。** 15% 的错误 frame_id ≈ 15% 的训练样本给了错误的 FrameEmbedding 偏置方向，噪声淹没了信号。

**判决**：❌ 不执行。

### 否决 2：Task2 阈值面向 pipeline 调优

**来源**：原方向 3。假设当前 Task2 使用的固定阈值 0 是为最大化 Task2 F1 设的，对下游 Task3 未必最优。计划网格搜索不同阈值，选 pipeline 总分最高的。

**否决依据**：已检查过 Task2 logits 分布，阈值 0 已经能很好地区分正负 span，logits 分布在 0 两侧有明显分离。调低阈值引入大量假阳性（Task3 precision 本已只有 53.58%），调高阈值漏掉真论元（直接拉低 Task3 recall）。当前阈值恰好位于分离边界上。

**判决**：❌ 不执行。

---

## 四、优先级与执行计划

```
Step 1 ── 方向 2: Task3 "非论元" 类别 ─────────────
  代码: 中等改动（dataset + model + predict）
  验证: 先训 1-2 epoch 看 dev loss 和 None 类区分度
  全量: 信号明确则完整训练 ~2.5h
  预期: +0.4~0.8 总分

Step 2 ── 方向 1: FGM全开 + Cosine + 多epoch ─────
  代码: 5行（取消注释 + 替换scheduler）
  训练: 三任务完整训练 ~10h
  预期: +0.5~1.5 总分

Step 3 ── 方向 3: 多 seed Ensemble ──────────────
  训练: Step 1+2 完成后，换 seed 重训 2 个版本
  预期: +0.3~0.8 总分（无论前两步结果如何都做）
```

- 方向 2 先做快速验证（1-2h），有正向信号再全量；无信号立刻毙掉转方向 1
- 方向 1 和方向 2 需要各自重训 Task3，但方向 2 若验证失败则跳过
- 方向 3 是兜底——即使前两步都不涨，ensemble 也几乎肯定涨

---

## 五、各方向与现有实验的关系

| 方向 | 依赖哪个已有工作 | 冲突吗 |
|------|------|:--:|
| 方向 1 (FGM) | Phase 3.2 (FrameEmbedding)、NoisyTune | 不冲突，FGM 和 NT 互补 |
| 方向 2 (非论元类别) | Phase 3.2 | 标签空间 +1，FrameEmbedding 输出维度 +1 |
| 方向 3 (ensemble) | 种子方案 | 依赖种子固定 |
| 方向 4 (span表征) | GlobalPointer | Architecture change |

---

## 六、风险与应对

| 风险 | 概率 | 应对 |
|------|:--:|------|
| FGM 在 Task1/3 上不涨反退 | 低 | Task2 已验证 FGM 有效，Task1/3 同架构、同训练范式，退化概率很低 |
| Cosine annealing 对短 epoch 任务（Task2=5ep）效果有限 | 中 | Task2 可保持 linear，只改 Task1/3 |
| Ensemble 只涨 0.1-0.2，ROI 低 | 低 | 已知可靠的提分手段，尤其对不同 seed 的独立模型 |
| 方向 2（非论元类别）改动大，不收敛 | 中 | 先做小规模验证（1 epoch），确认 loss 下降正常再全量训练 |
