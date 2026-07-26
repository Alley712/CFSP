# CFSP Baseline 代码逻辑分析

## 〇、总体架构概览

三个任务共享一套核心方法——**BERT + GlobalPointer**，但根据各自目标做了不同的适配。三个任务的代码结构完全一致，每个任务由四类文件组成：

| 文件 | 职责 |
|------|------|
| `params.py` | 公共超参数配置（学习率、batch size、BERT 路径等） |
| `model_taskX.py` | 模型定义（BERT 编码器 + GlobalPointer 头） |
| `dataset_taskX.py` | 数据加载与预处理（含特殊 token 插入、坐标偏移） |
| `train_taskX.py` | 训练循环（NoisyTune、FGM 对抗训练、warmup 调度） |
| `predict_taskX.py` | 推理与结果输出 |

**运行顺序**：先依次训练 Task1 → Task2 → Task3（参数保存至 `saves/`），再依次预测 Task1 → Task2 → Task3（结果写出至 `dataset/`）。Task3 预测依赖 Task2 的输出结果 `B_task2_test.json`。

---

## 一、核心技术：GlobalPointer 详解

三个模型的核心都是 **GlobalPointer**，这是一种用于 span 检测/分类的通用方法。理解它是理解所有三个任务的关键。

### 1.1 整体流程

```
BERT 编码 → Dense 投影 → 拆分为 Q/K → 加 RoPE 位置编码 → einsum 计算 span 得分矩阵
```

每一步的具体形状变化如下：

```
输入: input_ids (batch, seq_len)
         │
         ▼
      BERT 编码
         │  (Task1/3 用最后4层平均; Task2 用最后一层)
         ▼
   hidden_token (batch, seq_len, 768)
         │
         ▼
    Dense(768, num_labels × 64 × 2)
         │
         ▼
   outputs (batch, seq_len, num_labels × 128)
         │
         ▼ torch.split → stack
   (batch, seq_len, num_labels, 128)
         │
         ▼ 拆分最后维度
   qw (batch, seq_len, num_labels, 64)
   kw (batch, seq_len, num_labels, 64)
         │
         ▼ RoPE 旋转位置编码
   qw', kw' (batch, seq_len, num_labels, 64)
         │
         ▼ torch.einsum('bmhd,bnhd->bhmn', qw, kw)
   logits (batch, num_labels, seq_len, seq_len)
         │
         ▼ ÷ √64 (缩放)
   最终得分 (batch, num_labels, seq_len, seq_len)
```

### 1.2 RoPE（旋转位置编码）

```python
# model_task1.py:24-34
def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim, device):
    position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)
    indices = torch.arange(0, output_dim // 2, dtype=torch.float)
    indices = torch.pow(10000, -2 * indices / output_dim)
    embeddings = position_ids * indices
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
    embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
    return embeddings
```

生成 `(batch, seq_len, 64)` 的正弦位置编码。对 Q 和 K 分别施加旋转变换：

```python
# model_task1.py:52-60
qw = qw * cos_pos + qw2 * sin_pos    # 旋转 Q
kw = kw * cos_pos + kw2 * sin_pos    # 旋转 K
```

这样一来，最终 `einsum` 计算出的 `logits[b][l][i][j]` 天然包含了位置 i 和 j 的相对位置信息——这正是 GlobalPointer 能识别 span 边界的关键。

### 1.3 Einstein Summation 计算 Span 得分

```python
logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
```

这个操作的直观含义是：
- `qw[b][m][h][d]`：batch b 中，位置 m 作为**起始**位置，在第 h 个标签类型下的 query 向量
- `kw[b][n][h][d]`：batch b 中，位置 n 作为**结束**位置，在第 h 个标签类型下的 key 向量
- `logits[b][h][m][n]`：batch b 中，第 h 个标签下，**从 m 开始到 n 结束**的 span 得分

所以这个 `(seq_len, seq_len)` 矩阵的每个位置 `(i, j)` 代表 "字符 i 到字符 j 这个 span 对应某个标签的分数"。

---

## 二、子任务 1：框架识别（Frame Identification）

### 2.1 任务本质

> **多分类问题**：给定句子 + 目标词位置 → 从几百个框架中选出正确的那个。

### 2.2 数据处理（`dataset_task1.py`）

```
输入: cfn-train.json + frame_info.json
输出: (input_ids, attention_mask, target_span, label_idx, sentence_id)
```

关键步骤：

1. **字符级分词**：`tokenizer(list(d1['text']), is_split_into_words=True)` — 将句子按单字切开，每个汉字作为一个 token
2. **目标词位置**：取 `target[-1]`（最后一个元素），坐标 **+1** 是因为 BERT 的 `[CLS]` token 占位
3. **标签映射**：从 `frame_info.json` 中读取所有框架名，构建 `label2idx` 字典

### 2.3 模型结构（`model_task1.py`）

```
BERT (chinese-bert-wwm-ext)
  │
  ├── 取最后 4 层 hidden states 做平均
  │   hidden_token = (H[-4] + H[-3] + H[-2] + H[-1]) / 4
  │
  ├── Dense(768, num_frames × 64 × 2)
  │
  ├── RoPE + einsum → logits (batch, num_frames, seq_len, seq_len)
  │
  └── 从 logits 中取目标词位置对应的切片
      token_logits[i] = logits[i, :, target_start, target_end]
      形状: (batch, num_frames)
```

**核心操作**：`logits[i][:, target[0], target[1]]` 从 `(num_frames, seq_len, seq_len)` 的矩阵中，抽取起始位置=目标词起始、结束位置=目标词结束的那个向量，得到 `(num_frames,)` 维的 logits，每个元素代表"目标词激活该框架的分数"。

**Loss**：标准 `CrossEntropyLoss`

### 2.4 推理（`predict_task1.py`）

```python
pred = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
# 输出格式: [[sentence_id, "框架名称"], ...]
```

---

## 三、子任务 2：论元范围识别（Argument Identification）

### 3.1 任务本质

> **Span 检测问题**（类似 NER，但论元可以重叠）：给定句子 + 目标词 → 找出所有论元的 (start, end) 边界。

### 3.2 数据处理（`dataset_task2.py`）

**特殊 token 插入是 Task 2 的关键设计**：在目标词前后分别插入特殊标记 `[1]` 和 `[2]`。

```python
# dataset_task2.py:49-50
input_ids = input_ids[0:target[0]] + [1] + input_ids[target[0]:target[1]+1] + [2] + input_ids[target[1]+1:]
attention_mask = attention_mask + [1, 1]  # 两个特殊 token 也要被 attention
```

插入后的序列结构：
```
[CLS] ... [1] 目标词 [2] ... [SEP]
```

**为什么插 token？**：GlobalPointer 是位置无关的（靠 RoPE 编码相对位置），插入特殊标记让模型能感知目标词的具体位置，从而知道"分析的是哪个词支配的论元"。

**坐标偏移**（因为插入了 2 个 token，目标词后面的位置凭空多了 2）：

```python
# dataset_task2.py:44-48
if line["end"] + 1 < target[0]:
    label.append([line["start"] + 1, line["end"] + 1])     # 在目标词之前，只 +1（[CLS]偏移）
elif line["start"] + 1 > target[1]:
    label.append([line["start"] + 3, line["end"] + 3])     # 在目标词之后，额外 +2（插入了 [1] 和 [2]）
# 注意：与目标词重叠的论元被丢弃了（未处理的 else 分支）
```

**标签构造**：对每个训练样本，构造一个 `(max_len, max_len)` 的 0/1 矩阵 `H_label`，其中 `H_label[start][end] = 1` 表示存在一个 `[start, end]` 的真实论元。

### 3.3 模型结构（`model_task2.py`）

与 Task 1 的区别：

| 对比项 | Task 1 | Task 2 |
|--------|--------|--------|
| BERT 输出 | 最后 4 层平均 | 最后一层 (`last_hidden_state`) |
| num_labels | 框架种类数（~几百） | **1**（只判断"是不是论元"） |
| logits 形状 | `(batch, num_frames, seq_len, seq_len)` | `(batch, 1, seq_len, seq_len)` → squeeze → `(batch, seq_len, seq_len)` |
| 输出语义 | 每种框架的 span 得分 | 每个 (i,j) 是一个论元的得分 |

```python
# model_task2.py:64
logits = logits.squeeze(1)  # (batch, seq_len, seq_len)，每个位置表示"i 到 j 是不是论元"
```

**自定义 Loss**（`model_task2.py:77-83`）：

```python
def compute_loss(self, logits, labels, attention_mask):
    H_attention_mask = torch.triu(
        torch.matmul(attention_mask.unsqueeze(2).float(), attention_mask.unsqueeze(1).float()), diagonal=0)
    loss1 = torch.sum(torch.exp(-logits) * H_attention_mask * labels, dim=(1, 2))
    loss2 = torch.sum(torch.exp(logits) * H_attention_mask * (1 - labels), dim=(1, 2))
    loss = torch.sum(torch.log(1 + loss1 + 1e-9) + torch.log(1 + loss2 + 1e-9)) / H_attention_mask.shape[0]
```

这是一个 **soft 版本的二分类 loss**，针对 span 矩阵设计：
- `H_attention_mask`：上三角 mask，确保只考虑 i ≤ j 的有效 span，同时排除 padding 位置的 span
- `loss1`：对真实论元 `(labels=1)`，惩罚 `-logits` 太小（即鼓励 logits 大）
- `loss2`：对非论元 `(labels=0)`，惩罚 `+logits` 太大（即鼓励 logits 小）
- 最终 loss = `log(1 + loss1) + log(1 + loss2)` — 类似 soft hinge loss

### 3.4 推理（`predict_task2.py`）

```python
# predict_task2.py:103-107
H_pred = torch.where(
    output["logits"] >= 0,   # 阈值 0：logits ≥ 0 → 判定为论元
    torch.ones(...),
    torch.zeros(...)
) * H_attention_mask
```

然后 `torch.nonzero(H_pred)` 找出所有被判定为论元的 span 位置，并根据 special token 偏移做**逆向坐标还原**：

```python
# predict_task2.py:112-115
if idx[2] < target[idx[0]][0]:
    pred.append([sentence_id, idx[1] - 1, idx[2] - 1])      # 在目标词前，-1 去掉 [CLS]
elif idx[1] > target[idx[0]][1]:
    pred.append([sentence_id, idx[1] - 3, idx[2] - 3])      # 在目标词后，-3 去掉 [CLS]+[1]+[2]
```

输出格式：
```json
[[2611, 0, 2], [2611, 8, 8], [2611, 10, 11], ...]
```

### 3.5 训练技巧

Task 2 的 `train_task2.py` 中启用了 **FGM 对抗训练**（而 Task 1 和 Task 3 虽定义了 FGM 但注释掉了）：

```python
# train_task2.py:177-184
fgm.attack()  # 在 embedding 上添加对抗扰动
loss_sum = model(...)['loss']  # 用扰动后的参数再算一次 loss
loss_sum.backward()  # 累加对抗梯度
fgm.restore()  # 恢复原始 embedding
```

FGM 通过在 embedding 层添加梯度方向的扰动，使模型对输入扰动更鲁棒。

---

## 四、子任务 3：论元角色识别（Role Identification）

### 4.1 任务本质

> **Span 级别分类**：给定句子 + 目标词 + 论元 span → 判断该论元的语义角色（如"告发者""方式"等）。

### 4.2 数据处理（`dataset_task3.py`）

**训练数据**：将每句话中的每个论元展开为独立样本。

```python
# dataset_task3.py:26-43
for line in self.all_data:
    for spans in line["cfn_spans"]:   # 遍历每个论元
        self.data.append({
            'text': text,
            "label_class": self.label2idx[spans["fe_name"]],  # 角色标签
            "label_idx": [start, end],                         # 论元范围
            ...
        })
```

与 Task 2 同样插入 `[1]` 和 `[2]` 标记，且做同样的坐标偏移。标签是所有框架中所有角色的去重并集（全局标签空间）。

**预测数据**（`predict_task3.py` 中的 Dataset 类）：读取 Task 2 的预测结果 `B_task2_test.json`，将 Task 2 输出的每个 span 作为待分类样本。

### 4.3 模型结构（`model_task3.py`）

模型结构与 Task 1 **完全相同**：
- BERT 最后 4 层平均
- GlobalPointer + RoPE
- 从 `logits[b][:, target_start, target_end]` 提取目标词位置的得分向量
- CrossEntropyLoss

唯一区别在于 **标签空间**：

| 对比项 | Task 1 | Task 3 |
|--------|--------|--------|
| 标签 | 框架名称（~几百类） | 所有框架中角色的去重并集（~几十类） |
| 每次预测 | 每个句子 → 1 个框架 | 每个论元 span → 1 个角色 |

### 4.4 推理（`predict_task3.py`）

```python
pred = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
predicts.append([sentence_id, ori_target[0], ori_target[1], idx2label[pred[i]]])
```

使用 Task 2 输出中的原始坐标（`ori_target`），不进行坐标偏移还原。输出格式：
```json
[[2611, 0, 2, "告发者"], [2611, 8, 8, "方式"], ...]
```

---

## 五、三个任务的统一视角

### 5.1 GlobalPointer 的三种用法

```
Task 1 (框架识别):
  logits[b][frame][target_start][target_end] → 目标词 span 对应的框架分数
  用 CrossEntropy 做多分类

Task 2 (论元范围识别):
  logits[b][0][i][j] → 每个 (i,j) span 是论元的分数
  用自定义 soft-hinge loss 做二分类，阈值 0 判定

Task 3 (论元角色识别):
  logits[b][role][target_start][target_end] → 目标词 span 对应的角色分数（但输入的是论元的坐标作为 target）
  用 CrossEntropy 做多分类
```

### 5.2 共性设计模式

| 设计模式 | 说明 |
|----------|------|
| **字符级分词** | 所有任务都将文本按单字切分，保证论元边界精确对应到字符位置 |
| **坐标偏移** | 因 `[CLS]` 和特殊标记占位，都需要做坐标映射 |
| **NoisyTune** | 训练前在参数上加高斯噪声（std × 0.15），提升泛化性 |
| **Warmup + Linear Decay** | 学习率先线性 warmup 再线性衰减 |
| **预训练 BERT** | 使用 chinese-bert-wwm-ext 作为共享编码器，三个任务各自微调 |

### 5.3 三个任务的关键差异

```
                 Task 1              Task 2                Task 3
┌──────────┬─────────────────┬───────────────────┬──────────────────┐
│ 任务类型  │   多分类          │   Span 检测         │   多分类          │
│ BERT 输出 │   最后4层平均      │   最后一层          │   最后4层平均      │
│ 特殊 token │   无              │   插入 [1] 和 [2]  │   插入 [1] 和 [2] │
│ num_labels │   ~几百（框架数）  │   1（二分类）        │   ~几十（角色数）  │
│ Loss      │   CrossEntropy   │   Soft hinge loss   │   CrossEntropy   │
│ 对抗训练   │   注释掉          │   启用 FGM           │   注释掉          │
│ 输出      │   [id, "框架名"]  │   [id, start, end]   │  [id, s, e, "角色"]│
│ 权重      │   0.3            │   0.3                │   0.4             │
└──────────┴─────────────────┴───────────────────┴──────────────────┘
```

### 5.4 数据流全景图

```
                    cfn-train.json + frame_info.json
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
    dataset_task1       dataset_task2       dataset_task3
    (句→框架标签)        (句→span矩阵)        (每个论元→角色标签)
          │                   │                   │
          ▼                   ▼                   ▼
      model_task1         model_task2         model_task3
    BERT+GlobalPointer  BERT+GlobalPointer  BERT+GlobalPointer
    取最后4层平均        取最后一层           取最后4层平均
          │                   │                   │
          ▼                   ▼                   ▼
     train_task1          train_task2         train_task3
     (CrossEntropy)      (Soft hinge loss)    (CrossEntropy)
     (NoisyTune)         (NoisyTune+FGM)      (NoisyTune)
          │                   │                   │
          ▼                   ▼                   ▼
      saves/               saves/               saves/
  model_task1_best.bin  model_task2_best.bin  model_task3_best.bin
          │                   │                   │
          ▼                   ▼                   ▼
   predict_task1        predict_task2        predict_task3
          │                   │          (依赖 B_task2_test.json)
          ▼                   ▼                   │
   A_task1_test.json   A_task2_test.json    A_task3_test.json
          │                   │                   │
          └───────────────────┼───────────────────┘
                              │
                              ▼
                         submit.zip
                   (提交到天池平台评测)
```

---

## 六、总结

Baseline 的核心思路是**用 GlobalPointer 统一建模三个子任务**：

1. **Task 1（框架识别）**：GlobalPointer 计算"目标词 span → 框架"的得分矩阵，取目标词位置对应的切片，用 CrossEntropy 做多分类。
2. **Task 2（论元范围识别）**：GlobalPointer 计算"任意 span → 论元"的得分矩阵（单通道），用自定义 soft-hinge loss 做二分类，阈值 0 判定。通过插入 `[1]`/`[2]` 特殊 token 标记目标词位置。
3. **Task 3（论元角色识别）**：与 Task 1 相同结构，但对每个论元 span 做角色多分类。预测时依赖 Task 2 的输出结果。

三个模型各自独立训练，共享 BERT 预训练权重作为起点，通过 RoPE 编码的相对位置信息来捕捉论元的边界和语义关系。

**Baseline 性能**：

| task1_acc | task2_f1 | task3_f1 | task_score |
|:---------:|:--------:|:--------:|:----------:|
| 70.83     | 83.06    | 57.08    | 69.00      |
