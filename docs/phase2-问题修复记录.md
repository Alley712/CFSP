# Phase 2 问题修复记录

## 日期

2026-07-26

## 问题描述

运行 `python train_task1.py` 时报错，无法正常启动训练。

## 错误与修复

### 错误 1：`GradScaler` 导入失败

**错误信息**：
```
ImportError: cannot import name 'GradScaler' from 'torch.amp'
```

**原因**：当前环境安装的 PyTorch 版本（2.1.2）中，`GradScaler` 仅存在于 `torch.cuda.amp`，而非 `torch.amp`。

**修复**（`train_task1.py:10`、`train_task2.py:10`）：
```python
# 修复前
from torch.amp import autocast, GradScaler

# 修复后
from torch.cuda.amp import autocast, GradScaler
```

**同时修复 `autocast` 和 `GradScaler` 调用方式**（与旧版 API 保持一致）：
```python
# 修复前
scaler = GradScaler('cuda') if args.fp16 else None
with autocast('cuda', enabled=args.fp16):

# 修复后
scaler = GradScaler() if args.fp16 else None
with autocast(enabled=args.fp16):
```

### 错误 2：`transformers` 与 `PyTorch` 版本不兼容

**错误信息**：
```
[transformers] Disabling PyTorch because PyTorch >= 2.4 is required but found 2.1.2+cu118
ImportError: cannot import name 'AdamW' from 'transformers'
```

**原因**：环境安装的 `transformers` 版本（5.14.1）要求 PyTorch >= 2.4，与当前 PyTorch 2.1.2 不兼容。

**修复**：
1. 将 PyTorch 从 `2.1.2+cu118` 升级到 `2.7.1+cu118`
2. 将 transformers 降级到兼容版本（最终为 `4.36.0`）

### 错误 3：`tokenizers` 版本检查失败

**错误信息**：
```
ImportError: tokenizers>=0.14,<0.19 is required for a normal functioning of this module, but found tokenizers==0.13.3.
```

**原因**：site-packages 下同时存在两份 `tokenizers` 的 dist-info：
- `tokenizers-0.13.3.dist-info`（旧版本）
- `tokenizers-0.15.2.dist-info`（新版本）

`importlib.metadata.version('tokenizers')` 可能读取到旧版本号，导致 transformers 的版本检查失败。

**修复**：
```bash
rm -rf /root/miniconda3/lib/python3.10/site-packages/tokenizers-0.13.3.dist-info
```

## 最终环境

| 包 | 版本 |
|---|---|
| PyTorch | 2.7.1+cu118 |
| transformers | 4.36.0 |
| tokenizers | 0.15.2 |

## 验证

`python train_task1.py` 成功启动训练，Epoch 1 ~9 it/s 无报错。

## 遗留 Warning（不影响运行）

- `FutureWarning: torch.cuda.amp.autocast(args...) is deprecated` — 新版 PyTorch 推荐 `torch.amp.autocast('cuda', args...)` 语法，但不影响功能
- `FutureWarning: transformers.AdamW is deprecated` — 推荐使用 `torch.optim.AdamW`，但不影响功能

---

## 补充修复：`train_task2.py` Label 索引越界

### 日期

2026-07-26

### 错误信息
```
IndexError: index 75 is out of bounds for dimension 1 with size 75
```
发生在 `train_task2.py:84`：
```python
H_label[i][idx[0], idx[1]] = 1
```

### 根因分析

`dataset_task2.py` 中 tokenizer 对文本做字符级分词时，会**自动剥离开头/中间的空白字符**（如空格）。部分 CFN 数据文本以空格开头（如 `" 第十四条..."`），导致 token 序列比原始字符序列短。

但 label 索引是按原始字符位置 `+1/+3` 手动偏移计算的，没有考虑 tokenizer 剥离字符的情况。当字符被剥离后，`char_pos → token_pos` 的 1:1 映射被打破，索引就会超出 `input_ids` 的实际长度。

**示例**：样本 10501，文本长度 73，其中 2 个空格被剥离：
- 原始 label `[51, 72]` → token 实际位置应为 `[50, 71]`
- 但代码算出 `[51+3, 72+3] = [54, 75]`，而 `max_len = 75`（上限 74）→ 越界

### 修复

**1. `dataset_task2.py:36-71`** — 用 `word_ids()` 做正确的 char→token 位置映射：

- `data.word_ids()` 返回每个 token 对应的 word（字符）索引，自动跳过被剥离的字符（`word_idx = None`）
- 建立 `char_to_tokens` 字典：字符位置 → [首 token, 末 token]
- 对于 label 的 start/end 字符位置，查找实际 token 位置替代手动的 `+1/+3` 偏移
- 被剥离的字符上的标注直接跳过（`continue`）

**2. `train_task2.py:56-58`** — 防御性 bounds check：

```python
max_len = max([len(x[0]) for x in data])
# 防御性检查：确保H_label能容纳所有label索引
for d in data:
    for idx in d[3]:
        max_len = max(max_len, idx[0] + 1, idx[1] + 1)
```

### 验证

- 全量 10700 条训练数据边界检查通过（0 个越界样本）
- `python train_task2.py` 成功启动训练，Epoch 1 ~8-10 it/s 无报错

---

## 补充修复：Task 2 F1 接近 0 的根因分析与修复

### 日期

2026-07-26

### 问题

训练 Task 2 时 F1 极低（原始约 0.037），即使注释掉 FGM 也无改善：
```
current f1: 0.036966118315839715
H_precision: 0.34218288984606815, H_recall: 0.0195384874482839
```
Recall 仅 2%，模型几乎不预测任何正样本。

### 根因分析

#### 1. 损失函数的类不平衡问题（主要原因）

原始 `model_task2.py` 的 `compute_loss` 使用 GlobalPointer 风格的损失函数：

```python
def compute_loss(self, logits, labels, attention_mask):
    loss1 = sum(exp(-logits) * mask * labels, dim=(1,2))     # 正样本项
    loss2 = sum(exp(logits) * mask * (1 - labels), dim=(1,2)) # 负样本项
    loss = mean(log(1 + loss1) + log(1 + loss2))
```

数据集正负样本比例为 **~1:2000**（每样本约 2.5 个 FE span，对阵约 5000 个负 span）。

`log(1+sum)` 结构的致命缺陷：
- `loss2` 的有效项数是 `loss1` 的 ~2000 倍
- 分母 `(1 + loss1 + loss2)` 被 `loss2` 主导，稀释了每个正样本的梯度
- 模型学到的最优策略是把所有 logits 推为负值（loss1 → 0，loss2 最小化），导致 recall ≈ 0

#### 2. NoisyTune 初始化的预训练权重损伤（次要原因）

训练脚本在开始时对整个模型参数添加噪声：
```python
noise_lambda = 0.15
model.state_dict()[name][:] += (rand - 0.5) * noise_lambda * std(para)
```

实测 BERT embedding 权重产生 **4.7% 的相对扰动**，严重破坏预训练表征质量。

### 修复过程与迭代

#### 尝试 1：GlobalPointer loss + 正样本加权

```python
pos_weight = clamp(neg/pos, max=500)
loss = mean(pos_weight * log(1 + loss1) + log(1 + loss2))
```

| weight cap | F1 | Recall | 现象 |
|---|---|---|---|
| 500 | 0.002 | 99% | 过度矫正，全预测为正 |
| 50 | 0.002 | 100% | 仍然过度矫正 |

**失败原因**：`log(1+sum)` 结构下，一旦模型偏向一方，分母被该方向的聚合值主导，个体 span 的梯度消失，进入不可逆的死循环。对权重极度敏感但无稳定中间状态。

#### 尝试 2：替换为 BCEWithLogitsLoss（✓ 方向正确）

```python
def compute_loss(self, logits, labels, attention_mask):
    H_attention_mask = torch.triu(...)
    num_pos = labels.sum() + 1e-9
    num_neg = H_attention_mask.sum() - num_pos + 1e-9
    pos_weight = torch.clamp(num_neg / num_pos, max=50.0).detach()
    bce = F.binary_cross_entropy_with_logits(
        logits, labels, reduction='none', pos_weight=pos_weight)
    loss = (bce * H_attention_mask).sum() / H_attention_mask.sum()
    return loss
```

**优势**：每个 span 的 loss 独立计算，梯度自调节（sigmoid 饱和时梯度自动归零），不存在"总和控制个体"问题。

| pos_weight max | F1 | Precision | Recall |
|---|---|---|---|
| 100 | 0.062 | 3.3% | 50.2% |
| 50（+去 NoisyTune） | 0.069 | 3.8% | 46.9% |

epoch 2 达到最佳 F1=**0.086**（P=4.8%, R=46.8%），epoch 3-5 稳定不再提升。

#### dev 集 logit 分布验证

```
正样本 (n=5937):   mean=-0.33,  46.8% >= 0
负样本 (n=632万):  mean=-6.66,  0.9% >= 0
```

模型有效学到了区分（正负均值差 6.3 个 logit 单位），但 2000:1 的极端不平衡下，固定阈值 0 时，0.9% 的负样本尾部（约 57k 假阳性）完全淹没了真阳性（约 2.8k），导致 precision 仅 ~4.7%。

### 结论

- 损失函数重构为 BCEWithLogitsLoss + 去除 NoisyTune 让 F1 从 0.037 → **0.086**（+132%）
- 模型仍在学习（正负分布有效分离），瓶颈在于极端不平衡 + 固定阈值
- 后续方向：搜索最佳阈值、增加训练 epoch、或引入更强特征编码
