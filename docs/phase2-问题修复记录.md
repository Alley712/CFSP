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
