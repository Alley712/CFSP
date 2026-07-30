# 随机种子方案：训练可复现性修复

> 日期：2026-07-30
> 背景：E7 (70.77) 与 E12 (70.03) 用完全相同配置训练却差 0.74 分，根因是代码库没有任何随机种子设置
> 目标：固定所有随机源，使同配置两次训练结果 bit-exact 一致

---

## 一、现状

整个代码库没有一行 `seed` / `manual_seed` / `deterministic` 设置。以下随机源每次训练都产生不同结果：

```
Python random 模块 ───→ DataLoader shuffle 顺序
     │
torch.manual_seed ───→ torch.rand()（NoisyTune 噪声）
     │               ───→ nn.Dropout 随机 mask
     │               ───→ nn.Linear / nn.Embedding 初始权重
     │
cuDNN ───────────────→ attention / conv CUDA kernel 浮点累加顺序
```

---

## 二、随机源完整清单

### 2.1 全局随机源（影响最大）

| 随机源 | 控制方式 | 影响范围 |
|--------|----------|----------|
| Python `random` | `random.seed()` | DataLoader shuffle、AEDA 标点选取 |
| NumPy `np.random` | `np.random.seed()` | 任何 numpy 随机操作 |
| PyTorch CPU | `torch.manual_seed()` | `torch.rand()`、参数初始化、Dropout |
| PyTorch CUDA | `torch.cuda.manual_seed_all()` | GPU 上的 `torch.rand()` |
| cuDNN backend | `torch.backends.cudnn.deterministic` | CUDA kernel 非确定性 |
| cuDNN autotuner | `torch.backends.cudnn.benchmark` | 每次启动选择不同最优 kernel |

### 2.2 训练脚本中的具体随机点

#### NoisyTune（训练入口，3 处）

```
train_task1.py:114  →  torch.rand(para.size())
train_task2.py:138  →  torch.rand(para.size())
train_task3.py:117  →  torch.rand(para.size())
```

在 BERT 预训练权重上施加随机噪声。**每次 run 噪声分布不同，模型初始状态不同，收敛到不同局部最优。** E7 不可复现的主要原因。

#### DataLoader shuffle（训练入口，3 × 2 处）

```
train_task1.py:223  →  shuffle=True (train)
train_task1.py:232  →  shuffle=False (val)
train_task2.py:255  →  shuffle=True (train)
train_task2.py:264  →  shuffle=False (val)
train_task3.py:229  →  shuffle=True (train)
train_task3.py:238  →  shuffle=False (val)
```

`shuffle=True` 使用的是 PyTorch 内部基于全局 `random` 模块的实现。不加 `generator` 参数时，shuffle 顺序完全取决于调用时刻 `random` 模块的内部状态——这会受到之前任何 `random` 调用的影响，无法独立控制。

### 2.3 模型中的随机点

#### 随机初始化（3 个 model 文件）

| 组件 | 文件:行号 | 初始化方式 |
|------|:---:|------|
| `self.dense` (Linear) | model_task1.py:21, model_task2.py:21, model_task3.py:21 | kaiming_uniform |
| `self.lstm` (LSTM) | model_task{1,2,3}.py:14-15 | 定义但未使用 |
| `self.frame_embedding` (Embedding) | model_task3.py:27-28 | N(0,1) |
| `self.frame_proj` (Linear) | model_task3.py:29-30 | kaiming_uniform |

> BERT 预训练权重从 `pytorch_model.bin` 加载，**不受种子影响**。

#### Dropout（3 个 model 文件）

```
model_task1.py:20  →  nn.Dropout(classifier_dropout)
model_task2.py:19  →  nn.Dropout(classifier_dropout)
model_task3.py:20  →  nn.Dropout(classifier_dropout)
```

训练时随机丢弃神经元，预测时自动关闭（`model.eval()`）。受 `torch.manual_seed()` 控制。

### 2.4 AEDA 数据增强中的随机点

```
aeda.py:53  →  random.sample(safe_positions, n)
aeda.py:58  →  random.choice(chars)
```

使用 Python 标准库 `random` 模块，不受 `torch.manual_seed()` 控制。**必须单独设置 `random.seed()`。** 当前 AEDA 实验已放弃，但代码仍存在于 dataset 文件中。

### 2.5 预测脚本

预测脚本全部使用 `shuffle=False` + `num_workers=0`，无 Dropout（`model.eval()` 模式），**预测阶段无随机性**，无需额外修改。

---

## 三、实施方案

### 3.1 需要修改的文件

| 文件 | 改动 | 行数 |
|------|------|:--:|
| `train_task1.py` | 开头加全局种子 + DataLoader 增加 generator | +8 |
| `train_task2.py` | 同上 | +8 |
| `train_task3.py` | 同上 | +8 |
| `params.py` | 新增 `--seed` 参数 | +1 |

### 3.2 `params.py` 新增参数

```python
parser.add_argument('--seed', default=42, type=int,
                    help='Random seed for reproducibility')
```

放在 `--lr` 参数附近。默认值 42，消融时可改为其他值验证训练稳定性。

### 3.3 训练脚本开头加种子函数

在每个 `train_taskX.py` 的 `if __name__ == '__main__':` 之后、其他任何操作之前，调用统一的种子设置。写成内联函数或后续抽取为公共模块均可：

```python
import random
import numpy as np
import torch

def set_seed(seed):
    """固定所有随机源，确保训练可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 入口处调用
if __name__ == '__main__':
    args = construct_hyper_param()
    set_seed(args.seed)
    # ... 其余代码
```

### 3.4 DataLoader 加 generator

```python
g = torch.Generator()
g.manual_seed(args.seed)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,
    generator=g,
)
```

### 3.5 不用改的地方

- **模型文件**（`model_task1/2/3.py`）：随机初始化由 `torch.manual_seed()` 全局控制，无需单独修改
- **预测脚本**（`predict_task1/2/3.py`）：无随机操作
- **AEDA**（`aeda.py`）：当前实验已放弃，先不管。将来复用 `aeda.py` 时需加 `random.seed()`

---

## 四、性能影响

| 设置 | 影响 |
|------|------|
| `cudnn.deterministic=True` | 禁用部分非确定性 CUDA kernel，训练速度下降 < 10% |
| `cudnn.benchmark=False` | 不做 autotune，避免了每次启动选择不同 kernel 带来的非确定性 |
| `generator=g` | 无性能影响，仅显式控制 shuffle 的随机源 |

对于 RTX 3090 + BERT-base（102M 参数），总的性能损失可忽略不计（每个 epoch 多几秒），而可复现性的收益巨大。

---

## 五、预期效果

加种子前：
```
E7 (run 1): 70.77   ← 同配置
E12 (run 2): 70.03  ← 同配置
差距: 0.74 分
```

加种子后：
```
Run 1: 70.03
Run 2: 70.03  ← bit-exact 一致
差距: 0.00
```

消融实验的每一项改动，都能准确归因于方法本身而非随机波动。

---

## 六、与实验记录的关系

exp_log.md 中已知 E7/E8 的权重文件已被后续训练覆盖（`E7/E8 的权重已丢失（被后续训练覆盖），无法复现`）。加上种子后：

1. 重新训练出基线，记录 seed 值和分数
2. 每次方法改动只需跑一次，不需要多次重训取平均
3. 报告中的消融表格可以附上 seed 值以证明可复现性
