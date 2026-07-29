# Phase 3.4：框架约束的 Task 3 角色解码

> **日期**：2026-07-29
> **前置条件**：Phase 3.2 完成（E7 A 榜 70.77 / E8 B 榜 69.46，Task3 框架特征注入有效）
> **目标**：利用 `frame_info.json` 中的框架-角色从属关系约束 Task 3 解码，不依赖 FGM/EMA/学习率调度等通用训练技巧
> **预计工期**：1.5 天
> **实施环境**：`newline` conda 环境（PyTorch 1.13.0 + transformers 4.24.0 + RoBERTa-wwm-ext）

---

## 一、问题分析

### 1.1 当前瓶颈

Phase 3.2（E7/E8）已通过 FrameEmbedding + 偏置融合在 Task 3 中引入了**软约束**——框架信息作为可学习的偏置向量加到角色 logits 上。但模型仍然面临一个问题：

**Task 3 的标签空间是 1009 个全局唯一角色名，而每个框架平均只有约 42.7 个合法角色。** 这意味着对于任意一个框架，约 96% 的角色标签是非法竞争项——它们根本不属于当前框架，却仍在参与 softmax 竞争。

| 统计项 | 数值 |
|--------|:---:|
| 框架种类数 | 713 |
| Task3 全局唯一 FE 名称 | 1009 |
| 每框架平均 FE 数 | 42.7 |
| 非法角色占比（平均）| ~95.8% |

### 1.2 核心思路

**解码时显式利用 `frame_info.json` 中「哪个角色属于哪个框架」的结构知识，屏蔽非法角色的竞争。**

```
当前 (E7/E8):
  Task1 → top-1 frame → FrameEmbedding → soft bias → 与角色 logits 相加 → argmax

Phase 3.4:
  Task1 → top-K frames + 分数
                ↓
  frame_info.json → 合法角色映射
                ↓
  Task3 在每个候选框架下预测角色 → 非法角色 mask → 融合框架分数 → argmax
```

### 1.3 为什么不直接用 Top-1 硬约束

Task 1 准确率约 71-72%。如果 top-1 框架预测错误，正确角色会被合法角色 mask 直接屏蔽，无法恢复。因此推荐 **Top-K 框架边际化**，允许 top-3 或 top-5 框架下的所有合法角色参与竞争。

---

## 二、实施计划

### 2.1 三阶段实验设计

```
E12a: Top-1 硬约束（快速验证）
  └── 不改模型，只改 predict_task3.py 的解码逻辑
  └── 目的：验证合法角色约束是否有正向信号

E12b: Top-K 边际化（主实验）
  └── predict_task1.py 额外输出 top-K + 分数
  └── predict_task3.py 读 top-K，对每个 span 做多框架条件解码 + 融合
  └── 目的：在约束和容错之间取得平衡

E12c: 训练期 Mask（可选，取决于 E12b 结果）
  └── model_task3.py / train_task3.py 训练时加入框架角色 mask
  └── 目的：让训练和推理约束一致
```

### 2.2 修改范围一览

```
Phase 3.4 涉及文件:

E12a (只改预测):
  newline/predict_task3.py     ★ 核心改动：读取 top-1 frame，构建合法角色 mask，过滤非法 logits

E12b (改 Task1 输出 + Task3 解码):
  newline/predict_task1.py     ★ 改动：额外输出 top-K 框架及分数
  newline/predict_task3.py     ★ 核心改动：读取 top-K，多框架条件解码 + 融合

E12c (改训练，可选):
  newline/model_task3.py       ★ 改动：forward 支持 frame_role_mask
  newline/train_task3.py       ★ 改动：传递 frame_role_mask 到模型
  newline/dataset_task3.py     ★ 改动：构建 frame2roles 映射
  newline/predict_task3.py     (已由 E12b 修改)

公共工具:
  newline/frame_roles.py       ★ 新增：frame_info.json → frame2roles / role2idx 映射工具
```

---

## 三、详细设计

### 3.0 公共工具：`frame_roles.py`（新增）

为避免在多个文件中重复构建映射逻辑，抽取一个公共模块：

```python
"""
frame_info.json → 框架-角色约束映射。

用途：
  - predict_task3.py: 解码时构建合法角色 mask
  - dataset_task3.py: E12c 训练时构建合法角色 mask
  - 验证脚本: 统计 top-K 覆盖率
"""

import json


def load_frame_info(frame_info_path):
    """加载 frame_info.json，返回原始列表"""
    with open(frame_info_path, 'r', encoding='utf8') as f:
        return json.load(f)


def build_frame2roles(frame_info_path, role2idx):
    """
    构建 frame_id → 合法角色 id 集合的映射。

    Args:
        frame_info_path: frame_info.json 路径
        role2idx: {fe_name: idx} 角色名→索引的映射
                  (与 dataset_task3.py 中的 label2idx 保持一致)

    Returns:
        frame2roles: dict, frame_id → set of legal role_ids
        frame2idx: dict, frame_name → frame_id
        idx2frame: dict, frame_id → frame_name
    """
    frame_info = load_frame_info(frame_info_path)

    frame2idx = {}
    idx2frame = {}
    for i, item in enumerate(frame_info):
        frame2idx[item['frame_name']] = i
        idx2frame[i] = item['frame_name']

    frame2roles = {}
    for i, item in enumerate(frame_info):
        legal_roles = set()
        for fe in item['fes']:
            fe_name = fe['fe_name']
            if fe_name in role2idx:
                legal_roles.add(role2idx[fe_name])
        frame2roles[i] = legal_roles

    return frame2roles, frame2idx, idx2frame


def build_legal_mask(frame_id, frame2roles, num_labels, illegal_val=float('-inf')):
    """
    为指定 frame 构建合法角色 mask。

    Args:
        frame_id: int, 框架 ID
        frame2roles: dict, frame_id → set of legal role_ids
        num_labels: int, 总角色类别数 (1009)
        illegal_val: 非法角色的 mask 值

    Returns:
        mask: (num_labels,) tensor, 合法=0, 非法=illegal_val
    """
    import torch

    mask = torch.full((num_labels,), illegal_val)
    legal_ids = frame2roles.get(frame_id, set())
    for rid in legal_ids:
        mask[rid] = 0.0
    return mask
```

---

### 3.1 E12a：Top-1 硬约束

#### 3.1.1 `predict_task3.py` 改动

**核心逻辑**：在模型输出 logits 后、argmax 前，对当前框架下不合法的角色做 mask。

```python
# === 新增导入 ===
from frame_roles import build_frame2roles, build_legal_mask

# === Dataset.__init__ 中新增 ===
# 原代码: self.sent2frame = {...}  (从 Task1 top-1 构建)
# 新增: 构建 frame2roles 映射
self.frame2idx = {}
for i, item in enumerate(self.ori_labels):
    self.frame2idx[item['frame_name']] = i

# 构建合法角色映射 (需要 role2idx)
self.frame2roles, _, _ = build_frame2roles(label_file, self.label2idx)

# === 预测循环中新增 ===
# 原代码获取 frame_id:
frame_name = self.sent2frame.get(sent_id, ...)
frame_id = self.frame2idx[frame_name]

# 模型前向传播获得 logits
output = model(...)
token_logits = output['logits']  # (batch, num_labels)

# ★ E12a 新增：对每个样本，根据其框架 mask 非法角色
for i in range(batch_size):
    legal_mask = build_legal_mask(
        frame_ids[i], self.frame2roles, self.num_labels, illegal_val=-10.0)
    token_logits[i] = token_logits[i] + legal_mask

# 后续 argmax 不变
pred = torch.argmax(token_logits, dim=-1)
```

**改动量**：约 +15 行，无需重新训练。

#### 3.1.2 验证要点

- dev 集上 Precision 是否提升（非法角色被过滤）
- dev 集上 Recall 是否下降（top-1 错误导致误伤）
- 如果 Precision 提升显著 + Recall 基本持平 → E12a 通过，进入 E12b
- 如果 Recall 下降明显 → 说明 top-1 级联错误严重，直接进 E12b（top-K 缓解）

---

### 3.2 E12b：Top-K 边际化（主实验）

这是 Phase 3.4 的**核心实验**。

#### 3.2.1 `predict_task1.py` 改动——输出 Top-K

```python
# === predict_task1.py 新增 ===
import torch.nn.functional as F

# 在模型输出 logits 后：
# 原代码:
# probs = F.softmax(logits, dim=-1)
# pred = torch.argmax(probs, dim=-1)
# pred_frame = idx2label[pred]

# ★ E12b 新增：输出 top-K 框架及分数
K = 3  # 可调
topk_probs, topk_indices = torch.topk(F.softmax(logits, dim=-1), k=K, dim=-1)

# 保存两份输出：
# 1. A_task1_test.json: 保持原格式 [[sentence_id, "框架名"], ...]（向后兼容）
# 2. A_task1_test_topk.json: 新格式 [{"sentence_id": ..., "topk": [["框架名", prob], ...]}, ...]

results = []
results_topk = []
for i in range(batch_size):
    # 原格式（向后兼容）
    results.append([sentence_ids[i], idx2label[pred[i].item()]])

    # top-K 新格式
    frame_scores = []
    for k in range(K):
        frame_name = idx2label[topk_indices[i, k].item()]
        prob = topk_probs[i, k].item()
        frame_scores.append([frame_name, prob])
    results_topk.append({
        'sentence_id': sentence_ids[i],
        'topk': frame_scores
    })

# 保存
json.dump(results, open('A_task1_test.json', 'w'))
json.dump(results_topk, open('A_task1_test_topk.json', 'w'))
```

**输出格式**：

```json
// A_task1_test_topk.json
[
  {
    "sentence_id": 2611,
    "topk": [
      ["举报", 0.42],
      ["控告", 0.18],
      ["诉讼", 0.09]
    ]
  },
  ...
]
```

#### 3.2.2 `predict_task3.py` 改动——多框架条件解码 + 融合

**第一步：Dataset 加载 top-K**

```python
class Dataset(torch.utils.data.Dataset):
    def __init__(self, json_file, label_file, task1_topk_file, task2_file, tokenizer):
        # task1_topk_file: Task1 的 top-K 预测
        with open(task1_topk_file, 'r', encoding='utf8') as f:
            self.task1_data = json.load(f)

        # 构建 sentence_id → topK 映射
        self.sent2topk = {}
        for item in self.task1_data:
            self.sent2topk[item['sentence_id']] = item['topk']
            # item['topk'] = [["框架名", prob], ["框架名", prob], ...]

        # 框架映射
        self.frame2idx = {}
        self.idx2frame = {}
        for i, item in enumerate(self.ori_labels):
            self.frame2idx[item['frame_name']] = i
            self.idx2frame[i] = item['frame_name']

        # 合法角色映射
        self.frame2roles, _, _ = build_frame2roles(label_file, self.label2idx)

        # 其余不变...
```

**第二步：为每个 span 生成多框架条件的样本**

E12b 的一个关键设计选择：**每个 span 需要推理 K 次**（每个候选框架一次）。

有两种实现方式：

**方式 A（推荐）：batch 内展开**

```python
# 对每个原始 span 样本，复制 K 份，每份带不同 frame_id
# 好处：可以利用 batch 并行，一次 forward 处理 K×N 个样本
# 坏处：batch 膨胀 K 倍

# Dataset.__getitem__ 改为返回 K 组输入
def __getitem__(self, item):
    d1 = self.data[item]
    sent_id = d1['sentence_id']

    # 获取该句的 top-K 框架
    topk_frames = self.sent2topk.get(sent_id, [[self.idx2frame[0], 1.0]])

    # 为每个候选框架构造一份输入
    samples = []
    for frame_name, frame_prob in topk_frames[:K]:
        frame_id = self.frame2idx.get(frame_name, 0)
        # ... 原有编码逻辑 ...
        samples.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label_idx': label_idx,
            'frame_id': frame_id,
            'frame_prob': frame_prob,
            'sentence_id': sent_id,
        })
    return samples  # 返回 K 组
```

**方式 B（更简单）：逐框架循环**

```python
# 对每个 top-K 框架，分别跑一次模型 forward
# 好处：不改 Dataset，代码改动最小
# 坏处：K 次前向传播，预测时间 ×K

for frame_name, frame_prob in topk_frames[:K]:
    frame_id = self.frame2idx[frame_name]
    # 构造该 frame 下的 batch
    # 模型 forward
    # 收集该 frame 下的角色 logits
```

**推荐：方式 B（实现简单 + K=3 时时间可接受）**

**第三步：解码融合**

```python
# 核心融合循环
K = 3
alpha = 1.0       # 框架置信度权重
illegal_val = -10.0  # 非法角色 mask 值
fusion = 'max'     # 'max' 或 'logsumexp'

# 对测试集中每个唯一的 sent_id+span：
all_predictions = []
for sent_id, start, end in task2_spans:
    topk_frames = sent2topk.get(sent_id, [[backup_frame, 1.0]])

    # 收集每个候选框架下的角色分数
    frame_logits_list = []   # [(logits_tensor, frame_prob), ...]

    for frame_name, frame_prob in topk_frames[:K]:
        frame_id = frame2idx[frame_name]

        # 构建该 frame 条件下的模型输入
        # 跑模型 forward → logits (1, num_labels)
        logits = model(input_ids=..., frame_ids=frame_id, ...)['logits']

        # 加合法角色 mask
        legal_mask = build_legal_mask(frame_id, frame2roles, num_labels, illegal_val)
        logits = logits + legal_mask  # 非法角色被大幅压低

        frame_logits_list.append((logits, frame_prob))

    # 融合 K 个框架的角色分数
    if fusion == 'max':
        # max 融合：对每个角色取 K 个框架下的最大分数
        all_scores = []
        for logits, frame_prob in frame_logits_list:
            all_scores.append(logits + alpha * math.log(frame_prob))
        stacked = torch.stack(all_scores, dim=0)  # (K, num_labels)
        final_logits, _ = torch.max(stacked, dim=0)  # (num_labels,)
    elif fusion == 'logsumexp':
        # logsumexp 融合
        all_scores = []
        for logits, frame_prob in frame_logits_list:
            all_scores.append(logits + alpha * math.log(frame_prob))
        stacked = torch.stack(all_scores, dim=0)
        final_logits = torch.logsumexp(stacked, dim=0)

    pred_role_id = torch.argmax(final_logits).item()
    pred_role = idx2label[pred_role_id]
    all_predictions.append([sent_id, start, end, pred_role])
```

**第四步：融合策略选择**

| 参数 | 默认值 | 候选值 | 说明 |
|------|:-----:|--------|------|
| K | 3 | 3, 5 | 候选框架数 |
| alpha | 1.0 | 0.5, 1.0, 2.0 | 框架置信度权重 |
| illegal_val | -10.0 | -10, -inf | 非法角色惩罚 |
| fusion | max | max, logsumexp | 融合方式 |

**推荐第一版配置**：`K=3, alpha=1.0, illegal_val=-10, fusion=max`

#### 3.2.3 预测脚本入口

```python
if __name__ == '__main__':
    test_dataset = Dataset(
        "./dataset/cfn-test-B.json",
        "./dataset/frame_info.json",
        "./dataset/B_task1_test_topk.json",  # ★ 读取 top-K 格式
        "./dataset/B_task2_test.json",
        tokenizer)

    # K, alpha 等参数可通过命令行或全局变量控制
```

---

### 3.3 E12c：训练期 Mask（可选）

> **执行条件**：E12b 有稳定提升（Task3 F1 ≥ +0.5），且 A/B 榜一致。

#### 3.3.1 `model_task3.py` 改动

在 `forward()` 中增加 `frame_role_mask` 参数：

```python
def forward(self, ..., frame_role_mask=None):
    # ... 原有 GlobalPointer + FrameEmbedding 逻辑 ...
    # token_logits: (batch, num_labels)

    # ★ E12c 新增：训练时 mask 非法角色
    if frame_role_mask is not None:
        token_logits = token_logits + frame_role_mask
        # frame_role_mask: (batch, num_labels)，合法=0，非法=illegal_val

    # 原有 loss 计算
    if labels is not None:
        loss_fc = nn.CrossEntropyLoss()
        loss = loss_fc(token_logits, labels)
    ...
```

#### 3.3.2 `dataset_task3.py` 改动

```python
class Dataset:
    def __init__(self, ...):
        # 加载 frame_info.json 构建合法角色映射
        self.frame2roles, self.frame2idx, _ = build_frame2roles(
            label_file, self.label2idx)

    def __getitem__(self, item):
        # ... 原有逻辑 ...
        frame_id = d1["frame_id"]

        # ★ E12c 新增：构建该 frame 的合法角色 mask
        legal_roles = self.frame2roles.get(frame_id, set())
        mask = torch.zeros(self.num_labels)
        for i in range(self.num_labels):
            if i not in legal_roles:
                mask[i] = float('-inf')  # 或 -10.0

        return input_ids, attention_mask, label_idx, label, sentence_id, frame_id, mask
```

#### 3.3.3 `train_task3.py` 改动

```python
def get_model_input(data, device=None):
    # ... 现有逻辑 ...
    frame_role_masks = []
    for d in data:
        frame_role_masks.append(d[6])  # mask 是第 7 个元素

    frame_role_masks = torch.stack(frame_role_masks).to(device)
    return ..., frame_ids, frame_role_masks

# train/eval 函数中：
output = model(
    ...,
    frame_ids=frame_ids,
    frame_role_mask=frame_role_masks,  # ★ 新增
    ...
)
```

#### 3.3.4 风险应对

训练时用 gold frame → mask 100% 正确。推理时用 Task1 预测 → 框架可能错。
**缓解**：训练时以概率 p（如 0.1-0.2）随机使用无 mask 的原始 logits，防止模型过度依赖 mask 信号。

```python
if frame_role_mask is not None and training:
    # 以 10% 概率不使用 mask
    if random.random() < 0.1:
        frame_role_mask = None
```

---

## 四、验证实验（dev 集评估）

### 4.1 Task 1 Top-K 覆盖率

在 dev 集上跑 predict_task1，统计：

```python
# 伪代码
for sample in dev_set:
    gold_frame = sample['frame']
    topk_frames = task1_predict_topk(sample)  # top-3 或 top-5
    top1_hit = (gold_frame == topk_frames[0])
    top3_hit = (gold_frame in topk_frames[:3])
    top5_hit = (gold_frame in topk_frames[:5])

print(f"Top-1 Acc: {top1_hit_rate:.4f}")
print(f"Top-3 Acc: {top3_hit_rate:.4f}")
print(f"Top-5 Acc: {top5_hit_rate:.4f}")
```

**预期**：top-3 覆盖率如果 >90%，说明 E12b 有很大空间；如果 <85%，则 top-1 错误率过高，mask 风险较大。

### 4.2 合法角色命中率

```python
for sample in dev_set:
    gold_role = sample['cfn_spans'][i]['fe_name']
    topk_frames = task1_predict_topk(sample)

    # 统计 gold role 是否在 top-K 框架的合法角色并集中
    for k in [1, 3, 5]:
        legal_union = set()
        for fname, _ in topk_frames[:k]:
            legal_union |= frame2roles[frame2idx[fname]]
        hit[k] = (role2idx[gold_role] in legal_union)
```

该指标直接反映 mask 的误伤风险：hit@K 越高，Recall 越不容易受损。

### 4.3 Dev 集消融设计

| 实验 | K | alpha | illegal | fusion | 预期 Task3 Acc | 说明 |
|------|:--:|:-----:|:-------:|:------:|:---:|------|
| E7-baseline | — | — | — | — | (当前 best) | 无约束 |
| E12a-dev | 1 | — | -10 | — | P↑ R↓? | Top-1 硬约束 |
| E12b-k3-max | 3 | 1.0 | -10 | max | P↑→ | 主实验 |
| E12b-k3-log | 3 | 1.0 | -10 | logsumexp | ? | 对比融合方式 |
| E12b-k5-max | 5 | 1.0 | -10 | max | ? | 更多候选 |
| E12b-a05 | 3 | 0.5 | -10 | max | ? | 弱化框架权重 |
| E12b-a20 | 3 | 2.0 | -10 | max | ? | 强化框架权重 |

---

## 五、实施步骤

### Step 1：创建 `frame_roles.py` 公共工具（20 分钟）

```bash
cd /root/autodl-tmp/CFSP/newline
```

- 实现 `load_frame_info`, `build_frame2roles`, `build_legal_mask`
- 验证：打印 3 个框架的合法角色集合，确认映射正确

### Step 2：dev 集分析（30 分钟）

- 运行 Task1 预测，统计 top-1/3/5 覆盖率
- 统计合法角色命中率
- 判断 E12a/E12b 是否有空间

### Step 3：E12a — Top-1 硬约束（1 小时）

- 修改 `predict_task3.py`（只改解码部分）
- dev 集评估
- 如果正向 → 提交 A/B 榜
- 如果 Recall 下降明显 → 直接跳到 E12b

### Step 4：E12b — Top-K 边际化（2-3 小时）

- 修改 `predict_task1.py`：输出 top-K
- 修改 `predict_task3.py`：读取 top-K + 多框架条件解码 + 融合
- dev 集验证 + 调参（K, alpha, illegal_val, fusion）
- A/B 榜提交

### Step 5（可选）：E12c — 训练期 Mask（3-4 小时）

- 修改 `model_task3.py`, `dataset_task3.py`, `train_task3.py`
- 重新训练 Task 3（10 epochs, ~2.5h）
- dev 评估 + A/B 榜提交

### Step 6：结果汇总（30 分钟）

- 更新 `experiments/exp_log.md`
- 记录最终结论

---

## 六、实验记录模板

在 `experiments/exp_log.md` 中新增：

```markdown
## E12 系列 — 框架约束解码

### dev 集分析

| 统计项 | Top-1 | Top-3 | Top-5 |
|--------|:---:|:---:|:---:|
| 框架覆盖率 | ? | ? | ? |
| 合法角色命中率 | ? | ? | ? |

### A 榜

| 实验编号 | 配置 | Task1 | Task2 | Task3 | 总分 | 备注 |
|----------|------|:---:|:---:|:---:|:---:|------|
| E7 | 当前最佳 | 72.00 | 84.30 | 59.71 | 70.77 | 基线 |
| E12a | Top-1 hard mask | ? | ? | ? | ? | |
| E12b-k3 | Top-3 max 融合 | ? | ? | ? | ? | |
| E12b-k5 | Top-5 max 融合 | ? | ? | ? | ? | |
| E12b-best | 最优参数 | ? | ? | ? | ? | |

### B 榜

| 实验编号 | 配置 | Task1 | Task2 | Task3 | 总分 | 备注 |
|----------|------|:---:|:---:|:---:|:---:|------|
| E8 | 当前最佳 | 71.01 | 84.54 | 56.99 | 69.46 | 基线 |
| E12b-best-B | 同A榜最优参数 | ? | ? | ? | ? | |
```

---

## 七、代码总改动量估算

| 文件 | 实验 | 改动类型 | 改动行数 |
|------|:---:|------|:--:|
| `frame_roles.py` | E12a | **新增** | ~60 |
| `predict_task3.py` | E12a | 解码逻辑 | +20 |
| `predict_task1.py` | E12b | 输出 top-K | +25 |
| `predict_task3.py` | E12b | 多框架解码 + 融合 | +60 |
| `model_task3.py` | E12c | 训练期 mask | +8 |
| `dataset_task3.py` | E12c | mask 构建 | +15 |
| `train_task3.py` | E12c | 传递 mask | +10 |

总代码量：E12a+b 约 ~165 行，E12c 额外 ~33 行。属于**中等改动**，核心风险可控——改动集中在预测阶段的解码逻辑，E12a 和 E12b 不涉及重新训练。

---

## 八、预期效果与风险评估

### 8.1 预期提升

| 场景 | Task3 F1 提升 | 总分提升 (A榜) | 依据 |
|------|:-----------:|:------:|------|
| 乐观 | +2.0~2.5 | +0.8~1.0 (→ 71.6~71.8) | top-K 覆盖率高，非法角色竞争大幅减少 |
| 中性 | +0.8~1.5 | +0.3~0.6 (→ 71.1~71.4) | Precision 提升，Recall 基本持平 |
| 悲观 | -0.3~+0.5 | -0.1~+0.2 | Task1 错误率较高，mask 误伤 |

### 8.2 风险矩阵

| 风险 | 概率 | 影响 | 应对 |
|------|:---:|------|------|
| Task1 top-1 错误致 mask 误伤 | 中高 | Recall 下降 | 用 top-K 而非 top-1 |
| 部分 FE 跨框架共享，mask 收益有限 | 中 | 提升不显著 | 融合原始 task3 logits，不完全依赖框架 |
| K 倍推理耗时 | 中 | 预测变慢 | K=3 默认；如需加速用 batch 展开方式 |
| mapping 不一致 | 低 | mask 错误 | 统一从 frame_info.json 构建所有映射 |
| A 榜升 B 榜降 | 低 | 泛化风险 | 双榜对照 E7/E8 |

### 8.3 快速回退

E12a/E12b 不涉及重新训练，回退只需：
```bash
cd /root/autodl-tmp/CFSP/newline
git checkout predict_task1.py predict_task3.py
# 恢复 E7/E8 预测脚本
```
E12c 如需回退：
```bash
cp backups/phase3.2/model_task3.py backups/phase3.2/dataset_task3.py backups/phase3.2/train_task3.py .
```

---

## 九、与现有方案的对比

| 方案 | 框架信息使用 | 约束类型 | 是否需要重训 |
|------|------------|:--:|:--:|
| E5/E6 (RoBERTa only) | 无 | — | — |
| E7/E8 (FrameEmbedding) | soft bias | 软约束 | 是 (Task3) |
| E12a (Top-1 mask) | hard mask | 强约束 | 否 |
| E12b (Top-K 边际化) | mask + 融合 | 中强约束 | 否 |
| E12c (训练期 mask) | 训练+推理 mask | 最强 | 是 (Task3) |

Phase 3.4 不是替代 E7/E8，而是在其基础上**叠加解码层约束**。两种框架信息使用方式互补：FrameEmbedding 提供 soft preference，mask 提供 hard exclusion。

---

## 十、关键决策记录

1. **E12a 先于 E12b**：用最小改动验证合法角色约束是否有正向信号，降低无效投入风险
2. **先改解码，后改训练**：E12a/E12b 不需要重新训练，迭代速度快（修改→预测→评估 < 30 分钟）
3. **max 融合优先**：比 logsumexp 更简单，作为第一版。如效果接近则保留 max
4. **K=3 起步**：大多数情况下 top-3 已覆盖正确框架。K=5 仅在 top-3 覆盖率不足时尝试
5. **alpha=1.0 默认**：框架置信度与角色分数等权重。后续根据 dev 表现调整
6. **E12c 仅在前序实验稳定提升后执行**：训练改动涉及重新训练 Task 3（~2.5h），ROI 需先验证
