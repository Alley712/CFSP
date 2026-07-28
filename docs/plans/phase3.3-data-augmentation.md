# Phase 3.3：数据增强（AEDA 随机标点插入）

> **日期**：2026-07-28
> **前置条件**：Phase 3.2 完成（E7 A 榜 70.77 / E8 B 榜 69.46）
> **目标**：通过 AEDA 数据增强扩充训练集，提升模型泛化能力
> **预计工期**：0.5 天
> **实施环境**：`newline` conda 环境

---

## 一、方案选择

### 1.1 为什么选 AEDA

| 方案 | 优点 | 缺点 | 可行性 |
|------|------|------|:--:|
| **AEDA**（随机标点插入）| 已内置于代码（`aeda_chars` 变量）；无需外部依赖；中文字符级操作简单；已知有效 | 坐标需要重新计算 | ✅ 首选 |
| 同义词替换 | 语义保持性更好 | 需要同义词表；对中文效果不稳定 | 备选 |
| 回译（中→英→中）| 可产生自然改写 | 需要翻译 API；耗时；可能改变语义框架 | 时间不够 |

**决策**：只做 AEDA，不做同义词替换和回译。原因：① timebox 有限（截止 8 月 1 日）；② AEDA 是代码中预留但未启用的方案，实现成本最低；③ 中文标点插入对论元边界标注的影响可控。

### 1.2 AEDA 原理

```
原始: "张三代人购买了一台电脑"
       ↓ 随机选 2-3 个论元之间的空隙，插入随机标点
增强: "张三，代人购买了一台电脑，"
       ↓ 所有字符位置后移，需要重新计算坐标
```

每个原始样本生成 1 条增强样本，训练集翻倍。

---

## 二、实施计划

### 2.1 整体策略

```
原始训练集 (cfn-train.json, ~10700 条)
        │
        ├── 原始样本（保留）
        │
        └── 增强样本（每原始样本生成 1 条 → ~10700 条新增）
                │
                └── 总共 ~21400 条训练样本
```

**增强参数**：
- 每条原始样本生成 **1 条**增强样本（训练集扩大 1 倍）
- 每次插入最多 **3 个**随机标点（安全位置不足时自动减少；句尾允许插入）
- 标点池：`["。", "；", "？", "：", "！", "，"]`
- 插入位置：论元之间的安全空隙（目标词内部和所有论元内部禁止插入，内部字符由禁区并集保护；重叠/嵌套论元通过并集天然避免拆分）

### 2.2 三任务分别处理

#### Task 1（框架识别）

改动最简单：

```
原始: text="张三代人购买了一台电脑", target=[4,5]("购买"), frame="商业购买"
增强: text="张三代人，购买了一台电脑，"
      target 坐标不变（标点插在 target 之后）
      frame 不变
```

**实现**：Dataset.__init__ 中，遍历 all_data 后，对每条生成增强副本，追加到 self.all_data。
**坐标计算**：禁区排除目标词内部和论元内部后，从安全空隙中随机选取插入位置；坐标偏移根据插入位置在 target/span 之前还是之后自动累加。

#### Task 2（论元范围识别）

坐标计算稍复杂——label 是多组 span：

```
原始: text="张三代人购买了一台电脑", target=[4,5], 
      spans=[[0,1]("张三"), [2,3]("代人"), [6,9]("一台电脑")]
增强: text="张三代人，购买了一台电脑，"
      所有坐标需要根据插入位置逐个偏移
```

**实现**：同 Task 1 逻辑，但需要处理多个 span 的坐标偏移。

#### Task 3（论元角色识别）

Task 3 的 Dataset 将每个 span 展开为独立样本，且已有 frame_id。增强时需要：

```
1. 在 all_data 层面做 AEDA（文本增强 + 坐标修正）
2. 增强后的样本仍然按 span 展开
3. frame_id 和 fe_name（role）继承自原始标注
```

### 2.3 AEDA 核心函数

新增一个公共工具函数，三个 Dataset 共用：

```python
# 放在 newline/aeda.py 或直接内联到各 dataset 文件中

import random

def aeda_augment(text, target_start, target_end, spans, 
                 num_insert=3, chars=None):
    """
    对句子进行 AEDA 增强，返回增强文本和修正后的坐标。
    
    Args:
        text: 原始句子
        target_start, target_end: 目标词在原句中的位置（字符级）
        spans: [(start, end), ...] 论元 span 列表（可选）
        num_insert: 插入标点数量
        chars: 标点候选池
    
    Returns:
        new_text: 增强后的文本
        new_target: [start, end] 修正后的目标词位置
        new_spans: [(start, end), ...] 修正后的 span 列表
    """
    if chars is None:
        chars = ["。", "；", "？", "：", "！", "，"]

    # 1. 计算禁区（目标词内部 + 所有论元内部）
    forbidden = set()
    # 阻止在目标词内部插入：多字词的字间位置不可拆分
    for i in range(target_start + 1, target_end + 1):
        forbidden.add(i)
    # 论元内部：同理，阻止在论元字间插入
    for s, e in spans:
        for i in range(s + 1, e + 1):
            forbidden.add(i)
    
    # 2. 从安全位置中随机选择（不在任何禁区中，不是开头=0）
    safe_positions = [i for i in range(1, len(text) + 1) if i not in forbidden]
    if len(safe_positions) == 0:
        return text, [target_start, target_end], spans  # 无处可插，返回原文

    n = min(num_insert, len(safe_positions))
    positions = sorted(random.sample(safe_positions, n))
    
    # 3. 插入标点，同时跟踪累计偏移
    new_text_chars = list(text)
    for i, pos in enumerate(positions):
        punct = random.choice(chars)
        new_text_chars.insert(pos + i, punct)  # +i 是因为前面已插入的字符
    
    new_text = ''.join(new_text_chars)
    
    # 4. 计算坐标偏移：对每个插入位置，统计在 target/span 之前的数量
    def offset(pos):
        return sum(1 for p in positions if p <= pos)
    
    new_target = [target_start + offset(target_start),
                  target_end + offset(target_end)]
    
    new_spans = [[s + offset(s), e + offset(e)] for s, e in spans]
    
    return new_text, new_target, new_spans
```

### 2.4 各文件改动

#### `dataset_task1.py`

```python
# 在 __init__ 末尾，数据加载完成后

# Phase 3.3: AEDA augmentation
augmented = []
for item in self.all_data:
    text, target, frame = item['text'], item['target'], item['frame']
    new_text, new_target, _ = aeda_augment(text, target[0], target[1], [])
    augmented.append({
        'text': new_text,
        'target': new_target,
        'frame': frame,
        'sentence_id': item['sentence_id'] + 100000  # 防止 ID 冲突
    })
self.all_data.extend(augmented)
```

#### `dataset_task2.py`

```python
# 同 Task 1，但需要修正所有 span 坐标
augmented = []
for item in self.all_data:
    text, target, spans = item['text'], item['target'], item['spans']
    new_text, new_target, new_spans = aeda_augment(
        text, target[0], target[1], spans)
    augmented.append({
        'text': new_text,
        'target': new_target,
        'spans': new_spans,
        'sentence_id': item['sentence_id'] + 100000
    })
self.all_data.extend(augmented)
```

#### `dataset_task3.py`

```python
# Task 3 数据已经是 span 级别展开的
# 需要在 all_data 层面增强，然后重新展开

# 方案：在遍历 all_data 之前，先对 all_data 做增强
augmented_all_data = []
for line in self.all_data:
    text = line['text']
    target = [line["target"][-1]["start"], line["target"][-1]["end"]]
    spans = [[s['start'], s['end']] for s in line['cfn_spans']]
    
    new_text, new_target, new_spans = aeda_augment(
        text, target[0], target[1], spans)
    
    # 构造增强版本的 cfn_spans（坐标已修正）
    new_cfn_spans = []
    for i, s in enumerate(line['cfn_spans']):
        new_cfn_spans.append({
            'start': new_spans[i][0],
            'end': new_spans[i][1],
            'fe_abbr': s['fe_abbr'],
            'fe_name': s['fe_name']
        })
    
    augmented_all_data.append({
        'text': new_text,
        'target': [{'start': new_target[0], 'end': new_target[1], 
                    'pos': line['target'][-1]['pos']}],
        'frame': line['frame'],
        'cfn_spans': new_cfn_spans,
        'sentence_id': line['sentence_id'] + 100000
    })

# 将增强数据追加到 all_data
self.all_data.extend(augmented_all_data)
# 后续按原逻辑遍历 self.all_data 展开 span 样本
```

### 2.5 增强参数与 Epoch 调整

| 参数 | 默认值 | 说明 |
|------|:-----:|------|
| `aug_ratio` | 1 | 每条原始样本生成 1 条增强样本（训练集翻倍）|
| `num_insert` | 3 | 每次插入标点数（安全位置不足时自动减少）|
| `chars` | 中文标点池 | 插入的标点候选 |

**Epoch 调整**：数据翻倍后将 epoch 减半，总训练步数与增强前持平。

| 任务 | 原 epoch | 新 epoch | 理由 |
|------|:---:|:---:|------|
| Task 1 | 10 | **5** | 数据翻倍，步数不变 |
| Task 2 | 5 | **3** | 同上 |
| Task 3 | 10 | **5** | 同上 |

> 决策依据：砍 epoch 优于降比例（`aug_ratio=0.5`）。AED 提供的价值是多样性而非新知识——模型看到 2 个不同变体各 5 次，比死看 1 个变体 10 次更能学到底层语义。降比例则会减少增强样本的覆盖范围，且总步数反而更多。

---

## 三、影响评估

### 3.1 各任务受益分析

| 任务 | 预期受益 | 理由 |
|------|:--:|------|
| Task 1（框架识别）| **高** | 分类任务对文本表面变化敏感，AEDA 可增强鲁棒性 |
| Task 2（论元边界）| **中低** | span 检测依赖精确定位，标点插入可能引入噪声；但已有 FGM 抗噪基础 |
| Task 3（论元角色）| **中** | 受益于 Task 1 框架预测更准的间接提升，以及自身训练数据增多 |

### 3.2 预期提升

| 场景 | 总分预期 | 依据 |
|------|:------:|------|
| 乐观 | +1~2 分（→ 71.8~72.8 A 榜）| 训练集翻倍 + 抗噪能力增强 |
| 中性 | +0.5~1 分（→ 71.3~71.8）| 部分任务受益有限 |
| 悲观 | +0~0.5 分 | AEDA 对 span 定位任务可能轻微拖后腿 |

### 3.3 风险

| 风险 | 应对 |
|------|------|
| 标点插入打乱论元边界 → Task 2 变差 | 仅插入 2-3 个标点，不覆盖论元区域，降低噪声强度 |
| 训练集翻倍 → 训练时间翻倍 | epoch 减半（10→5, 5→3），总训练步数不变（已实施）|
| 增强样本质量差 → 所有任务变差 | dev 集验证，无提升则回退 |

---

## 四、实施步骤

### Step 1：编写 aeda.py 工具模块（30 分钟）

### Step 2：依次修改三个 Dataset（各 15 分钟）

### Step 3：本地验证——数据加载 + 坐标正确性抽查（30 分钟）

### Step 4：分别训练三个任务 + dev 评估（2-3 小时）

### Step 5：A/B 榜预测 + 提交

---

## 五、消融设计

为区分三个任务各自的增强收益，分三次提交：

```
E9a: E7 + AEDA on Task1 only     → 确认 Task1 收益
E9b: E9a + AEDA on Task2         → 确认 Task2 收益（或回退）
E9c: E9b + AEDA on Task3         → 确认 Task3 收益（或整体提交）
```

如果时间不够，直接三任务全开（E9），一次提交验证。

---

## 六、实验记录模板

| 实验编号 | 改动 | Task1 | Task2 | Task3 | 总分 | 备注 |
|----------|------|:---:|:---:|:---:|:---:|------|
| E9 | E7 + AEDA 全开 | ? | ? | ? | ? | 三任务训练集翻倍 |
| E9a | E7 + AEDA Task1 only | ? | ? | ? | ? | 消融 |
