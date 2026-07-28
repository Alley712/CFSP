# Phase 3.2：Task 3 架构增强 —— 框架特征注入

> **日期**：2026-07-28
> **前置条件**：Phase 3.1 完成（E5/E6 RoBERTa 升级，A 榜 70.25 / B 榜 69.30）
> **目标**：将 Task 1 输出的框架信息注入 Task 3 模型，提升论元角色识别准确率
> **预计工期**：1.5 天
> **实施环境**：`newline` conda 环境（PyTorch 1.13.0 + transformers 4.24.0 + RoBERTa-wwm-ext）

---

## 一、问题分析

> 详细的概念解释和数据流拆解见：**`docs/notes/task3-architecture-explanation.md`**

**核心问题**：当前 Task 3 模型不知道目标词激活了哪个框架，只能从 **1009 个全局角色标签**中盲猜。实测数据验证了 `dataset_task3.py` 完全忽略了训练数据中的 `frame` 字段，框架与角色的从属关系全部丢失。

基线证据：E5 (RoBERTa) 相比 E3 (BERT) 的 Task3 F1 仅 +0.17（58.23 → 58.40），远小于 Task1 (+1.21) 和 Task2 (+0.48)，说明更强文本编码器对角色分类帮助有限，瓶颈在架构。

**解决方案**：在 GlobalPointer 输出的 1009 维角色 logits 上叠加一个框架偏置向量（`frame_bias`），该偏置由新增的 FrameEmbedding(713×256) + Linear(256×1009) 产生。这两层与 RoBERTa **完全无关**，是随机初始化、通过 Task3 训练从零学习的参数，总计约 44 万参数（占 BERT 的 0.4%）。

---

## 二、实施计划

### 2.1 总体架构变更

```
                        当前架构
         句子 + [1][2] → BERT → GlobalPointer → 角色 logits
                                 (100+ 类盲猜)

                        增强架构
         句子 + [1][2] → BERT → GlobalPointer → span 分数
         框架名称 ──→ Frame Embedding ──→ 投影 ──→ 融合 ──→ 角色 logits
                                                  (框架条件化)
```

**核心思路**：保持 GlobalPointer 的 span 评分能力不变，额外注入框架嵌入作为条件信号，让模型根据框架信息调整对不同角色的偏好。

### 2.2 修改范围一览

```
newline/
├── model_task3.py      ★ 核心改动：添加 FrameEmbedding + 融合层
├── dataset_task3.py    ★ 训练数据：每样本增加 frame_id
├── train_task3.py      ★ 训练脚本：传递 frame_ids 到模型
├── predict_task3.py    ★ 预测脚本：读取 Task1 输出获取 frame
└── params.py           (无需改动)
```

---

## 三、详细设计

### 3.1 `model_task3.py` — 架构增强

#### 新增组件

```python
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # === 保留原 GlobalPointer 所有组件 ===
        self.bert = BertModel(config)
        self.inner_dim = 64
        self.num_labels = config.num_labels        # 全局角色类别数 (~100+)
        self.dense = nn.Linear(config.hidden_size,
                               config.num_labels * self.inner_dim * 2)

        # === 新增：框架嵌入 ===
        self.num_frames = config.num_frames         # 框架种类数 (~713)
        self.frame_emb_dim = 256                    # 框架嵌入维度
        self.frame_embedding = nn.Embedding(
            self.num_frames, self.frame_emb_dim)

        # === 新增：融合层 ===
        # 将 GlobalPointer 的 span 分数和框架嵌入融合
        self.frame_proj = nn.Linear(
            self.frame_emb_dim, config.num_labels)  # 框架 → 角色偏置
```

#### 融合方式选择

提供**三种融合策略**，按优先级排列：

**策略 A（首选）：框架条件偏置（Frame-conditioned Bias）**

```
frame_embed → Linear(num_labels) → frame_bias
final_logits = globalpointer_logits + frame_bias
```

- 最简洁，参数量最小
- 直观含义：框架告诉模型该框架下哪些角色更可能、哪些不可能
- 对原有模型结构改动最小，易于调试

**策略 B（备选）：框架条件门控（Frame-conditioned Gating）**

```
frame_embed → Linear(num_labels) → Sigmoid → gate
final_logits = globalpointer_logits * gate
```

- 相当于框架信息对每个角色标签做软开关
- 可以直接将不可能的角色接近归零

**策略 C（可选）：门控偏置组合（Gated Bias）**

```
frame_embed → Linear(num_labels × 2) → gate, bias
final_logits = globalpointer_logits * gate + bias
```

- 结合 A 和 B 的优势
- 但参数量翻倍，先不优先尝试

**Phase 3.2 执行策略**：先实现策略 A，在 dev 集验证；如提升 <0.5 分，追加策略 B/C。

#### forward() 变更签名

```python
def forward(self,
            input_ids=None,         # 不变
            attention_mask=None,    # 不变
            target=None,            # span 位置 [start, end]（不变）
            frame_ids=None,         # ★ 新增：框架 ID (batch,)
            labels=None,            # 不变
            device=None,            # 不变
            for_test=False):        # 不变
    # 1. BERT 编码（不变）
    bert_out = self.bert(...)
    hidden_token = last_4_layers_avg(bert_out)

    # 2. GlobalPointer 计算 span 分数矩阵（不变）
    outputs = self.dense(hidden_token)
    qw, kw = split(outputs)
    qw, kw = apply_rope(qw, kw)
    logits = einsum('bmhd,bnhd->bhmn', qw, kw) / sqrt(dim)

    # 3. 提取 target 位置的分数（不变）
    token_logits = extract_target_position(logits, target)
    # token_logits: (batch, num_labels)

    # 4. ★ 框架特征注入（新增）
    if frame_ids is not None:
        frame_emb = self.frame_embedding(frame_ids)  # (batch, 256)
        frame_bias = self.frame_proj(frame_emb)      # (batch, num_labels)
        token_logits = token_logits + frame_bias     # 策略 A

    # 5. Loss 计算（不变）
    ...
```

#### 要点说明

1. **保留 GlobalPointer**：GlobalPointer 的 span 评分能力（通过 RoPE 位置编码 + QK 注意力）已经被验证有效，不改动其内部逻辑
2. **frame_ids 可为 None**：向后兼容，预测时如果缺 frame 可退化为原模型
3. **BertConfig 扩展**：需要动态设置 `config.num_frames`，不在 JSON 配置文件里写死

### 3.2 `dataset_task3.py` — 训练数据增强

#### 变更内容

每个训练样本新增 `frame_id` 字段：

```python
class Dataset(torch.utils.data.Dataset):
    def __init__(self, json_file, label_file, tokenizer, for_test=False):
        # === 新增：构建 frame_name → frame_id 映射 ===
        self.frame2idx = {}
        for i, line in enumerate(self.ori_labels):
            self.frame2idx[line["frame_name"]] = i
        # 注意：ori_labels 来自 frame_info.json，每个框架一条

        # ... 原有 label 映射不变 ...

        for line in self.all_data:
            text = line["text"]
            target = [...]  # 目标词位置
            frame_name = line["frame"]  # gold frame（训练时已知）
            frame_id = self.frame2idx[frame_name]  # ★ 新增

            for spans in line["cfn_spans"]:
                self.data.append({
                    ...
                    "frame_id": frame_id,  # ★ 新增
                    ...
                })

    def __getitem__(self, item):
        d1 = self.data[item]
        ...
        frame_id = d1["frame_id"]  # ★ 新增
        return input_ids, attention_mask, label_idx, label, sentence_id, frame_id
```

#### 要点

- `frame_id` 来自训练数据中的 gold frame（`line["frame"]`），不含噪声
- `frame2idx` 的构建顺序与 `task1` 的 `idx2label` 保持一致（都是按 frame_info.json 的顺序遍历），确保 frame_id 一一对应
- 返回的元组从 5 元素变为 6 元素（末尾加 `frame_id`）

### 3.3 `train_task3.py` — 训练脚本适配

#### 变更范围

**`get_model_input()` 函数**：新增 frame_ids 收集

```python
def get_model_input(data, device=None):
    # ... 现有逻辑 ...
    frame_ids = []
    for d in data:
        # ... 现有解析 ...
        frame_ids.append(d[5])  # ★ 新增：frame_id 是第 6 个元素

    frame_ids = torch.tensor(frame_ids, dtype=torch.long).to(device)  # ★
    return input_ids, attention_mask, target, labels, sentence_id, frame_ids
```

**模型初始化**：设置 num_frames

```python
config = BertConfig.from_json_file(args.config_file)
config.num_labels = train_dataset.num_labels
config.num_frames = len(train_dataset.frame2idx)  # ★ 新增
model = Model(config)
```

**train() 和 eval() 函数**：传递 frame_ids

```python
output = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    target=target,
    frame_ids=frame_ids,    # ★ 新增
    labels=labels,
    device=device,
    for_test=False
)
```

#### 超参数不变

Phase 3.2 保持训练超参数与 E5/E6 一致：
- lr = 2e-5
- batch_size = 3
- epochs = 10（先用 10 验证；如 dev 仍在提升则延至 15-20）
- NoisyTune = ON（保持与 baseline 一致）
- FGM = 注释（不启用，与 E5 一致）

### 3.4 `predict_task3.py` — 预测脚本适配

这是**最关键的变更**：预测时必须从 Task 1 输出中获取每个句子的预测框架。

#### 新增依赖

预测时需额外读取 Task 1 的输出文件：

```python
class Dataset(torch.utils.data.Dataset):
    def __init__(self, json_file, label_file, task1_file, task2_file, tokenizer):
        # task1_file: Task 1 的预测结果，格式 [[sentence_id, "框架名"], ...]
        with open(task1_file, 'r', encoding='utf8') as f:
            task1_data = json.load(f)

        # 构建 sentence_id → frame_name 映射
        self.sent2frame = {}
        for item in task1_data:
            self.sent2frame[item[0]] = item[1]  # sentence_id → frame_name

        # 构建 frame_name → frame_id 映射
        self.frame2idx = {}
        for i, line in enumerate(self.ori_labels):
            self.frame2idx[line["frame_name"]] = i

        # task2_data: Task 2 预测结果 [[sentence_id, start, end], ...]
        ...

        # 构建样本时注入 frame_id
        for line in self.task2_data:
            sent_id = line[0]
            frame_name = self.sent2frame.get(sent_id, self.ori_labels[0]["frame_name"])
            # ↑ 如果 Task1 未预测该句（理论上不应发生），fallback 到第一个框架
            frame_id = self.frame2idx[frame_name]
            self.data.append({
                ...
                "frame_id": frame_id,  # ★ 使用 Task1 预测的框架
                ...
            })
```

#### 要点

- 预测时使用 **Task 1 预测的框架**，不是 gold frame（测试集没有 gold frame）
- 这意味着如果 Task 1 预测错了框架，Task 3 的输入就是错的——这是 pipeline 架构的固有级联误差
- **可以通过改进 Task 1 间接提升 Task 3**（Phase 3.3-3.4 中考虑增强 Task 1）

#### 主程序入口

```python
if __name__ == '__main__':
    test_dataset = Dataset(
        "./dataset/cfn-test-B.json",     # 测试数据
        "./dataset/frame_info.json",     # 框架定义
        "./dataset/B_task1_test.json",   # ★ 新增：Task 1 预测输出
        "./dataset/B_task2_test.json",   # 原有：Task 2 预测输出
        tokenizer)
```

#### A/B 榜切换

`newline/` 使用 B 榜路径（`B_task1_test.json`, `B_task2_test.json`, `B_task3_test.json`）。
`baseline/` 如需同步修改则使用 A 榜路径（`A_task1_test.json` 等）。

### 3.5 代码改动清单

| 文件 | 改动类型 | 改动行数（估计）| 说明 |
|------|----------|:--------------:|------|
| `model_task3.py` | 架构增强 | +15 | 添加 Embedding + Linear + 融合逻辑 |
| `dataset_task3.py` | 数据流 | +8 | 每样本新增 frame_id |
| `train_task3.py` | 训练适配 | +6 | collate_fn 收集 frame_ids |
| `predict_task3.py` | 预测适配 | +15 | 读取 Task1 输出，构建 sent2frame 映射 |

总代码改动量：~50 行，属于**低风险改动**。

---

## 四、实施步骤

### Step 1：备份 + 创建实验分支（10 分钟）

```bash
cd /root/autodl-tmp/CFSP/newline

# 备份原始文件
mkdir -p backups/phase3.1
cp model_task3.py dataset_task3.py train_task3.py predict_task3.py backups/phase3.1/

# Git 分支（可选）
git checkout -b phase3.2-frame-injection
```

### Step 2：修改 `model_task3.py`（1 小时）

1. `__init__` 中新增 `self.frame_embedding` 和 `self.frame_proj`
2. `forward` 签名新增 `frame_ids=None`
3. 在 `token_logits` 计算后、loss 计算前插入融合逻辑
4. 确保 `frame_ids=None` 时行为等同于原模型（向后兼容）

### Step 3：修改 `dataset_task3.py`（0.5 小时）

1. 新增 `frame2idx` 映射字典
2. 遍历 `cfn_spans` 时附带 `frame_id`
3. `__getitem__` 返回元组末尾追加 `frame_id`

### Step 4：修改 `train_task3.py`（0.5 小时）

1. `get_model_input()` 收集 frame_ids
2. `config.num_frames` 设置
3. train/eval 函数传递 frame_ids

### Step 5：修改 `predict_task3.py`（1 小时）

1. Dataset 新增 `task1_file` 参数
2. 读取 Task1 预测结果，构建 `sent2frame` 映射
3. 样本生成时注入 `frame_id`
4. `get_model_input` 和 collate_fn 适配
5. 主程序传入 Task1 文件路径

### Step 6：本地验证（1 小时）

```bash
# 激活 newline 环境
conda activate newline

# 1. 数据加载验证（5 样本）
python -c "
from dataset_task3 import Dataset
from transformers import BertTokenizer
tokenizer = BertTokenizer(vocab_file='./chinese_roberta_wwm_ext/vocab.txt', do_lower_case=True)
ds = Dataset('./dataset/cfn-train.json', './dataset/frame_info.json', tokenizer)
print(f'Train size: {len(ds)}, num_labels: {ds.num_labels}, num_frames: {len(ds.frame2idx)}')
sample = ds[0]
print(f'Sample: {len(sample)} fields, frame_id={sample[5]}')
"

# 2. 模型前向传播验证
python -c "
import torch
from model_task3 import Model
from transformers import BertConfig
config = BertConfig.from_json_file('./chinese_roberta_wwm_ext/config.json')
config.num_labels = 120
config.num_frames = 713
model = Model(config)
# 构造 dummy input
batch_size = 2
seq_len = 50
input_ids = torch.randint(0, 21128, (batch_size, seq_len))
attn_mask = torch.ones(batch_size, seq_len)
target = [[5, 7], [10, 12]]
frame_ids = torch.tensor([100, 200])
labels = torch.tensor([5, 10])
out = model(input_ids=input_ids, attention_mask=attn_mask, target=target,
            frame_ids=frame_ids, labels=labels, device='cpu')
print(f'Loss: {out[\"loss\"].item():.4f}, Logits shape: {out[\"logits\"].shape}')
print('Forward pass OK')
"

# 3. 向后兼容验证（frame_ids=None 应正常工作）
python -c "
out = model(input_ids=input_ids, attention_mask=attn_mask, target=target,
            frame_ids=None, labels=labels, device='cpu')
print(f'Loss (no frame): {out[\"loss\"].item():.4f}')
print('Backward compatibility OK')
"
```

### Step 7：完整训练 + dev 评估（2-3 小时）

```bash
cd /root/autodl-tmp/CFSP/newline

# 训练 Task 3 (增强版)
python train_task3.py

# 预测 dev 集评估
# 注意：需要先有 Task1 和 Task2 的 dev 预测结果
# 如果没有，先用现有 best model 生成
python predict_task1.py  # 生成 dev 预测 (需临时修改输出路径)
python predict_task2.py  # 生成 dev 预测
python predict_task3.py  # 生成 dev 预测 (读取 Task1 + Task2 输出)
```

#### dev 评估对照

| 检查项 | 预期 |
|--------|------|
| Loss 下降速度 | 与 Phase 3.1 相近或略快 |
| Dev acc（角色分类正确率）| 相对 Phase 3.1 提升 2-5% |
| frame_ids=None vs 有 frame_ids | 有明显差异 → 框架信息确实在起作用 |
| OOM | 不应出现（仅增加 ~0.2M 参数）|

### Step 8：A/B 榜提交（按需）

如果 dev 验证有提升（acc 相对 E5 提升 ≥ 1%）：

```bash
# A 榜：切换到 baseline 目录
cd /root/autodl-tmp/CFSP/baseline
conda activate baseline
# 将 newline 修改同步到 baseline（或直接在 baseline 上重做修改）
python predict_task1.py  # 生成 A_task1_test.json
python predict_task2.py  # 生成 A_task2_test.json
python predict_task3.py  # 生成 A_task3_test.json（读取 Task1 + Task2）
cd dataset
zip submit_task3_enhanced.zip A_task1_test.json A_task2_test.json A_task3_test.json
cp submit_task3_enhanced.zip ../../submissions/List_A/

# B 榜：在 newline 目录
cd /root/autodl-tmp/CFSP/newline
conda activate newline
python predict_task1.py  # 生成 B_task1_test.json
python predict_task2.py  # 生成 B_task2_test.json
python predict_task3.py  # 生成 B_task3_test.json
cd dataset
zip submit_task3_enhanced.zip B_task1_test.json B_task2_test.json B_task3_test.json
cp submit_task3_enhanced.zip ../../submissions/List_B/
```

---

## 五、实验记录模板

在 `experiments/exp_log.md` 中新增以下行：

```markdown
| **E7** | **E5 + Task3 框架特征注入 (策略A)** | ? | ? | ? | ? | A 榜，框架偏置融合 |
| **E8** | **E6 + Task3 框架特征注入 (策略A)** | ? | ? | ? | ? | B 榜，框架偏置融合 |
```

如果策略 A 效果不理想，追加：

```markdown
| **E7b** | E7 + 策略B (框架门控) | ? | ? | ? | ? | 门控替代偏置 |
| **E7c** | E7 + 策略C (门控+偏置) | ? | ? | ? | ? | 组合方案 |
```

---

## 六、预期效果与风险评估

### 6.1 预期提升

| 场景 | Task3 F1 预期 | 总分预期 | 依据 |
|------|:-----------:|:------:|------|
| 乐观 | +3~5 分 | 71.5~72.5 | 框架信息是强先验，大幅缩小标签空间 |
| 中性 | +1~3 分 | 70.8~71.5 | 部分角色跨框架共享，边际收益递减 |
| 悲观 | +0~1 分 | 70.3~70.8 | 模型可能已通过目标词隐式学到了框架信息 |

### 6.2 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|:----:|------|------|
| Task 1 预测框架错误 → Task 3 级联错误 | 中 | Task3 F1 不升反降 | ① 训练时用 gold frame（无噪声）；② 预测时如果效果反降，用 Task1 top-3 做 beam search |
| 框架嵌入过拟合 | 低 | dev 好、test 差 | 对 frame_embedding 加 dropout (0.1~0.3) |
| `num_frames` 不一致 | 低 | 训练崩溃 | 训练和预测都从 `frame_info.json` 加载，保证 frame_id 映射一致 |
| 新增参数导致显存不足 | 极低 | OOM | 仅增加 ~0.2M 参数（713×256 + 256×120 ≈ 0.2M），batch_size=3 下约增加 10MB 显存 |

### 6.3 快速回退方案

如果 dev 验证无提升或变差：

```bash
cd /root/autodl-tmp/CFSP/newline
cp backups/phase3.1/model_task3.py .
cp backups/phase3.1/dataset_task3.py .
cp backups/phase3.1/train_task3.py .
cp backups/phase3.1/predict_task3.py .
# 恢复 E5/E6 状态
```

---

## 七、与其他 Phase 的衔接

```
Phase 3.2 ──→ Phase 3.3 数据增强
    │               │
    │   如果框架注入有效，数据增强时：
    │   ① Task3 的增强样本也携带 frame_id
    │   ② AEDA 随机插入不影响 frame 标签
    │   ③ 同义词替换不改变框架归属
    │
    └──→ Phase 3.4 训练技巧
            │
            FGM 全开 + EMA 可在增强模型上叠加
            需验证 FGM 与 frame_embedding 的互动
```

---

## 八、关键决策记录

1. **在 newline（RoBERTa）环境开发，在 baseline（BERT）环境同步**：newline 已有 RoBERTa 跑通的 E5/E6，以此为基准；baseline 同步修改用于 A 榜提交对比
2. **保留 GlobalPointer 不改动**：GlobalPointer 的 span 评分能力已验证有效，只在其输出上叠加框架信号，降低回归风险
3. **训练时用 gold frame，预测时用 Task1 输出**：训练阶段不引入 Task1 的预测噪声，确保模型学到的是正确映射；预测时实打实使用 pipeline 输出
4. **策略 A（偏置融合）优先**：最简洁，最不容易过拟合，最容易调试
