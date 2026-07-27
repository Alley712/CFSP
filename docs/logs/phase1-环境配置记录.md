# Phase 1：环境配置与代码兼容性修复

> **日期**：2026-07-25
> **目标**：配置 GPU 训练环境，修复 baseline 代码与新版依赖库的兼容性问题
> **结果**：✅ 完成，模型可加载，数据可读取，GPU 前向传播通过

---

## 1. 硬件与环境

| 项目 | 详情 |
|------|------|
| GPU | NVIDIA GeForce RTX 3090 (24GB VRAM，租用) |
| Python | 3.10 |
| CUDA | 11.8 |
| PyTorch | 2.1.2+cu118 |
| transformers | 4.36.0 |
| numpy | 1.24.x |

---

## 2. 环境搭建过程

### 2.1 硬件选择

本机 GPU（RTX 3050 Laptop 4GB）显存不足以支撑 BERT 系模型训练，改用 AutoDL 等平台租用 RTX 3090（24GB VRAM），时租费用约 ¥1-2/h。

### 2.2 环境初始化

租用实例预装 CUDA 11.8 + Python 3.10，基础包（PyTorch 2.1.2、numpy 等）已就绪，只需额外安装：

```bash
pip install transformers==4.36.0 tqdm
```

### 2.3 环境变量

```bash
export CUDA_VISIBLE_DEVICES=0
```

3090 单卡即可满足训练需求（batch_size=8 时显存占用约 3-4 GB），无需多卡或梯度累积。

---

## 3. 预训练模型

下载了两个中文预训练模型到 `models/` 目录：

| 模型 | 来源 | 参数规模 | 词表大小 |
|------|------|----------|----------|
| chinese-bert-wwm-ext | HFL | 102M | 21,128 |
| chinese-roberta-wwm-ext | HFL | 102M | 21,128 |

`models/` 目录已加入 `.gitignore`，不纳入版本管理（模型文件约 390MB）。

当前 Phase 1 使用 BERT 版本；RoBERTa 版本留待 Phase 2-3 使用。

---

## 4. 代码兼容性修复

baseline 代码为 2023 年编写，依赖库 API 在 2025-2026 年间有破坏性变更。共修复了 **4 类问题**，涉及 **10 个文件**。

### 4.1 `encode_plus()` → `tokenizer(is_split_into_words=True)`

**原因**：新版 transformers 移除了 `BertTokenizer.encode_plus()` 方法。

**旧代码**：
```python
data = self.tokenizer.encode_plus(list(d1['text']))
```

**新代码**：
```python
data = self.tokenizer(list(d1['text']), is_split_into_words=True)
```

> `is_split_into_words=True` 表示输入已经是字符级别的列表（如 `['警', '方', '逮', '捕', '了', '嫌', '疑', '人']`），tokenizer 不需要再做分词。

**涉及文件**：`dataset_task1.py`、`dataset_task2.py`、`dataset_task3.py`、`predict_task1.py`、`predict_task2.py`、`predict_task3.py`

### 4.2 `BertTokenizer(vocab_file=...)` → `AutoTokenizer.from_pretrained(model_dir)`

**原因**：新版 transformers 中，`BertTokenizer(vocab_file=...)` 只显式指定词表文件时，只能加载 5 个特殊 token（`[PAD]`、`[UNK]`、`[CLS]`、`[SEP]`、`[MASK]`），其余 21,123 个词全部丢失。使用 `AutoTokenizer.from_pretrained()` 从模型目录加载才能正确读取完整词表。

**旧代码**：
```python
tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
```

**新代码**：
```python
tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
```

**涉及文件**：`train_task1.py`、`train_task2.py`、`train_task3.py`、`predict_task1.py`、`predict_task2.py`、`predict_task3.py`

### 4.3 `np.compat.long` → `np.int64`

**原因**：`np.compat` 是 Python 2 兼容层，NumPy 2.0 中已移除。

**旧代码**：
```python
input_ids = np.array(input_ids_list, dtype=np.compat.long)
```

**新代码**：
```python
input_ids = np.array(input_ids_list, dtype=np.int64)
```

**涉及文件**：`train_task1.py`、`train_task2.py`、`train_task3.py`、`predict_task1.py`、`predict_task2.py`、`predict_task3.py`

### 4.4 数据路径更新

数据集实际存放在 `data/cfn-dataset/`，而 baseline 代码默认路径为 `./dataset/`。已将所有硬编码路径从 `./dataset/` 替换为 `../data/cfn-dataset/`。

### 4.5 `params.py` 配置更新

在 `params.py` 中新增 `--model_dir` 参数，各路径默认值指向 `../models/chinese-bert-wwm-ext/`：

```python
parser.add_argument("--model_dir",
                    default='../models/chinese-bert-wwm-ext',
                    type=str,
                    help="Pretrained model directory")
```

---

## 5. 新增源代码结构

创建了 `src/` 目录作为现代化代码的起点：

```
src/
├── config.py          # 统一配置（数据路径、模型路径等）
├── params.py          # 训练超参数
├── task1/
│   ├── dataset.py     # Task 1 现代 Dataset 实现
│   ├── model.py       # Task 1 模型（从 baseline 复制）
│   └── train.py       # Task 1 训练脚本（FP16 + 梯度累积）
├── task2/
│   └── model.py       # Task 2 模型（从 baseline 复制）
└── task3/
    └── model.py       # Task 3 模型（从 baseline 复制）
```

> `src/` 的代码目前是一个**框架**，后续 Phase 2-3 将在此基础上扩展。

---

## 6. 验证结果

```text
Vocab size: 21128         ✅ 词表完整加载
Hidden size: 768          ✅ BERT 配置正确
Model params: 115 M       ✅ 模型参数规模正常
Train size: 10700         ✅ 训练集读取成功
Dev size: 2300            ✅ 验证集读取成功
Num labels: 713           ✅ 框架类别数
Device: cuda              ✅ GPU 可用
Loss: 6.84                ✅ 前向传播正常（随机初始化）
GPU mem: 1.71 GB / 24 GB  ✅ 显存充足
```

---

## 7. 相关文件

- 设计文档：`docs/superpowers/specs/2026-07-25-cfsp-design.md`
- 任务详解：`docs/任务详解.md`
- 环境配置：`src/config.py`
- Baseline 代码：`baseline/`
