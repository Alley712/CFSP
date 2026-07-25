# Phase 1：环境配置与代码兼容性修复

> **日期**：2026-07-25
> **目标**：配置 GPU 训练环境，修复 baseline 代码与新版依赖库的兼容性问题
> **结果**：✅ 完成，模型可加载，数据可读取，GPU 前向传播通过

---

## 1. 硬件与环境

| 项目 | 详情 |
|------|------|
| GPU | NVIDIA GeForce RTX 3050 Laptop (4GB VRAM) |
| Python | 3.10 |
| CUDA | 12.1 |
| PyTorch | 2.5.1+cu121 |
| transformers | 5.3.0 |
| numpy | 2.2.6 |

---

## 2. 环境搭建过程

### 2.1 问题：PyTorch CPU 版本

原环境安装的是 PyTorch CPU 版本（`2.12.0+cpu`），`torch.cuda.is_available()` 返回 `False`。国内 PyPI 镜像（清华、阿里云等）只提供 CPU 版本的 PyTorch wheel，CUDA 版本只能在 `download.pytorch.org` 下载。

### 2.2 解决方案

从 PyTorch 官网手动下载 `torch-2.5.1+cu121-cp310-cp310-win_amd64.whl`（约 2.4GB），使用迅雷等下载工具加速，然后本地安装：

```bash
pip install --no-deps "torch-2.5.1+cu121-cp310-cp310-win_amd64.whl"
```

使用 `--no-deps` 是因为 `pip 21.2.3` 存在 SSL 代理 bug（`check_hostname requires server_hostname`），无法访问 PyPI 验证依赖。所需依赖（filelock、fsspec、jinja2、typing-extensions、networkx）已提前安装。

### 2.3 torchvision 冲突

`torchvision 0.27.0` 是为 PyTorch 2.12 编译的，与 PyTorch 2.5.1 的 C++ 算子不兼容，会导致 `RuntimeError: operator torchvision::nms does not exist`。

**解决方案**：卸载 torchvision 和 easyocr。CFSP 是纯 NLP 任务，不涉及图像处理，不需要这些库。

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
GPU mem: 1.71 GB / 4 GB   ✅ 显存充足
```

---

## 7. 相关文件

- 设计文档：`docs/superpowers/specs/2026-07-25-cfsp-design.md`
- 任务详解：`docs/任务详解.md`
- 环境配置：`src/config.py`
- Baseline 代码：`baseline/`
