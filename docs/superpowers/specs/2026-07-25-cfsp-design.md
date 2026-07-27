# CFSP 项目设计方案

> 日期：2026-07-25 | 截止：2026-08-01 12:00

## 一、项目概述

汉语框架语义解析（CFSP）比赛，三个子任务：
- Task 1：框架识别（分类，权重 0.3）
- Task 2：论元范围识别（span 检测，权重 0.3）
- Task 3：论元角色识别（span + 角色分类，权重 0.4）

目标：在 baseline 69.00 基础上提升至 **75-80 分**。

## 二、环境约束

- GPU：RTX 3090 租用卡 24GB VRAM（AutoDL 等平台）
- 方案：FP16 混合精度（默认），batch_size=8，无需梯度累积

## 三、技术选型

| 组件 | 选择 | 理由 |
|------|------|------|
| 预训练模型 | `hfl/chinese-roberta-wwm-ext` | 比 BERT-wwm 效果好，中文 NLP 通用 baseline |
| 微调框架 | HuggingFace Transformers + PyTorch | 最通用、文档丰富 |
| 精度 | FP16 混合精度（torch.cuda.amp） | 加速训练，3090 24GB 显存充裕 |
| 任务架构 | 基于 baseline 的 GlobalPointer 改进 | 保留成熟部分，针对性改进薄弱环节 |

## 四、项目结构

```
D:/CFSP/
├── data/
│   ├── cfn-dataset/          # 原始数据集（已存在）
│   └── raw_zips/             # 原始压缩包
│       ├── cfn-dataset.zip
│       └── CFN-B.zip
├── baseline/                 # 官方 baseline 代码（旧依赖环境：PyTorch 1.13）
│   ├── chinese_bert_wwm_ext/ # BERT 预训练模型
│   ├── dataset/              # 数据集副本
│   ├── saves/                # 各任务模型权重
│   ├── dataset_task1-3.py    # 数据加载
│   ├── model_task1-3.py      # 模型定义（GlobalPointer）
│   ├── train_task1-3.py      # 训练脚本
│   ├── predict_task1-3.py    # 预测脚本
│   ├── params.py             # 超参数配置
│   └── requirements.txt
├── newline/                  # 重构版 baseline（新依赖环境：PyTorch 2.1）
│   ├── chinese_bert_wwm_ext/ # BERT 预训练模型
│   ├── dataset/              # 数据集副本
│   ├── saves/                # 各任务模型权重
│   ├── dataset_task1-3.py
│   ├── model_task1-3.py
│   ├── train_task1-3.py
│   ├── predict_task1-3.py
│   ├── params.py
│   └── requirements.txt
├── models/                   # 预训练模型权重目录
├── submissions/              # 天池提交文件
│   ├── List_A/
│   │   ├── submit_1st.zip    # 首次打榜（新环境，Score 37.39）
│   │   └── submit_2nd(best).zip  # 降级复现（Score 69.68）
│   └── List_B/
│       └── submit_1st(best).zip  # B 榜复现（Score 68.80）
├── experiments/              # 实验记录
│   ├── phase1-环境配置记录.md
│   └── phase2-问题修复记录.md
├── downloads/                # 手动下载的文件
│   └── torch-2.5.1+cu121-cp310-cp310-win_amd64.whl
└── docs/
    ├── 任务介绍.md
    ├── 任务详解.md
    ├── baseline代码逻辑分析.md
    ├── 于艳华夏令营机器学习考核须知.md
    └── superpowers/
        └── specs/
            └── 2026-07-25-cfsp-design.md
```

> **说明**：`baseline/` 和 `newline/` 是同一套代码的两个副本，分别对应旧版 (PyTorch 1.13) 和新版 (PyTorch 2.1) 依赖环境。当前以 `baseline/`（旧版环境）为主进行训练和提交，因为版本差异对 Task 2 自定义 loss 的影响已验证（详见 `experiments/phase2-问题修复记录.md`）。

## 五、分阶段执行计划

### 阶段 1：环境配置（预计 0.5 天）

**目标**：可以在 3050 上正常训练 baseline。

1. 安装 PyTorch（CUDA 12.x 版）+ transformers + tqdm
2. 下载 `hfl/chinese-roberta-wwm-ext` 模型权重到 `models/`
3. 验证 CUDA + FP16 可用
4. 修改 baseline 代码中的模型路径，跑一个小批次确认无误

**成功标准**：`python src/task1/train.py` 能正常启动训练，GPU 利用率 > 80%，无 OOM。

### 阶段 2：跑通 Baseline（预计 0.5 天）

**目标**：获取天池 A 榜 baseline 分数。

1. 分别训练三个任务（使用 chinese-roberta-wwm-ext）
2. 生成 A_task{1,2,3}_test.json 预测文件
3. 打包 submit.zip 提交到天池
4. 记录 A 榜分数

**成功标准**：天池 A 榜有成绩，分数接近或超过 69.00。

### 阶段 3：改进迭代（预计 3 天）

#### 3.1 预训练模型升级（优先级最高，半天）

```
BERT-wwm-ext → RoBERTa-wwm-ext
```

改动范围：替换 `config_file` / `vocab_file` / `init_checkpoint` 路径。

RoBERTa 相比 BERT 的改进：
- 动态掩码
- 去掉 NSP 任务
- 更大 batch、更长训练
- 中文 NLP 任务上通常 +1~3 个点

**风险**：RoBERTa 词表和 BERT 不同，tokenizer 加载方式需要改（`BertTokenizer` → `AutoTokenizer` 或 `RobertaTokenizer`）。

#### 3.2 Task 3 架构增强（优先级最高，1.5 天）

Baseline 的 Task 3 模型结构与 Task 1 完全相同——仅用 BERT + GlobalPointer 做分类。问题在于：

- Task 3 需要同时利用**框架信息**（Task 1 的输出）和**论元边界**（Task 2 的输出）
- 但 baseline 把这三个任务完全独立训练，Task 3 只用到了句子文本

**改进方案**：

1. **框架特征注入**：将框架名称编码为一个固定的 embedding，拼接到 BERT 输出的 [CLS] token 上
2. **目标词位置编码增强**：baseline 已通过特殊 token 标记目标词，考虑追加可学习的位置编码
3. **负采样策略**：对每个正确的论元，随机采样 2-3 个负样本（句中其他 span），缓解正负样本不平衡

```
输入: 句子 + 目标词 + 框架名
        ↓
   BERT 编码
        ↓
   [CLS] + 框架embedding → 用于分类的特征
        ↓
   GlobalPointer 解码 → 每个 span 得分最高的角色标签
```

#### 3.3 数据增强（1 天）

对训练集做简单增强，增加数据多样性：

1. **AEDA**（随机插入标点）：已在 baseline dataset 中引用但未启用
2. **同义词替换**：用 synonyms 库随机替换非目标词、非论元词
3. **回译**（可选）：中→英→中，用翻译 API

增强策略：每条原始数据生成 1-2 条增强数据，训练集扩大 ~2 倍。

#### 3.4 训练技巧调优（穿插进行）

- **FGM 对抗训练**：baseline 已实现但仅在 Task 2 启用。在 Task 1、Task 3 也启用
- **NoisyTune**：baseline 已有，用参数噪声增强鲁棒性
- **warmup + 余弦退火**：替代 linear warmup + linear decay
- **EMA**（指数移动平均）：对模型参数做平滑，提升泛化

### 阶段 4：消融实验 + 分析（预计 2 天）

**目标**：系统记录每项改进的贡献，为 PPT 提供素材。

**实验记录格式**（`experiments/exp_log.md`）：

```markdown
| 实验编号 | 改动 | Task1_acc | Task2_F1 | Task3_F1 | 总分 | 备注 |
|----------|------|-----------|----------|----------|------|------|
| E0       | Baseline (BERT-wwm) | 70.83 | 83.06 | 57.08 | 69.00 | 复现 |
| E1       | 替换 RoBERTa-wwm | ? | ? | ? | ? | one-line change |
| E2       | E1 + Task3 架构增强 | ? | ? | ? | ? | |
| E3       | E2 + 数据增强 | ? | ? | ? | ? | |
| E4       | E3 + FGM全开 | ? | ? | ? | ? | |
```

**错误分析**（重点在 Task 3）：
- 随机抽取 50 个预测错误的 case
- 分类：框架识别错误导致 vs. 论元边界错误导致 vs. 角色标签混淆
- 典型案例写到 PPT

### 阶段 5：报告 + 提交（预计 1 天）

**PPT 内容大纲**（≤10 页）：

1. **题目背景**：什么是 CFSP，三个子任务简介
2. **数据与框架**：数据格式、框架体系、统计信息
3. **模型架构**：整体 pipeline + 各任务模型结构图
4. **实验设计**：改进项列表 + 预期效果
5. **实验结果**：消融实验表格（每个改动的贡献）
6. **错误分析**：典型错误 case + 分类饼图
7. **对比分析**：不同预训练模型、不同方法的优劣
8. **改进方向**：未完成的想法、可能的进一步优化
9. **参考来源**：Baseline 代码、预训练模型、论文/博客
10. **总结**

**提交**：
- 最终 submit.zip 提交到天池
- PPT 发送至 `shuiyiyihan@bupt.edu.cn`
- 命名：`学校_姓名_夏令营解题报告.ppt`

## 六、风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| 3090 租用实例不稳定 | 低 | 定期保存 checkpoint，代码和数据在本地有备份 |
| RoBERTa 词表不兼容 | 中 | 改为 `AutoTokenizer.from_pretrained()`，自动处理 |
| Task 3 架构改动不 work | 中高 | 回退到 baseline 版本，至少跑出结果 |
| 数据增强降低性能 | 低 | 只增强 1 倍而非 2 倍，确保增强质量 |
| 时间不够 | 中 | 优先保证阶段 1-2 完成，剩余改进按优先级砍掉 |

## 七、关键决策记录

1. **不做联合训练**：零基础在 7 天内实现三个任务的联合训练风险太高，改为 pipeline 独立训练 + Task 3 利用 Task 1 输出
2. **不引入外部数据**：CFN 数据集协议为 CC BY-NC 4.0，不混入外部标注数据
3. **PPT 优先于刷分**：30% 的 PPT 评分要求系统性的实验记录和分析，这比多跑几个模型更重要
