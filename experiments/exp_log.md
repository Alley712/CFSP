# 实验记录

> 目标：从 baseline 69.00 提升至 75-80 分
> 提交截止：2026-08-01 12:00

## 实验总表

| 实验编号 | 改动 | Task1 Acc | Task2 F1 | Task3 F1 | 总 Score | 备注 |
|----------|------|-----------|----------|----------|----------|------|
| E0 | Baseline 官方（BERT-wwm, PyTorch 1.13, transformers 4.24） | 70.83 | 83.06 | 57.08 | 69.00 | 官方报告分数 |
| E1 | 新环境直接运行（PyTorch 2.1.2, transformers 5.14.1） | — | — | — | — | 无法启动，GradScaler / AdamW 导入错误 |
| E2 | E1 + 兼容性修复（PyTorch 2.7.1, transformers 4.36） | 62.08 | 42.63 | 14.95 | 37.39 | A 榜首次提交，NoisyTune ON |
| E3 | **降级复现 A 榜**（PyTorch 1.13, transformers 4.24） | 70.79 | 83.82 | 58.23 | **69.68** | 不做代码修改，NoisyTune ON |
| E4 | **降级复现 B 榜**（PyTorch 1.13, transformers 4.24） | 70.81 | 84.29 | 55.68 | **68.80** | B 榜测试集，NoisyTune ON |
| **E5** | **E3 + RoBERTa-wwm-ext**（PyTorch 1.13, transformers 4.24） | **72.00** | **84.30** | **58.40** | **70.25** | 仅换模型，其余不变，A 榜 |

## Task 2 消融实验（dev 集）

> 环境：PyTorch 2.7.1, transformers 4.36, BERT-wwm-ext

| 实验编号 | Loss 函数 | NoisyTune | Epochs | Dev F1 | Precision | Recall | 备注 |
|----------|-----------|-----------|--------|--------|-----------|--------|------|
| T2-E0 | 原始 GlobalPointer `log(1+sum(exp(...)))` | ON | 5 | 0.002 | — | — | 原始代码不改，近乎全零 |
| T2-E1 | 原始 GlobalPointer | ON | 5 | 0.037 | 34.2% | 1.9% | Phase 2 初版 |
| T2-E2 | BCEWithLogitsLoss + pos_weight=50 | OFF | 5 | 0.069 | 3.8% | 46.9% | epoch 2 峰值 F1=0.086 |
| T2-E3 | 原始 GlobalPointer | OFF | 5 | — | — | — | 收敛中但 epoch 不足 |
| T2-E4 | **原始 GlobalPointer** | **OFF** | **20** | **0.393** | 51.0% | 32.0% | epoch 10 达峰值，11-20 plateau |

### T2 消融结论

- NoisyTune 在新版 PyTorch 下对自定义 loss 有灾难性影响（关掉后 F1 ×10+）
- BCEWithLogitsLoss 方案方向正确但受限于极端类别不平衡（2000:1）
- 最终回归原始 GlobalPointer loss + 关 NoisyTune + 延长至 20 epoch，dev F1 从 0.037 → 0.393

## 降级复现详细数据

### E3 — A 榜（2026-07-27 16:00:48）

| 指标 | Baseline 官方 | 复现结果 | 差值 |
|------|:-----------:|:------:|:----:|
| Task1 Acc | 70.83 | 70.79 | -0.04 |
| Task2 F1 | 83.06 | **83.82** | +0.76 |
| Task2 Precision | — | 86.99 | — |
| Task2 Recall | — | 80.88 | — |
| Task3 F1 | 57.08 | **58.23** | +1.15 |
| Task3 Precision | — | 58.01 | — |
| Task3 Recall | — | 58.45 | — |
| **总 Score** | **69.00** | **69.68** | **+0.68** |

### E4 — B 榜（2026-07-27 16:40:23）

| 指标 | Baseline 官方 | 复现结果 | 差值 |
|------|:-----------:|:------:|:----:|
| Task1 Acc | 70.83 | 70.81 | -0.02 |
| Task2 F1 | 83.06 | **84.29** | +1.23 |
| Task2 Precision | — | 87.01 | — |
| Task2 Recall | — | 81.73 | — |
| Task3 F1 | 57.08 | 55.68 | -1.40 |
| Task3 Precision | — | 55.62 | — |
| Task3 Recall | — | 55.73 | — |
| **总 Score** | **69.00** | **68.80** | **-0.20** |

### E5 — RoBERTa 升级 A 榜（2026-07-27 22:09:47）

> 环境：PyTorch 1.13, transformers 4.24, chinese-roberta-wwm-ext
> 改动：仅将预训练模型从 BERT-wwm-ext 替换为 RoBERTa-wwm-ext，其余配置不变（epoch=5, NoisyTune ON, FGM 仅 Task 2）

| 指标 | E3 (BERT-wwm) | E5 (RoBERTa-wwm) | 差值 |
|------|:-----------:|:------:|:----:|
| Task1 Acc | 70.79 | **72.00** | +1.21 |
| Task2 F1 | 83.82 | **84.30** | +0.48 |
| Task2 Precision | 86.99 | 87.13 | +0.14 |
| Task2 Recall | 80.88 | 81.65 | +0.77 |
| Task3 F1 | 58.23 | **58.40** | +0.17 |
| Task3 Precision | 58.01 | 57.95 | -0.06 |
| Task3 Recall | 58.45 | 58.85 | +0.40 |
| **总 Score** | **69.68** | **70.25** | **+0.57** |

### 分析

- **三项全涨**：Task1 +1.21、Task2 +0.48、Task3 +0.17，总分突破 70 大关
- **Task1 受益最大**（+1.21）：框架识别是分类任务，RoBERTa 更强的语义表征直接提升了分类准确率
- **Task3 提升最小**（+0.17）：仍然受限于级联依赖——Task2 输出的边界质量虽有改善但仍是瓶颈
- 这是一个"零成本"提升（只改一行模型路径），性价比极高，验证了设计文档中"预训练模型升级优先级最高"的判断

## 版本差异根因总结

```
PyTorch 版本升级 (1.13 → 2.x)
    │
    ├── AdamW / LayerNorm / Attention 内部实现变化
    │       └── BERT 每层输出的微小数值漂移（12层累积）
    │               ├── Task 1 (CrossEntropy): 影响可控 → 差距 12.4%
    │               └── Task 2 (log(1+sum(exp(...)))): 数值放大 → loss 梯度失效
    │                       ├── NoisyTune 叠加噪声 → 梯度被完全淹没
    │                       └── Task 3 级联依赖 → 误差传播 → 差距 73.8%
    └── 结论：版本差异是根因，Task 2 的自定义 loss 是放大器，NoisyTune 是加速器
```

## 后续计划

- [x] E5: 替换 RoBERTa-wwm-ext（在 PyTorch 1.13 环境上）✅ **70.25**
- [ ] E6: E5 + Task 3 架构增强（框架特征注入）
- [ ] E7: E6 + 数据增强（AEDA）
- [ ] E8: E7 + FGM 全开 + EMA
