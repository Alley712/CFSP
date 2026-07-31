# Phase 4 方向 3 扩展：5-Seed Ensemble

> 日期：2026-07-31
> 状态：3-seed Ensemble 已完成（E21），计划扩展至 5-seed
> 截止：2026-08-01 12:00

---

## 一、动机

### 1.1 3-seed Ensemble 回顾（E21）

| | Task1 | Task2 | Task3 | 总分 |
|------|:--:|:--:|:--:|:--:|
| A 榜 | 71.67 | 83.69 | 59.78 | **70.52** |
| B 榜 | 71.78 | 84.32 | 57.85 | **69.97** |
| Dev | 73.28 | 82.74 | 57.87 | **69.95** |

3-seed 相比单模型（E12/E13）提升 +0.49~0.54 总分，验证了 Ensemble 方向有效。

### 1.2 为什么要扩展到 5-seed

1. **Ensemble 收益通常随 seed 数增加而递减但持续**：从 1→3 涨了 0.5，3→5 预计再涨 0.1~0.3
2. **投票更稳定**：Task 2 的 Span Majority Voting 在 3-seed 时阈值是 ≥2（即 2/3 或 3/3），5-seed 阈值可设为 ≥3（即 3/5、4/5、5/5），投票结果更鲁棒
3. **Soft Voting 受益于更多独立采样**：Task 1/3 的 logits 平均随着模型数增加趋向更稳定的估计
4. **边际成本低**：已有 3 套权重，只需追加训练 2 套

### 1.3 收益预估

| seed 数 | 预计 A 榜 | 相比 1-seed 提升 | 相比 3-seed 提升 |
|---------|:--------:|:-------------:|:-------------:|
| 1 (E12) | 70.03 | — | — |
| 3 (E21) | 70.52 | +0.49 | — |
| 5 (E22) | **70.65~70.85** | +0.62~0.82 | +0.13~0.33 |

---

## 二、新增 Seed 选择

| Seed | 用途 | 状态 |
|------|------|:--:|
| 42 | E12 备份权重 | ✅ 已有 |
| 123 | Phase 4.3 训练 | ✅ 已有 |
| 456 | Phase 4.3 训练 | ✅ 已有 |
| **789** | 新增 | 🔲 待训练 |
| **1024** | 新增 | 🔲 待训练 |

Seed 值选 789 和 1024：与已有 seed（42/123/456）间隔足够大，确保不同的初始化轨迹。

---

## 三、训练计划

### 3.1 训练配置

沿用 E12/E13 配置（commit `720024b`），不启用 Phase 4 的 FGM+Cos 实验改动：

| 参数 | Task 1 | Task 2 | Task 3 |
|------|:--:|:--:|:--:|
| Epochs | 10 | 5 | 10 |
| FGM | ❌ | ✅ | ❌ |
| NoisyTune | ✅ | ✅ | ✅ |
| Scheduler | Linear | Linear | Linear |
| LR | 2e-5 | 2e-5 | 2e-5 |
| Batch size | 3 | 3 | 3 |

### 3.2 GPU 分配与时间

3 张 RTX 3090，2 个新 seed 同时训练：

```
GPU 0: seed 789  (Task1 → Task2 → Task3) ≈ 7h
GPU 1: seed 1024 (Task1 → Task2 → Task3) ≈ 7h
GPU 2: idle / 备用
```

**墙钟时间：~7h**（与 3-seed 训练相同，因为新 seed 数也是 2 个）。

### 3.3 执行步骤

```bash
cd /root/autodl-tmp/CFSP/newline

# Seed 789
CUDA_VISIBLE_DEVICES=0 python train_task1.py --seed 789
CUDA_VISIBLE_DEVICES=0 python train_task2.py --seed 789
CUDA_VISIBLE_DEVICES=0 python train_task3.py --seed 789

# Seed 1024
CUDA_VISIBLE_DEVICES=1 python train_task1.py --seed 1024
CUDA_VISIBLE_DEVICES=1 python train_task2.py --seed 1024
CUDA_VISIBLE_DEVICES=1 python train_task3.py --seed 1024
```

或使用 `train_parallel.sh`，修改 SEEDS 数组为 `(789 1024)`。

---

## 四、Ensemble 策略调整

### 4.1 Task 1 — Soft Voting（不变）

5 模型 logits 取平均 → argmax。逻辑不变。

### 4.2 Task 2 — Span Majority Voting（阈值调整）

| seed 数 | 阈值 | 含义 |
|:--:|:--:|------|
| 3 | ≥2 | 2/3 或 3/3 模型同意 |
| **5** | **≥3** | **3/5, 4/5, 5/5 模型同意** |

阈值从 ≥2 提升到 ≥3，对 span 质量要求更严格。预计提高 Precision、略微降低 Recall，与 E21 的 Task2 变化趋势一致（P↑R↓）。

### 4.3 Task 3 — Soft Voting（不变）

5 模型 logits 取平均 → argmax。逻辑不变。

### 4.4 代码改动

`ensemble_predict.py`：`--seeds` 默认值从 `42,123,456` 改为 `42,123,456,789,1024`，或通过命令行传入。

Task 2 投票阈值：`threshold = max(2, (n_models + 1) // 2)`。n_models=5 时 threshold=3，公式无需改动。

---

## 五、评估计划

### 5.1 本地 Dev 评测

```bash
cd /root/autodl-tmp/CFSP/newline
python ensemble_predict.py --test_file cfn-dev.json --output_prefix dev --seeds 42,123,456,789,1024
python score.py -g dataset/cfn-dev.json \
    -1 dataset/dev_task1_test_ensemble.json \
    -2 dataset/dev_task2_test_ensemble.json \
    -3 dataset/dev_task3_test_ensemble.json
```

### 5.2 天池提交

```bash
# A 榜
python ensemble_predict.py --test_file cfn-test-A.json --output_prefix A --seeds 42,123,456,789,1024

# B 榜
python ensemble_predict.py --test_file cfn-test-B.json --output_prefix B --seeds 42,123,456,789,1024
```

### 5.3 对比维度

对每个榜，对比 1-seed / 3-seed / 5-seed 的逐任务指标：

| 实验 | Seeds | 预期 A 榜 |
|------|:-----:|:--------:|
| E12 | 1 (42) | 70.03 |
| E21 | 3 (42/123/456) | 70.52 |
| **E22** | **5 (42/123/456/789/1024)** | **70.65~70.85** |

---

## 六、风险与应对

| 风险 | 概率 | 应对 |
|------|:--:|------|
| 5-seed 不涨反降 | 低 | Task 2 阈值调回 ≥2 或在 3/4/5 中搜索最优 |
| 新 seed 训练不稳定 | 低 | 已有 3 个 seed 成功先例，PyTorch 1.13 环境稳定 |
| 时间不足 | 中 | 截止 8/1 12:00，训练 ~7h，推理 ~30min，8/1 早上提交来得及 |
| 磁盘空间不够 | 低 | 每套 3 任务权重 ~1.8G，2 套额外 ~3.6G，当前磁盘充足 |

---

## 七、决策点

- **如果 5-seed Dev 比 3-seed 低 ≥0.3**：放弃 5-seed，3-seed 作为最终提交
- **如果 5-seed Dev 比 3-seed 低 0~0.3**：仍提交 A/B 榜对比，保留高分版本
- **如果 5-seed Dev 比 3-seed 高**：5-seed 作为最终提交
