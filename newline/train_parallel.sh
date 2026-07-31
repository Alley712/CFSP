#!/bin/bash
# ============================================================
# Multi-seed Parallel Training Launcher (3 GPUs)
# ============================================================
# Seed 42: 已有 E12 权重，本轮跳过
# Seed 123, 456: 需要训练
#
# 三任务训练期间完全独立（各自使用 gold label），可并行
# 预计墙钟时间: ~5-6h（单卡串行约 7h × 2 = 14h）
# ============================================================

# set -e disabled: don't exit if one seed fails, let the other finish

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SEEDS=(123 456)

# 每卡绑定各自的 seed
# GPU 0 → seed 123 (Task1 → Task2 → Task3)
# GPU 1 → seed 456 (Task1 → Task2 → Task3)
# GPU 2 → seed 123 Task3 提前跑 ? 不，三任务单seed 必须串行（共享 RoBERTa 权重加载兼容性）

echo "============================================"
echo "  Parallel Multi-seed Training (2 seeds × 3 tasks)"
echo "  GPU 0: seed 123 chain"
echo "  GPU 1: seed 456 chain"
echo "  GPU 2: reserve for later"
echo "  Start: $(date)"
echo "  Est. wall time: ~7h"
echo "============================================"

train_seed() {
    local seed=$1
    local gpu=$2
    export CUDA_VISIBLE_DEVICES=$gpu

    echo "[GPU $gpu] Seed $seed — Task 1 start: $(date)"
    cd "$SCRIPT_DIR"
    python train_task1.py --seed "$seed"
    echo "[GPU $gpu] Seed $seed — Task 1 done: $(date)"

    echo "[GPU $gpu] Seed $seed — Task 2 start: $(date)"
    python train_task2.py --seed "$seed"
    echo "[GPU $gpu] Seed $seed — Task 2 done: $(date)"

    echo "[GPU $gpu] Seed $seed — Task 3 start: $(date)"
    python train_task3.py --seed "$seed"
    echo "[GPU $gpu] Seed $seed — Task 3 done: $(date)"

    echo "[GPU $gpu] Seed $seed — ALL DONE: $(date)"
}

# 两个 seed 并行跑在不同的 GPU 上
train_seed 123 0 &
PID1=$!

train_seed 456 1 &
PID2=$!

echo "Waiting for both seed chains..."
wait $PID1 $PID2

echo ""
echo "============================================"
echo "  All training complete!"
echo "  End: $(date)"
echo "============================================"

ls -lh "$SCRIPT_DIR"/saves/model_task*_best_seed*.bin
