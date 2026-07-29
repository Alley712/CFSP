#!/bin/bash
# 依次训练 Task 1 → Task 2 → Task 3（无 AEDA）
# 用法: bash train_all.sh

set -e
cd /root/autodl-tmp/CFSP/newline

echo "========================================"
echo "  Task 1: 框架识别 (10 epochs)"
echo "========================================"
python train_task1.py
echo "[$(date '+%H:%M:%S')] Task 1 done."

echo ""
echo "========================================"
echo "  Task 2: 论元范围识别 (5 epochs)"
echo "========================================"
python train_task2.py
echo "[$(date '+%H:%M:%S')] Task 2 done."

echo ""
echo "========================================"
echo "  Task 3: 论元角色识别 (10 epochs)"
echo "========================================"
python train_task3.py
echo "[$(date '+%H:%M:%S')] Task 3 done."

echo ""
echo "========================================"
echo "  训练全部完成"
echo "========================================"
