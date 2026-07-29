#!/usr/bin/env python3
"""
本地算分脚本 —— 对照 dev 集 gold 标注评测三个子任务的预测结果。

指标与比赛平台一致:
  Task1: Accuracy (框架名完全匹配)
  Task2: Token 级 P / R / F1 (span 覆盖的 token 重叠计算)
  Task3: 完全匹配 P / R / F1 (span + 角色标签都匹配才算对)
  总分: 0.3 * Task1_acc + 0.3 * Task2_F1 + 0.4 * Task3_F1

用法:
  # 三任务全部评测
  python score.py --gold data/cfn-dataset/cfn-dev.json \
                  --task1 newline/dataset/A_task1_test.json \
                  --task2 newline/dataset/A_task2_test.json \
                  --task3 newline/dataset/A_task3_test.json

  # 只评测部分任务
  python score.py --gold data/cfn-dataset/cfn-dev.json --task1 preds/task1.json
  python score.py --gold data/cfn-dataset/cfn-dev.json --task2 preds/task2.json
  python score.py --gold data/cfn-dataset/cfn-dev.json --task3 preds/task3.json
"""

import argparse
import json
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_gold(gold_path):
    """加载 dev/train gold 标注，返回 {sentence_id: gold_record}。

    注意：dev 集中部分 sentence_id 对应多个目标词（同一句子从不同 target
    激活不同框架）。预测格式（同比赛提交）假设一句一个 target，因此这里取
    最后一条作为 gold。训练时每个 (sentence_id, target) 是独立样本。
    """
    with open(gold_path, 'r', encoding='utf8') as f:
        data = json.load(f)

    seen = {}
    gold = {}
    for item in data:
        sid = item['sentence_id']
        if sid in seen:
            seen[sid] += 1
        else:
            seen[sid] = 1
        gold[sid] = {
            'text': item['text'],
            'frame': item['frame'],
            'target': item['target'],
            'cfn_spans': item.get('cfn_spans', []),
        }

    dup_count = sum(1 for v in seen.values() if v > 1)
    if dup_count > 0:
        print(f"  ⚠ {dup_count} 个 sentence_id 在 gold 中出现多次（多目标词），"
              f"取最后一条")

    return gold


def load_task1_pred(path):
    """加载 Task1 预测: [[sentence_id, "frame_name"], ...] → {sid: frame_name}"""
    with open(path, 'r', encoding='utf8') as f:
        data = json.load(f)
    return {item[0]: item[1] for item in data}


def load_task2_pred(path):
    """加载 Task2 预测: [[sentence_id, start, end], ...] → {sid: [(start, end), ...]}"""
    with open(path, 'r', encoding='utf8') as f:
        data = json.load(f)
    pred = defaultdict(list)
    for item in data:
        sid, start, end = item[0], item[1], item[2]
        pred[sid].append((start, end))
    return dict(pred)


def load_task3_pred(path):
    """加载 Task3 预测: [[sentence_id, start, end, "role"], ...] → {sid: [(start, end, role), ...]}"""
    with open(path, 'r', encoding='utf8') as f:
        data = json.load(f)
    pred = defaultdict(list)
    for item in data:
        sid, start, end, role = item[0], item[1], item[2], item[3]
        pred[sid].append((start, end, role))
    return dict(pred)


# ---------------------------------------------------------------------------
# Span 工具
# ---------------------------------------------------------------------------

def span_tokens(start, end):
    """闭区间 [start, end] → token 位置集合。"""
    return set(range(start, end + 1))


def token_overlap(spans_a, spans_b):
    """两组 span 的 token 级交集大小。"""
    tokens_a = set()
    tokens_b = set()
    for s, e in spans_a:
        tokens_a |= span_tokens(s, e)
    for s, e in spans_b:
        tokens_b |= span_tokens(s, e)
    return len(tokens_a & tokens_b)


# ---------------------------------------------------------------------------
# Task 1 评测
# ---------------------------------------------------------------------------

def evaluate_task1(gold, pred_frames):
    """Task 1: 框架识别准确率。"""
    correct = 0
    total = 0
    details = []  # 记录错误样本
    for sid, g in gold.items():
        if sid not in pred_frames:
            continue
        total += 1
        if pred_frames[sid] == g['frame']:
            correct += 1
        else:
            details.append({
                'sentence_id': sid,
                'text': g['text'][:40],
                'gold': g['frame'],
                'pred': pred_frames[sid],
            })

    acc = correct / total if total > 0 else 0.0
    return {
        'accuracy': acc,
        'correct': correct,
        'total': total,
        'errors': details,
    }


# ---------------------------------------------------------------------------
# Task 2 评测
# ---------------------------------------------------------------------------

def evaluate_task2(gold, pred_spans):
    """Task 2: 论元范围识别 — Token 级 P / R / F1。"""
    total_intersect = 0
    total_pred_tokens = 0
    total_gold_tokens = 0

    for sid, g in gold.items():
        gold_spans = [(s['start'], s['end']) for s in g['cfn_spans']]
        pred_s = pred_spans.get(sid, [])

        # Token 级统计
        gold_tokens = set()
        for s, e in gold_spans:
            gold_tokens |= span_tokens(s, e)
        pred_tokens = set()
        for s, e in pred_s:
            pred_tokens |= span_tokens(s, e)

        total_gold_tokens += len(gold_tokens)
        total_pred_tokens += len(pred_tokens)
        total_intersect += len(gold_tokens & pred_tokens)

    precision = total_intersect / total_pred_tokens if total_pred_tokens > 0 else 0.0
    recall = total_intersect / total_gold_tokens if total_gold_tokens > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'intersect_tokens': total_intersect,
        'pred_tokens': total_pred_tokens,
        'gold_tokens': total_gold_tokens,
    }


# ---------------------------------------------------------------------------
# Task 3 评测
# ---------------------------------------------------------------------------

def evaluate_task3(gold, pred_roles):
    """Task 3: 论元角色识别 — 完全匹配 P / R / F1。"""
    correct = 0
    total_pred = 0
    total_gold = 0

    for sid, g in gold.items():
        gold_set = set()
        for s in g['cfn_spans']:
            gold_set.add((s['start'], s['end'], s['fe_name']))
        pred_set = set(pred_roles.get(sid, []))

        total_gold += len(gold_set)
        total_pred += len(pred_set)
        correct += len(gold_set & pred_set)

    precision = correct / total_pred if total_pred > 0 else 0.0
    recall = correct / total_gold if total_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'correct': correct,
        'pred_total': total_pred,
        'gold_total': total_gold,
    }


# ---------------------------------------------------------------------------
# 格式化输出
# ---------------------------------------------------------------------------

def fmt_pct(v):
    return f"{v * 100:.2f}"


def print_results(task1_result, task2_result, task3_result):
    """打印结果表格。"""
    print()
    print("=" * 65)
    print("  CFSP 本地评测结果")
    print("=" * 65)

    # Task 1
    if task1_result:
        r = task1_result
        print(f"\n  Task 1 — 框架识别 (Accuracy)")
        print(f"    Acc     : {fmt_pct(r['accuracy'])}%  ({r['correct']}/{r['total']})")
        if r['errors']:
            print(f"    错误示例 (前 5 条):")
            for e in r['errors'][:5]:
                print(f"      [{e['sentence_id']}] \"{e['text']}...\"")
                print(f"        Gold: {e['gold']}  |  Pred: {e['pred']}")

    # Task 2
    if task2_result:
        r = task2_result
        print(f"\n  Task 2 — 论元范围识别 (Token 级)")
        print(f"    Precision: {fmt_pct(r['precision'])}%")
        print(f"    Recall   : {fmt_pct(r['recall'])}%")
        print(f"    F1       : {fmt_pct(r['f1'])}%")
        print(f"    (intersect={r['intersect_tokens']}, pred_tokens={r['pred_tokens']}, "
              f"gold_tokens={r['gold_tokens']})")

    # Task 3
    if task3_result:
        r = task3_result
        print(f"\n  Task 3 — 论元角色识别 (完全匹配)")
        print(f"    Precision: {fmt_pct(r['precision'])}%")
        print(f"    Recall   : {fmt_pct(r['recall'])}%")
        print(f"    F1       : {fmt_pct(r['f1'])}%")
        print(f"    (correct={r['correct']}, pred={r['pred_total']}, gold={r['gold_total']})")

    # 总分
    if task1_result and task2_result and task3_result:
        score = (0.3 * task1_result['accuracy']
                 + 0.3 * task2_result['f1']
                 + 0.4 * task3_result['f1']) * 100
        print(f"\n  {'─' * 45}")
        print(f"  总分 (weighted)")
        print(f"    = 0.3 × {fmt_pct(task1_result['accuracy'])} + "
              f"0.3 × {fmt_pct(task2_result['f1'])} + "
              f"0.4 × {fmt_pct(task3_result['f1'])}")
        print(f"    = {score:.2f}")
    elif task1_result and task2_result:
        score = (0.3 * task1_result['accuracy'] + 0.3 * task2_result['f1']) * 100
        print(f"\n  部分总分 (Task1+2): {score:.2f}")

    print()
    print("=" * 65)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='CFSP 本地算分 —— 对照 dev gold 评测预测结果',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python score.py -g data/cfn-dataset/cfn-dev.json \\
                  -1 preds/A_task1_test.json \\
                  -2 preds/A_task2_test.json \\
                  -3 preds/A_task3_test.json

  python score.py -g data/cfn-dataset/cfn-dev.json -3 preds/A_task3_test.json
        """,
    )
    parser.add_argument('-g', '--gold', required=True,
                        help='Gold 标注文件路径 (cfn-dev.json 或 cfn-train.json)')
    parser.add_argument('-1', '--task1',
                        help='Task1 预测文件 (格式: [[sid, "frame"], ...])')
    parser.add_argument('-2', '--task2',
                        help='Task2 预测文件 (格式: [[sid, start, end], ...])')
    parser.add_argument('-3', '--task3',
                        help='Task3 预测文件 (格式: [[sid, start, end, "role"], ...])')

    args = parser.parse_args()

    if not any([args.task1, args.task2, args.task3]):
        parser.print_help()
        print("\n错误: 请至少指定 --task1 / --task2 / --task3 中的一个")
        sys.exit(1)

    # 加载 gold
    print(f"加载 gold 标注: {args.gold}")
    gold = load_gold(args.gold)
    print(f"  {len(gold)} 条标注样本")

    task1_result = None
    task2_result = None
    task3_result = None

    # Task 1
    if args.task1:
        print(f"\n加载 Task1 预测: {args.task1}")
        pred_frames = load_task1_pred(args.task1)
        print(f"  {len(pred_frames)} 条预测")
        task1_result = evaluate_task1(gold, pred_frames)

    # Task 2
    if args.task2:
        print(f"\n加载 Task2 预测: {args.task2}")
        pred_spans = load_task2_pred(args.task2)
        print(f"  {len(pred_spans)} 条样本, {sum(len(v) for v in pred_spans.values())} 个 span")
        task2_result = evaluate_task2(gold, pred_spans)

    # Task 3
    if args.task3:
        print(f"\n加载 Task3 预测: {args.task3}")
        pred_roles = load_task3_pred(args.task3)
        print(f"  {len(pred_roles)} 条样本, {sum(len(v) for v in pred_roles.values())} 个论元角色")
        task3_result = evaluate_task3(gold, pred_roles)

    print_results(task1_result, task2_result, task3_result)


if __name__ == '__main__':
    main()
