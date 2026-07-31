#!/usr/bin/python3
"""
错误分析脚本 — 生成 error_cases.md

Task 1: 框架混淆对 (confusion pairs)
Task 2: 边界错误 (FN / FP / 边界偏差)
Task 3: 角色分类错误 — Oracle 模式 (gold span + gold frame 输入)
"""

import codecs
import json
import os
import sys
from collections import Counter, defaultdict
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from model_task1 import Model as Model1
from model_task2 import Model as Model2
from model_task3 import Model as Model3

# ============================================================
# 配置
# ============================================================

DEV_FILE = "./dataset/cfn-dev.json"
FRAME_INFO = "./dataset/frame_info.json"
MODEL_SEED = 42  # 用 E12 baseline (seed 42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_FILE = "error_cases.md"

# ============================================================
# 工具函数
# ============================================================

def load_json(path):
    with codecs.open(path, 'r', encoding='utf8') as f:
        return json.load(f)


def build_frame_role_map(frame_info):
    """frame_name → list of legal FE names."""
    m = {}
    for f in frame_info:
        m[f["frame_name"]] = [fe["fe_name"] for fe in f["fes"]]
    return m


def pad(d, max_len, v=0):
    return d + [v] * (max_len - len(d))


def tokenize_and_insert(text, target):
    """Tokenize text and insert [1]/[2] around target. Returns (input_ids, attention_mask)."""
    data = Tokenizer.encode_plus(list(text))
    input_ids = data.data['input_ids']
    attention_mask = data.data['attention_mask']
    tgt = [target[-1]["start"] + 1, target[-1]["end"] + 1]
    input_ids = input_ids[0:tgt[0]] + [1] + input_ids[tgt[0]:tgt[1] + 1] + [2] + input_ids[tgt[1] + 1:]
    attention_mask = attention_mask + [1, 1]
    return input_ids, attention_mask, tgt


Tokenizer = None  # set in main()


# ============================================================
# Task 1 — 框架混淆分析
# ============================================================

def analyze_task1(model, dev_data, idx2label):
    """返回: confusion_pairs [(gold, pred, count)], per_sample_errors [{...}]"""
    model.eval()
    all_preds = []
    all_golds = []

    # Build simple dataset
    with torch.no_grad():
        for item in tqdm(dev_data, desc="Task 1 predict"):
            text = item["text"]
            target_orig = item["target"]
            tgt_token = [target_orig[-1]["start"] + 1, target_orig[-1]["end"] + 1]

            data = Tokenizer.encode_plus(list(text))
            input_ids = torch.tensor([data.data['input_ids']], dtype=torch.long).to(DEVICE)
            attention_mask = torch.tensor([data.data['attention_mask']], dtype=torch.long).to(DEVICE)

            output = model(input_ids=input_ids, attention_mask=attention_mask,
                          target=[tgt_token], labels=None, device=DEVICE, for_test=True)
            pred_idx = torch.argmax(F.softmax(output["logits"], dim=-1), dim=-1).item()
            all_preds.append(idx2label[pred_idx])
            all_golds.append(item["frame"])

    # Confusion pairs
    confusion = Counter()
    errors = []
    for i, (gold, pred) in enumerate(zip(all_golds, all_preds)):
        if gold != pred:
            confusion[(gold, pred)] += 1
            errors.append({
                "idx": i,
                "text": dev_data[i]["text"],
                "target": dev_data[i]["target"],
                "gold_frame": gold,
                "pred_frame": pred,
            })
    top_pairs = confusion.most_common(10)
    return top_pairs, errors, all_preds, all_golds


# ============================================================
# Task 2 — 边界错误分析
# ============================================================

def analyze_task2(model, dev_data):
    """返回: error_cases {FN: [...], FP: [...], BOUNDARY: [...]}"""
    model.eval()
    fn_cases, fp_cases, boundary_cases = [], [], []

    with torch.no_grad():
        for item in tqdm(dev_data, desc="Task 2 predict"):
            text = item["text"]
            target_orig = item["target"]
            input_ids, attention_mask, tgt = tokenize_and_insert(text, target_orig)

            gold_spans = [(s["start"], s["end"]) for s in item["cfn_spans"]]
            gold_token_sets = [set(range(s, e + 1)) for s, e in gold_spans]

            # Model forward
            ids_t = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
            mask_t = torch.tensor([attention_mask], dtype=torch.long).to(DEVICE)

            output = model(input_ids=ids_t, attention_mask=mask_t, target=[tgt],
                          labels=None, device=DEVICE, for_test=True)
            logits = output["logits"][0]  # (seq_len, seq_len)

            H_attention_mask = torch.triu(torch.matmul(
                mask_t.unsqueeze(2).float(), mask_t.unsqueeze(1).float()), diagonal=0)[0]
            H_pred = torch.where(logits >= 0, torch.ones_like(logits),
                                torch.zeros_like(logits)) * H_attention_mask

            pred_spans = []
            for idx in torch.nonzero(H_pred):
                r, c = idx[0].item(), idx[1].item()
                if c < tgt[0]:
                    pred_spans.append((r - 1, c - 1))
                elif r > tgt[1]:
                    pred_spans.append((r - 3, c - 3))

            pred_token_sets = [set(range(s, e + 1)) for s, e in pred_spans]

            # Classify each gold span
            for gs, ge in gold_spans:
                g_tokens = set(range(gs, ge + 1))
                best_overlap = 0
                best_ps = None
                for ps, pe in pred_spans:
                    p_tokens = set(range(ps, pe + 1))
                    overlap = len(g_tokens & p_tokens)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_ps = (ps, pe)

                if best_overlap == 0:
                    # FN: gold span not covered
                    fn_cases.append({
                        "text": text,
                        "target": target_orig,
                        "gold_span": (gs, ge),
                        "gold_text": text[gs:ge + 1],
                    })
                elif g_tokens != set(range(best_ps[0], best_ps[1] + 1)):
                    # Boundary deviation
                    boundary_cases.append({
                        "text": text,
                        "target": target_orig,
                        "gold_span": (gs, ge),
                        "gold_text": text[gs:ge + 1],
                        "pred_span": best_ps,
                        "pred_text": text[best_ps[0]:best_ps[1] + 1],
                    })

            # FP: pred spans with no gold overlap
            for ps, pe in pred_spans:
                p_tokens = set(range(ps, pe + 1))
                has_overlap = False
                for gs, ge in gold_spans:
                    g_tokens = set(range(gs, ge + 1))
                    if p_tokens & g_tokens:
                        has_overlap = True
                        break
                if not has_overlap:
                    fp_cases.append({
                        "text": text,
                        "target": target_orig,
                        "pred_span": (ps, pe),
                        "pred_text": text[ps:pe + 1] if ps < len(text) and pe < len(text) else "(OOB)",
                    })

    return {"FN": fn_cases, "FP": fp_cases, "BOUNDARY": boundary_cases}


# ============================================================
# Task 3 — Oracle 模式角色分类错误
# ============================================================

def analyze_task3_oracle(model, dev_data, frame_role_map, idx2label, frame2idx):
    """Oracle 模式: gold span + gold frame 输入，纯粹分析角色分类错误。

    返回: {legal_wrong: [...], illegal: [...]}
    """
    model.eval()
    legal_wrong_cases = []
    illegal_cases = []

    with torch.no_grad():
        for item in tqdm(dev_data, desc="Task 3 oracle"):
            text = item["text"]
            target_orig = item["target"]
            gold_frame = item["frame"]
            gold_frame_id = frame2idx[gold_frame]
            legal_roles = frame_role_map.get(gold_frame, [])
            tgt = [target_orig[-1]["start"] + 1, target_orig[-1]["end"] + 1]

            for span in item["cfn_spans"]:
                gold_role = span["fe_name"]
                s, e = span["start"], span["end"]

                # Build input with gold span
                input_ids, attention_mask, _ = tokenize_and_insert(text, target_orig)

                # Compute label_idx (same logic as dataset)
                if e + 1 < tgt[0]:
                    label_idx = [s + 1, e + 1]
                elif s + 1 > tgt[1]:
                    label_idx = [s + 3, e + 3]
                else:
                    continue  # span overlaps target, shouldn't happen

                ids_t = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
                mask_t = torch.tensor([attention_mask], dtype=torch.long).to(DEVICE)
                frame_t = torch.tensor([gold_frame_id], dtype=torch.long).to(DEVICE)

                output = model(input_ids=ids_t, attention_mask=mask_t,
                              target=[label_idx], labels=None, device=DEVICE,
                              for_test=True, frame_ids=frame_t)
                logits = output["logits"][0]
                pred_idx = torch.argmax(F.softmax(logits, dim=-1), dim=-1).item()
                pred_role = idx2label[pred_idx]

                if pred_role == gold_role:
                    continue  # correct

                case = {
                    "text": text,
                    "target": target_orig,
                    "span": (s, e),
                    "span_text": text[s:e + 1],
                    "gold_frame": gold_frame,
                    "gold_role": gold_role,
                    "pred_role": pred_role,
                    "legal_roles": legal_roles[:5],
                    "legal_count": len(legal_roles),
                }

                if pred_role in legal_roles:
                    legal_wrong_cases.append(case)
                else:
                    illegal_cases.append(case)

    return {"legal_wrong": legal_wrong_cases, "illegal": illegal_cases}


# ============================================================
# Markdown 输出
# ============================================================

def format_target(target, text):
    """Format target as '词 [start,end]', extracting word from original text."""
    t = target[-1]
    target_text = text[t['start']:t['end'] + 1]
    return f"{target_text} [{t['start']},{t['end']}]"


def write_markdown(task1_data, task2_data, task3_data):
    """生成 error_cases.md"""
    lines = []
    lines.append("# 错误案例分析 (E12 baseline, seed 42)")
    lines.append("")
    lines.append("> 模型: chinese-roberta-wwm-ext + FrameEmbedding")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- Task 1 ----
    lines.append("## Task 1 — 框架混淆")
    lines.append("")
    top_pairs, t1_errors, _, _ = task1_data
    frame_errors_total = len(t1_errors)

    lines.append(f"共 {frame_errors_total} 个框架预测错误 / 2300 条 dev 数据。")
    lines.append("")
    lines.append("### Top-10 混淆对")
    lines.append("")
    lines.append("| # | Gold 框架 | Pred 框架 | 次数 |")
    lines.append("|---|-----------|-----------|:--:|")
    for i, ((gold, pred), count) in enumerate(top_pairs, 1):
        lines.append(f"| {i} | {gold} | {pred} | {count} |")
    lines.append("")

    # Pick 1 example per confusion pair
    lines.append("### 典型案例")
    lines.append("")
    seen_pairs = set()
    case_num = 0
    for err in t1_errors:
        pair = (err["gold_frame"], err["pred_frame"])
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        case_num += 1
        if case_num > 10:
            break

        lines.append(f"#### 案例 {case_num} — 框架混淆: {err['gold_frame']} → {err['pred_frame']}")
        lines.append("")
        lines.append("| 项目 | 内容 |")
        lines.append("|------|------|")
        lines.append(f"| 句子 | {err['text'][:80]}{'...' if len(err['text'])>80 else ''} |")
        lines.append(f"| 目标词 | {format_target(err['target'], err['text'])} |")
        lines.append(f"| Gold | {err['gold_frame']} |")
        lines.append(f"| Pred | {err['pred_frame']} |")
        lines.append(f"| 分析 | 两个框架语义相近，目标词在上下文中可被两种解读覆盖 |")
        lines.append("")

    # ---- Task 2 ----
    lines.append("---")
    lines.append("")
    lines.append("## Task 2 — 边界错误")
    lines.append("")

    for cat, label in [("FN", "漏检 (False Negative)"), ("FP", "虚检 (False Positive)"), ("BOUNDARY", "边界偏差")]:
        cases = task2_data[cat]
        lines.append(f"### {label}")
        lines.append(f"共 {len(cases)} 例。")
        lines.append("")
        if not cases:
            lines.append("_(无案例)_")
            lines.append("")
            continue

        for j, c in enumerate(cases[:2], 1):
            lines.append(f"#### {label} 案例 {j}")
            lines.append("")
            lines.append("| 项目 | 内容 |")
            lines.append("|------|------|")
            lines.append(f"| 句子 | {c['text'][:80]}{'...' if len(c['text'])>80 else ''} |")
            lines.append(f"| 目标词 | {format_target(c['target'], c['text'])} |")
            if cat == "FN":
                lines.append(f"| Gold Span | [{c['gold_span'][0]},{c['gold_span'][1]}] 「{c['gold_text']}」 |")
                lines.append(f"| 问题 | 模型完全漏掉了这个论元 |")
            elif cat == "FP":
                lines.append(f"| Pred Span | [{c['pred_span'][0]},{c['pred_span'][1]}] 「{c['pred_text']}」 |")
                lines.append(f"| 问题 | 模型预测了一个不存在的论元 |")
            else:
                lines.append(f"| Gold Span | [{c['gold_span'][0]},{c['gold_span'][1]}] 「{c['gold_text']}」 |")
                lines.append(f"| Pred Span | [{c['pred_span'][0]},{c['pred_span'][1]}] 「{c['pred_text']}」 |")
                lines.append(f"| 问题 | 边界不完全匹配，可能是边界词归入/排除不一致 |")
            lines.append("")

    # ---- Task 3 ----
    lines.append("---")
    lines.append("")
    lines.append("## Task 3 — 角色分类错误 (Oracle 模式)")
    lines.append("")
    lines.append("> ⚠️ 使用 **gold span + gold frame** 作为输入，排除 Task1/2 误差传播，纯粹分析 Task3 的角色分类能力。")
    lines.append("")

    legal = task3_data["legal_wrong"]
    illegal = task3_data["illegal"]
    total_t3 = len(legal) + len(illegal)

    # Count gold spans total for accuracy
    lines.append(f"共 {total_t3} 个角色分类错误。")
    lines.append(f"其中 **合法但选错**: {len(legal)} 例，**非法角色**: {len(illegal)} 例。")
    lines.append("")

    # Legal but wrong
    lines.append("### 合法但选错 (Legal Role, Wrong Choice)")
    lines.append("")
    lines.append("pred role 在 gold frame 的合法 FE 列表中，但选错了具体角色。")
    lines.append("")

    # Group by gold frame for diversity
    seen_frames = set()
    for j, c in enumerate(legal):
        if c["gold_frame"] in seen_frames:
            continue
        seen_frames.add(c["gold_frame"])
        if len(seen_frames) > 3:
            break

        lines.append(f"#### 案例 — {c['gold_frame']}")
        lines.append("")
        lines.append("| 项目 | 内容 |")
        lines.append("|------|------|")
        lines.append(f"| 句子 | {c['text'][:80]}{'...' if len(c['text'])>80 else ''} |")
        lines.append(f"| 目标词 | {format_target(c['target'], c['text'])} |")
        lines.append(f"| Span | [{c['span'][0]},{c['span'][1]}] 「{c['span_text']}」 |")
        lines.append(f"| Gold Frame | {c['gold_frame']} |")
        lines.append(f"| Gold Role | {c['gold_role']} |")
        lines.append(f"| Pred Role | {c['pred_role']} |")
        lines.append(f"| 合法 FE (前5) | {', '.join(c['legal_roles'])} |")
        lines.append(f"| 分析 | pred 和 gold 都是该框架下的合法角色，语义相近导致混淆 |")
        lines.append("")

    # Illegal roles
    lines.append("### 非法角色 (Illegal Role)")
    lines.append("")
    lines.append("pred role 不在 gold frame 的合法 FE 列表中，模型输出了一个不应该出现的角色。")
    lines.append("")

    seen_frames2 = set()
    shown = 0
    for c in illegal:
        if c["gold_frame"] in seen_frames2:
            continue
        seen_frames2.add(c["gold_frame"])
        shown += 1
        if shown > 3:
            break

        lines.append(f"#### 案例 — {c['gold_frame']}")
        lines.append("")
        lines.append("| 项目 | 内容 |")
        lines.append("|------|------|")
        lines.append(f"| 句子 | {c['text'][:80]}{'...' if len(c['text'])>80 else ''} |")
        lines.append(f"| 目标词 | {format_target(c['target'], c['text'])} |")
        lines.append(f"| Span | [{c['span'][0]},{c['span'][1]}] 「{c['span_text']}」 |")
        lines.append(f"| Gold Frame | {c['gold_frame']} |")
        lines.append(f"| Gold Role | {c['gold_role']} |")
        lines.append(f"| Pred Role | {c['pred_role']} ⚠ 非法 |")
        lines.append(f"| 合法 FE (前5) | {', '.join(c['legal_roles'])} |")
        lines.append(f"| 分析 | pred 角色不属于当前框架——说明 FrameEmbedding 的偏置强度不足以完全抑制跨框架干扰 |")
        lines.append("")

    # ---- Overall stats ----
    lines.append("---")
    lines.append("")
    lines.append("## 错误分布总结")
    lines.append("")

    # Task3 oracle accuracy
    # Need total gold spans from dev
    dev = load_json(DEV_FILE)
    total_gold_spans = sum(len(item["cfn_spans"]) for item in dev)
    t3_correct = total_gold_spans - total_t3
    t3_acc = t3_correct / total_gold_spans * 100 if total_gold_spans > 0 else 0

    lines.append(f"| 任务 | 指标 | 值 |")
    lines.append(f"|------|------|:--:|")
    lines.append(f"| Task 1 | 框架预测错误数 (dev) | {frame_errors_total} / 2300 |")
    lines.append(f"| Task 2 | FN / FP / 边界偏差 | {len(task2_data['FN'])} / {len(task2_data['FP'])} / {len(task2_data['BOUNDARY'])} |")
    lines.append(f"| Task 3 (oracle) | 角色分类准确率 | {t3_acc:.1f}% ({t3_correct}/{total_gold_spans}) |")
    lines.append(f"| Task 3 (oracle) | 合法选错 / 非法角色 | {len(legal)} / {len(illegal)} |")

    with open(OUTPUT_FILE, 'w', encoding='utf8') as f:
        f.write('\n'.join(lines))
    print(f"\nSaved: {OUTPUT_FILE}")


# ============================================================
# 主入口
# ============================================================

def main():
    global Tokenizer

    print(f"Device: {DEVICE}")
    print(f"Model seed: {MODEL_SEED}")

    # Load data
    print("Loading data...")
    dev_data = load_json(DEV_FILE)
    frame_info = load_json(FRAME_INFO)
    frame_role_map = build_frame_role_map(frame_info)
    print(f"  Dev samples: {len(dev_data)}")
    print(f"  Frames: {len(frame_info)}")

    # Tokenizer
    Tokenizer = BertTokenizer(vocab_file='./chinese_roberta_wwm_ext/vocab.txt', do_lower_case=True)

    config = BertConfig.from_json_file('./chinese_roberta_wwm_ext/config.json')

    # ---- Task 1 ----
    print("\n=== Task 1: Frame Confusion ===")
    config1 = BertConfig.from_json_file('./chinese_roberta_wwm_ext/config.json')
    # Build idx2label
    idx2label_t1 = []
    for f in frame_info:
        idx2label_t1.append(f["frame_name"])
    config1.num_labels = len(idx2label_t1)
    model1 = Model1(config1)
    ckpt1 = f"saves/model_task1_best_seed{MODEL_SEED}.bin"
    model1.load_state_dict(torch.load(ckpt1, map_location='cpu'), strict=False)
    model1.to(DEVICE)
    model1.eval()
    t1_data = analyze_task1(model1, dev_data, idx2label_t1)

    # ---- Task 2 ----
    print("\n=== Task 2: Boundary Errors ===")
    max_cls = len(frame_role_map)
    config2 = BertConfig.from_json_file('./chinese_roberta_wwm_ext/config.json')
    config2.num_labels = 1
    config2.max_cls = max_cls
    model2 = Model2(config2)
    ckpt2 = f"saves/model_task2_best_seed{MODEL_SEED}.bin"
    model2.load_state_dict(torch.load(ckpt2, map_location='cpu'), strict=False)
    model2.to(DEVICE)
    model2.eval()
    t2_data = analyze_task2(model2, dev_data)

    # ---- Task 3 Oracle ----
    print("\n=== Task 3: Oracle Role Classification ===")
    idx2label_t3 = []
    for f in frame_info:
        for fe in f["fes"]:
            if fe["fe_name"] not in idx2label_t3:
                idx2label_t3.append(fe["fe_name"])
    frame2idx = {}
    for i, f in enumerate(frame_info):
        frame2idx[f["frame_name"]] = i
    config3 = BertConfig.from_json_file('./chinese_roberta_wwm_ext/config.json')
    config3.num_labels = len(idx2label_t3)
    config3.num_frames = len(frame2idx)
    model3 = Model3(config3)
    ckpt3 = f"saves/model_task3_best_seed{MODEL_SEED}.bin"
    model3.load_state_dict(torch.load(ckpt3, map_location='cpu'), strict=False)
    model3.to(DEVICE)
    model3.eval()
    t3_data = analyze_task3_oracle(model3, dev_data, frame_role_map, idx2label_t3, frame2idx)

    # ---- Output ----
    print("\n=== Generating error_cases.md ===")
    write_markdown(t1_data, t2_data, t3_data)
    print("Done.")


if __name__ == '__main__':
    main()
