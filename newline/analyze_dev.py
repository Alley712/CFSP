"""
Phase 3.4 dev 集分析: Task1 top-K 覆盖率和合法角色命中率。

输出:
  - top-1/3/5 框架准确率
  - 对应合法角色命中率 (gold role 是否在 top-K 框架的合法 FE 并集中)
"""

import json
import torch
import torch.nn.functional as F
import numpy as np
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer
import sys
sys.path.insert(0, '.')
from params import args
from model_task1 import Model as Task1Model
from frame_roles import load_frame_info, build_frame2roles


def load_dev_data(path):
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)


def build_frame_labels(frame_info_path):
    """返回 idx2label 和 label2idx"""
    frame_info = load_frame_info(frame_info_path)
    idx2label = [item['frame_name'] for item in frame_info]
    label2idx = {v: i for i, v in enumerate(idx2label)}
    return idx2label, label2idx


def build_role_labels(frame_info_path):
    """构建全局 role2idx (与 dataset_task3.py 一致)"""
    frame_info = load_frame_info(frame_info_path)
    idx2label = []
    for item in frame_info:
        for fe in item['fes']:
            if fe['fe_name'] not in idx2label:
                idx2label.append(fe['fe_name'])
    label2idx = {v: i for i, v in enumerate(idx2label)}
    return idx2label, label2idx


class Task1DevDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        d1 = self.data[item]
        encoded = self.tokenizer.encode_plus(list(d1['text']))
        target = [d1['target'][-1]['start'] + 1, d1['target'][-1]['end'] + 1]
        return (encoded.data['input_ids'], encoded.data['attention_mask'],
                target, d1['sentence_id'], d1.get('frame', ''))


def collate_fn(data, device):
    def pad(d, max_len, v=0):
        return d + [v] * (max_len - len(d))
    bs = len(data)
    max_len = max(len(x[0]) for x in data)
    input_ids = torch.from_numpy(np.array([pad(d[0], max_len, 0) for d in data], dtype=np.int64)).to(device)
    attention_mask = torch.from_numpy(np.array([pad(d[1], max_len, 0) for d in data], dtype=np.int64)).to(device)
    target = [d[2] for d in data]
    sentence_id = [d[3] for d in data]
    gold_frame = [d[4] for d in data]
    return input_ids, attention_mask, target, sentence_id, gold_frame


def main():
    device = torch.device('cuda')
    K = 5

    # 加载数据
    dev_data = load_dev_data('./dataset/cfn-dev.json')
    print(f"dev 集: {len(dev_data)} 条样本")

    idx2label, label2idx = build_frame_labels('./dataset/frame_info.json')
    print(f"框架数: {len(idx2label)}")

    _, role2idx = build_role_labels('./dataset/frame_info.json')
    print(f"全局角色数: {len(role2idx)}")

    frame2roles, frame2idx, idx2frame = build_frame2roles(
        './dataset/frame_info.json', role2idx)
    print(f"frame2roles: {len(frame2roles)} 个框架")

    # 加载 Task1 模型
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    ds = Task1DevDataset(dev_data, tokenizer)
    config = BertConfig.from_json_file(args.config_file)
    config.num_labels = len(idx2label)
    model = Task1Model(config)
    state = torch.load('saves/model_task1_best.bin', map_location='cpu')
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=partial(collate_fn, device=device))

    # 存储结果
    sample_results = []  # [{sid, gold_frame, topk_frames, topk_probs, gold_roles}]

    with torch.no_grad():
        for batch in tqdm(loader, desc='Task1 dev analysis'):
            input_ids, attention_mask, target, sentence_ids, gold_frames = batch
            logits = model(input_ids=input_ids, attention_mask=attention_mask,
                           target=target, labels=None, device=device, for_test=True)['logits']
            probs = F.softmax(logits, dim=-1)  # (batch, num_frames)
            topk_probs, topk_indices = torch.topk(probs, k=K, dim=-1)

            for i in range(len(sentence_ids)):
                topk_frames = [idx2label[idx.item()] for idx in topk_indices[i]]
                topk_p = topk_probs[i].tolist()
                sample_results.append({
                    'sentence_id': sentence_ids[i],
                    'gold_frame': gold_frames[i],
                    'topk_frames': topk_frames,
                    'topk_probs': topk_p,
                })

    # --- 统计 ---
    total = len(sample_results)
    topk_correct = {k: 0 for k in [1, 3, 5]}

    for r in sample_results:
        gold = r['gold_frame']
        for k in [1, 3, 5]:
            if gold in r['topk_frames'][:k]:
                topk_correct[k] += 1

    print(f"\n{'='*50}")
    print(f"Task 1 Top-K 框架覆盖率 (dev 集, N={total})")
    print(f"{'='*50}")
    for k in [1, 3, 5]:
        acc = topk_correct[k] / total * 100
        print(f"  Top-{k}: {topk_correct[k]}/{total} = {acc:.2f}%")

    # --- 合法角色命中率 ---
    # 构造 sid → gold roles 映射（需要从 cfn-dev.json 的 cfn_spans 中提取）
    # 注意：dev 集中相同 sid 可能多目标词，这里用全部条目
    sid2gold_roles = {}
    for item in dev_data:
        sid = item['sentence_id']
        if sid not in sid2gold_roles:
            sid2gold_roles[sid] = set()
        for span in item.get('cfn_spans', []):
            fe_name = span['fe_name']
            if fe_name in role2idx:
                sid2gold_roles[sid].add(role2idx[fe_name])

    # 统计
    role_hit = {k: 0 for k in [1, 3, 5]}
    role_total = 0

    for r in sample_results:
        sid = r['sentence_id']
        gold_roles = sid2gold_roles.get(sid, set())
        if not gold_roles:
            continue
        role_total += len(gold_roles)

        for k in [1, 3, 5]:
            legal_union = set()
            for fname in r['topk_frames'][:k]:
                fid = frame2idx.get(fname)
                if fid is not None:
                    legal_union |= frame2roles.get(fid, set())
            for rid in gold_roles:
                if rid in legal_union:
                    role_hit[k] += 1

    print(f"\n{'='*50}")
    print(f"合法角色命中率 (gold role 在 top-K 框架合法 FE 并集中)")
    print(f"  gold role 总数: {role_total}")
    print(f"{'='*50}")
    for k in [1, 3, 5]:
        hit_rate = role_hit[k] / role_total * 100 if role_total > 0 else 0
        print(f"  Top-{k}: {role_hit[k]}/{role_total} = {hit_rate:.2f}%")

    # 结论
    print(f"\n{'='*50}")
    print("结论:")

    top1_acc = topk_correct[1] / total * 100
    top3_acc = topk_correct[3] / total * 100
    top3_gain = top3_acc - top1_acc

    role_top1 = role_hit[1] / role_total * 100 if role_total > 0 else 0
    role_top3 = role_hit[3] / role_total * 100 if role_total > 0 else 0
    role_loss = role_top1 - role_top3 if role_top3 > 0 else 0

    print(f"  Top-1 → Top-3 框架覆盖率提升: {top3_gain:.1f} 个百分点")
    print(f"  Top-1 → Top-3 合法角色命中:   {role_top1:.1f}% → {role_top3:.1f}%")
    print(f"  Top-1 硬约束 = {100 - role_top1:.1f}% 的 gold role 会被误屏蔽")
    print(f"  Top-3 软约束 = {100 - role_top3:.1f}% 的 gold role 会被误屏蔽")

    if top3_gain > 10:
        print(f"\n  ✅ E12b (Top-K 边际化) 有显著空间")
    elif top3_gain > 5:
        print(f"\n  👍 E12b 值得尝试")
    else:
        print(f"\n  ⚠ Top-K 增益有限，E12a (top-1 硬约束) 风险较高")


if __name__ == '__main__':
    main()
