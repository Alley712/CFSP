#!/usr/bin/python3
"""
Multi-seed Ensemble Prediction (方向 3)
========================================
- Task 1: soft voting (average logits across seeds → argmax)
- Task 2: span majority voting (keep spans predicted by ≥2/3 seeds)
- Task 3: soft voting (average logits across seeds → argmax)

Usage:
    # Ensemble prediction on B test set (default)
    python ensemble_predict.py

    # With custom seeds
    python ensemble_predict.py --seeds 42,123,456

    # With custom test file
    python ensemble_predict.py --test_file cfn-test-B.json  # or cfn-test-A.json
"""

import os
import json
import codecs
import argparse
from functools import partial
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from model_task1 import Model as Model1
from model_task2 import Model as Model2
from model_task3 import Model as Model3


# ============================================================
# Dataset classes (inline, matching predict_task*.py)
# ============================================================

class DatasetTask1(torch.utils.data.Dataset):
    """Test dataset for Task 1 — Frame Identification."""

    def __init__(self, json_file, label_file, tokenizer):
        self.tokenizer = tokenizer
        with codecs.open(json_file, 'r', encoding='utf8') as f:
            self.all_data = json.load(f)
        with codecs.open(label_file, 'r', encoding='utf8') as f:
            self.ori_labels = json.load(f)
        self.idx2label = []
        self.label2idx = {}
        for i, line in enumerate(self.ori_labels):
            self.idx2label.append(line["frame_name"])
            self.label2idx[line["frame_name"]] = i
        self.num_labels = len(self.idx2label)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        d1 = self.all_data[item]
        data = self.tokenizer.encode_plus(list(d1['text']))
        input_ids = data.data['input_ids']
        attention_mask = data.data['attention_mask']
        target = [d1["target"][-1]["start"] + 1, d1["target"][-1]["end"] + 1]
        sentence_id = d1["sentence_id"]
        return input_ids, attention_mask, target, sentence_id


class DatasetTask2(torch.utils.data.Dataset):
    """Test dataset for Task 2 — Argument Span Detection."""

    def __init__(self, json_file, label_file, tokenizer):
        self.tokenizer = tokenizer
        with codecs.open(json_file, 'r', encoding='utf8') as f:
            self.all_data = json.load(f)
        with codecs.open(label_file, 'r', encoding='utf8') as f:
            self.ori_labels = json.load(f)
        self.label2idx = {}
        self.label2cls = {}
        self.num_labels = 0
        for k, line in enumerate(self.ori_labels):
            frame_name = line["frame_name"]
            self.label2cls[frame_name] = k
            if frame_name not in self.label2idx:
                self.label2idx[frame_name] = {}
            for i, fes in enumerate(line["fes"]):
                self.label2idx[frame_name][fes["fe_name"]] = i
            if self.num_labels < len(line["fes"]):
                self.num_labels = len(line["fes"])
        self.max_cls = len(self.label2idx)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        d1 = self.all_data[item]
        data = self.tokenizer.encode_plus(list(d1['text']))
        input_ids = data.data['input_ids']
        attention_mask = data.data['attention_mask']
        target = [d1["target"][-1]["start"] + 1, d1["target"][-1]["end"] + 1]
        input_ids = input_ids[0: target[0]] + [1] + input_ids[target[0]: target[1] + 1] + [2] + input_ids[target[1] + 1:]
        attention_mask = attention_mask + [1, 1]
        sentence_id = d1["sentence_id"]
        return input_ids, attention_mask, target, sentence_id


class DatasetTask3(torch.utils.data.Dataset):
    """Test dataset for Task 3 — Role Classification.
    Reads Task 1 and Task 2 ensemble outputs as input."""

    def __init__(self, json_file, label_file, task1_file, task2_file, tokenizer):
        self.tokenizer = tokenizer
        with codecs.open(json_file, 'r', encoding='utf8') as f:
            self.all_data = json.load(f)
        with codecs.open(label_file, 'r', encoding='utf8') as f:
            self.ori_labels = json.load(f)
        with codecs.open(task1_file, 'r', encoding='utf8') as f:
            task1_data = json.load(f)
        with codecs.open(task2_file, 'r', encoding='utf8') as f:
            self.task2_data = json.load(f)

        self.idx2label = []
        for line in self.ori_labels:
            for fes in line["fes"]:
                if fes["fe_name"] not in self.idx2label:
                    self.idx2label.append(fes["fe_name"])
        self.label2idx = {}
        for i in range(len(self.idx2label)):
            self.label2idx[self.idx2label[i]] = i

        # sent2frame from Task 1 ensemble output
        self.sent2frame = {}
        for item in task1_data:
            self.sent2frame[item[0]] = item[1]

        # frame_name → frame_id mapping
        self.frame2idx = {}
        for i, line in enumerate(self.ori_labels):
            self.frame2idx[line["frame_name"]] = i

        self.data_dict = {}
        for line in self.all_data:
            text = line["text"]
            target = [line["target"][-1]["start"] + 1, line["target"][-1]["end"] + 1]
            self.data_dict[line["sentence_id"]] = {"text": text, "target": target}

        self.data = []
        for line in self.task2_data:
            sent_id = line[0]
            text = self.data_dict[sent_id]["text"]
            target = self.data_dict[sent_id]["target"]
            if line[2] + 1 < target[0]:
                label_idx = [line[1] + 1, line[2] + 1]
            elif line[1] + 1 > target[1]:
                label_idx = [line[1] + 3, line[2] + 3]
            else:
                # Span overlaps with target — skip (shouldn't happen, but ensemble may produce noise)
                continue
            frame_name = self.sent2frame.get(sent_id, self.ori_labels[0]["frame_name"])
            frame_id = self.frame2idx[frame_name]
            self.data.append({
                'text': text,
                "label_idx": label_idx,
                "sentence_id": sent_id,
                "target": target,
                "ori_target": [line[1], line[2]],
                "frame_id": frame_id
            })
        self.num_labels = len(self.idx2label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        d1 = self.data[item]
        data = self.tokenizer.encode_plus(list(d1['text']))
        input_ids = data.data['input_ids']
        attention_mask = data.data['attention_mask']
        label_idx = d1["label_idx"]
        target = d1["target"]
        input_ids = input_ids[0: target[0]] + [1] + input_ids[target[0]: target[1] + 1] + [2] + input_ids[target[1] + 1:]
        attention_mask = attention_mask + [1, 1]
        sentence_id = d1["sentence_id"]
        ori_target = d1["ori_target"]
        frame_id = d1["frame_id"]
        return input_ids, attention_mask, label_idx, sentence_id, ori_target, frame_id


# ============================================================
# Collate functions
# ============================================================

def pad(d, max_len, v=0):
    return d + [v] * (max_len - len(d))


def collate_task1(data, device=None):
    bs = len(data)
    max_len = max([len(x[0]) for x in data])
    input_ids_list, attention_mask_list, target, sentence_id = [], [], [], []
    for d in data:
        input_ids_list.append(pad(d[0], max_len, 0))
        attention_mask_list.append(pad(d[1], max_len, 0))
        target.append(d[2])
        sentence_id.append(d[3])
    input_ids = torch.from_numpy(np.array(input_ids_list, dtype=np.compat.long)).to(device)
    attention_mask = torch.from_numpy(np.array(attention_mask_list, dtype=np.compat.long)).to(device)
    return input_ids, attention_mask, target, sentence_id


def collate_task2(data, device=None):
    bs = len(data)
    max_len = max([len(x[0]) for x in data])
    input_ids_list, attention_mask_list, target, sentence_id = [], [], [], []
    for d in data:
        input_ids_list.append(pad(d[0], max_len, 0))
        attention_mask_list.append(pad(d[1], max_len, 0))
        target.append(d[2])
        sentence_id.append(d[3])
    input_ids = torch.from_numpy(np.array(input_ids_list, dtype=np.compat.long)).to(device)
    attention_mask = torch.from_numpy(np.array(attention_mask_list, dtype=np.compat.long)).to(device)
    return input_ids, attention_mask, target, sentence_id


def collate_task3(data, device=None):
    bs = len(data)
    max_len = max([len(x[0]) for x in data])
    input_ids_list, attention_mask_list, target, sentence_id, ori_target, frame_ids = [], [], [], [], [], []
    for d in data:
        input_ids_list.append(pad(d[0], max_len, 0))
        attention_mask_list.append(pad(d[1], max_len, 0))
        target.append(d[2])
        sentence_id.append(d[3])
        ori_target.append(d[4])
        frame_ids.append(d[5])
    input_ids = torch.from_numpy(np.array(input_ids_list, dtype=np.compat.long)).to(device)
    attention_mask = torch.from_numpy(np.array(attention_mask_list, dtype=np.compat.long)).to(device)
    frame_ids = torch.tensor(frame_ids, dtype=torch.long).to(device)
    return input_ids, attention_mask, target, sentence_id, ori_target, frame_ids


# ============================================================
# Model loading
# ============================================================

def load_task1_models(seeds, config, device):
    """Load Task 1 models for all seeds."""
    models = []
    for seed in seeds:
        ckpt = f"saves/model_task1_best_seed{seed}.bin"
        if not os.path.exists(ckpt):
            print(f"[WARN] Missing checkpoint: {ckpt}, skipping seed {seed}")
            continue
        model = Model1(config)
        state = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
        models.append((seed, model))
    return models


def load_task2_models(seeds, config, device):
    """Load Task 2 models for all seeds."""
    models = []
    for seed in seeds:
        ckpt = f"saves/model_task2_best_seed{seed}.bin"
        if not os.path.exists(ckpt):
            print(f"[WARN] Missing checkpoint: {ckpt}, skipping seed {seed}")
            continue
        model = Model2(config)
        state = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
        models.append((seed, model))
    return models


def load_task3_models(seeds, config, device):
    """Load Task 3 models for all seeds."""
    models = []
    for seed in seeds:
        ckpt = f"saves/model_task3_best_seed{seed}.bin"
        if not os.path.exists(ckpt):
            print(f"[WARN] Missing checkpoint: {ckpt}, skipping seed {seed}")
            continue
        model = Model3(config)
        state = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
        models.append((seed, model))
    return models


# ============================================================
# Ensemble inference
# ============================================================

def ensemble_task1(models, test_loader, idx2label, device):
    """
    Task 1 Ensemble: soft voting.
    Average logits from all models, then argmax.
    """
    all_predictions = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc='Task1 Ensemble'):
            input_ids, attention_mask, target, sentence_ids = batch

            # Collect logits from all models
            all_logits = []
            for seed, model in models:
                output = model(input_ids=input_ids, attention_mask=attention_mask,
                              target=target, labels=None, device=device, for_test=True)
                all_logits.append(output["logits"])

            # Average logits (soft voting)
            avg_logits = torch.stack(all_logits).mean(dim=0)
            pred = torch.argmax(F.softmax(avg_logits, dim=-1), dim=-1)

            for i in range(len(sentence_ids)):
                all_predictions.append([sentence_ids[i], idx2label[pred[i]]])

    print(f"  Task 1 predictions: {len(all_predictions)}")
    return all_predictions


def ensemble_task2(models, test_loader, device):
    """
    Task 2 Ensemble: span majority voting.
    Each model predicts spans (logits >= 0). Keep spans predicted by ≥ ceil(N/2) models.
    """
    # First pass: collect all span predictions from each model, per sentence
    # sent_spans[sent_id] = [model_0_spans, model_1_spans, ...]
    sent_spans = {}  # sent_id -> list of list of (start, end)

    with torch.no_grad():
        for seed, model in models:
            print(f"  Task 2 seed {seed} predicting...")
            for step, batch in tqdm(enumerate(test_loader), total=len(test_loader),
                                     desc=f'Task2 seed{seed}'):
                input_ids, attention_mask, target, sentence_ids = batch

                output = model(input_ids=input_ids, attention_mask=attention_mask,
                              target=target, labels=None, device=device, for_test=True)

                H_attention_mask = torch.triu(
                    torch.matmul(attention_mask.unsqueeze(2).float(),
                                 attention_mask.unsqueeze(1).float()), diagonal=0)
                H_pred = torch.where(
                    output["logits"] >= 0,
                    torch.ones(output["logits"].shape).to(device),
                    torch.zeros(output["logits"].shape).to(device)
                ) * H_attention_mask

                predict_idx = torch.nonzero(H_pred)
                for i in range(len(sentence_ids)):
                    sid = sentence_ids[i]
                    if sid not in sent_spans:
                        sent_spans[sid] = []
                    spans = []
                    for idx in predict_idx:
                        if idx[0] != i:
                            continue
                        if idx[2] < target[i][0]:
                            spans.append((idx[1].item() - 1, idx[2].item() - 1))
                        elif idx[1] > target[i][1]:
                            spans.append((idx[1].item() - 3, idx[2].item() - 3))
                    sent_spans[sid].append(set(spans))

    # Second pass: majority voting
    n_models = len(models)
    threshold = max(2, (n_models + 1) // 2)  # majority: ≥ ceil(N/2), but at least 2
    print(f"  Span voting: threshold = {threshold}/{n_models}")

    all_predictions = []
    for sid, model_span_sets in sent_spans.items():
        # Flatten all spans and count occurrences
        span_counter = Counter()
        for span_set in model_span_sets:
            for span in span_set:
                span_counter[span] += 1

        # Keep spans with >= threshold votes
        voted_spans = [span for span, count in span_counter.items() if count >= threshold]
        for start, end in voted_spans:
            all_predictions.append([sid, start, end])

    print(f"  Task 2 predictions: {len(all_predictions)}")
    return all_predictions


def ensemble_task3(models, test_loader, idx2label, device):
    """
    Task 3 Ensemble: soft voting.
    Average logits from all models, then argmax.
    """
    all_predictions = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc='Task3 Ensemble'):
            input_ids, attention_mask, target, sentence_ids, ori_target, frame_ids = batch

            # Collect logits from all models
            all_logits = []
            for seed, model in models:
                output = model(input_ids=input_ids, attention_mask=attention_mask,
                              target=target, labels=None, device=device, for_test=True,
                              frame_ids=frame_ids)
                all_logits.append(output["logits"])

            # Average logits (soft voting)
            avg_logits = torch.stack(all_logits).mean(dim=0)
            pred = torch.argmax(F.softmax(avg_logits, dim=-1), dim=-1)

            for i in range(len(pred)):
                all_predictions.append([sentence_ids[i], ori_target[i][0], ori_target[i][1],
                                        idx2label[pred[i]]])

    print(f"  Task 3 predictions: {len(all_predictions)}")
    return all_predictions


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Multi-seed Ensemble Prediction')
    parser.add_argument('--seeds', type=str, default='42,123,456',
                        help='Comma-separated list of seeds (default: 42,123,456)')
    parser.add_argument('--test_file', type=str, default='cfn-test-B.json',
                        help='Test file name (cfn-test-A.json or cfn-test-B.json)')
    parser.add_argument('--data_dir', type=str, default='./dataset',
                        help='Dataset directory')
    parser.add_argument('--output_prefix', type=str, default='B',
                        help='Output file prefix (A or B)')
    parser.add_argument('--batch_size', type=int, default=3,
                        help='Batch size')
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    print(f"Ensemble seeds: {seeds}")
    print(f"Test file: {args.test_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Shared config
    config_file = './chinese_roberta_wwm_ext/config.json'
    vocab_file = './chinese_roberta_wwm_ext/vocab.txt'

    tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=True)

    # ---- Phase 1: Task 1 Ensemble ----
    print("\n" + "=" * 60)
    print("  Phase 1: Task 1 Ensemble (Frame Identification)")
    print("=" * 60)

    dataset1 = DatasetTask1(
        os.path.join(args.data_dir, args.test_file),
        os.path.join(args.data_dir, 'frame_info.json'),
        tokenizer)

    config1 = BertConfig.from_json_file(config_file)
    config1.num_labels = dataset1.num_labels

    models1 = load_task1_models(seeds, config1, device)
    print(f"  Loaded {len(models1)} Task 1 models: {[s for s, _ in models1]}")

    test_loader1 = DataLoader(
        batch_size=args.batch_size, dataset=dataset1, shuffle=False,
        num_workers=0, collate_fn=partial(collate_task1, device=device), drop_last=False)

    task1_preds = ensemble_task1(models1, test_loader1, dataset1.idx2label, device)

    task1_file = f'dataset/{args.output_prefix}_task1_test_ensemble.json'
    with open(task1_file, 'w', encoding='utf8') as f:
        json.dump(task1_preds, f, indent=1, ensure_ascii=False)
    print(f"  Saved: {task1_file}")

    # ---- Phase 2: Task 2 Ensemble ----
    print("\n" + "=" * 60)
    print("  Phase 2: Task 2 Ensemble (Argument Span Detection)")
    print("=" * 60)

    dataset2 = DatasetTask2(
        os.path.join(args.data_dir, args.test_file),
        os.path.join(args.data_dir, 'frame_info.json'),
        tokenizer)

    config2 = BertConfig.from_json_file(config_file)
    config2.num_labels = 1
    config2.max_cls = dataset2.max_cls

    models2 = load_task2_models(seeds, config2, device)
    print(f"  Loaded {len(models2)} Task 2 models: {[s for s, _ in models2]}")

    test_loader2 = DataLoader(
        batch_size=args.batch_size, dataset=dataset2, shuffle=False,
        num_workers=0, collate_fn=partial(collate_task2, device=device), drop_last=False)

    task2_preds = ensemble_task2(models2, test_loader2, device)

    task2_file = f'dataset/{args.output_prefix}_task2_test_ensemble.json'
    with open(task2_file, 'w', encoding='utf8') as f:
        json.dump(task2_preds, f, indent=1, ensure_ascii=False)
    print(f"  Saved: {task2_file}")

    # ---- Phase 3: Task 3 Ensemble ----
    print("\n" + "=" * 60)
    print("  Phase 3: Task 3 Ensemble (Role Classification)")
    print("=" * 60)

    dataset3 = DatasetTask3(
        os.path.join(args.data_dir, args.test_file),
        os.path.join(args.data_dir, 'frame_info.json'),
        task1_file,  # Uses Task 1 ensemble output
        task2_file,  # Uses Task 2 ensemble output
        tokenizer)

    config3 = BertConfig.from_json_file(config_file)
    config3.num_labels = dataset3.num_labels
    config3.num_frames = len(dataset3.frame2idx)

    models3 = load_task3_models(seeds, config3, device)
    print(f"  Loaded {len(models3)} Task 3 models: {[s for s, _ in models3]}")

    test_loader3 = DataLoader(
        batch_size=args.batch_size, dataset=dataset3, shuffle=False,
        num_workers=0, collate_fn=partial(collate_task3, device=device), drop_last=False)

    task3_preds = ensemble_task3(models3, test_loader3, dataset3.idx2label, device)

    task3_file = f'dataset/{args.output_prefix}_task3_test_ensemble.json'
    with open(task3_file, 'w', encoding='utf8') as f:
        json.dump(task3_preds, f, indent=1, ensure_ascii=False)
    print(f"  Saved: {task3_file}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  Ensemble Complete!")
    print(f"  Task 1: {len(task1_preds)} predictions → {task1_file}")
    print(f"  Task 2: {len(task2_preds)} predictions → {task2_file}")
    print(f"  Task 3: {len(task3_preds)} predictions → {task3_file}")
    print("=" * 60)

    # ---- Submission zip ----
    import zipfile
    zip_name = f'dataset/submit_ensemble_{args.output_prefix}.zip'
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(task1_file, f'{args.output_prefix}_task1_test.json')
        zf.write(task2_file, f'{args.output_prefix}_task2_test.json')
        zf.write(task3_file, f'{args.output_prefix}_task3_test.json')
    print(f"  Submission: {zip_name}")


if __name__ == '__main__':
    main()
