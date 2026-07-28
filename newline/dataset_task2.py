import torch
import codecs
import json
from transformers import BertTokenizer
from aeda import aeda_augment


class Dataset(torch.utils.data.Dataset):

    def __init__(self, json_file, label_file, tokenizer, for_test=False, augment_train=False):
        aeda_chars = [".", ";", "?", ":", "!", ",", "，", "。"]
        self.for_test = for_test
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

        # Phase 3.3: AEDA augmentation — 训练集翻倍
        if augment_train:
            augmented = []
            for item in self.all_data:
                text = item['text']
                target = item['target'][-1]
                spans = [[s['start'], s['end']]
                         for s in item['cfn_spans']]
                new_text, new_target, new_spans = aeda_augment(
                    text, target['start'], target['end'], spans)
                new_cfn_spans = []
                for i, s in enumerate(item['cfn_spans']):
                    new_cfn_spans.append({
                        'start': new_spans[i][0],
                        'end': new_spans[i][1],
                        'fe_abbr': s['fe_abbr'],
                        'fe_name': s['fe_name']
                    })
                augmented.append({
                    'text': new_text,
                    'target': [{'start': new_target[0],
                                'end': new_target[1],
                                'pos': target['pos']}],
                    'frame': item['frame'],
                    'cfn_spans': new_cfn_spans,
                    'sentence_id': item['sentence_id'] + 100000
                })
            self.all_data.extend(augmented)

        pass

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        d1 = self.all_data[item]
        data = self.tokenizer.encode_plus(list(d1['text']))
        input_ids = data.data['input_ids']
        attention_mask = data.data['attention_mask']
        target = [d1["target"][-1]["start"] + 1, d1["target"][-1]["end"] + 1]
        target_cls = self.label2cls[d1["frame"]]
        label = []
        for line in d1["cfn_spans"]:
            if line["end"] + 1 < target[0]:
                label.append([line["start"] + 1, line["end"] + 1])
            elif line["start"] + 1 > target[1]:
                label.append([line["start"] + 3, line["end"] + 3])
        input_ids = input_ids[0: target[0]] + [1] + input_ids[target[0]: target[1] + 1] + [2] + input_ids[target[1] + 1:]
        attention_mask = attention_mask + [1, 1]
        sentence_id = d1["sentence_id"]

        return input_ids, attention_mask, target, label, sentence_id, target_cls


if __name__ == '__main__':
    tokenizer = BertTokenizer(
        vocab_file='./chinese_bert_wwm_ext/vocab.txt',
        do_lower_case=True)
    dataset = Dataset("./dataset/cfn-train.json",
                      "./dataset/frame_info.json",
                      tokenizer=tokenizer)

    dataset[0]





