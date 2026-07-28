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
        self.idx2label = []
        self.label2idx = {}
        for i, line in enumerate(self.ori_labels):
            self.idx2label.append(line["frame_name"])
            self.label2idx[line["frame_name"]] = i

        self.num_labels = len(self.idx2label)

        # Phase 3.3: AEDA augmentation — 训练集翻倍
        if augment_train:
            augmented = []
            for item in self.all_data:
                text = item['text']
                target = item['target'][-1]
                new_text, new_target, _ = aeda_augment(
                    text, target['start'], target['end'], [])
                augmented.append({
                    'text': new_text,
                    'target': [{'start': new_target[0],
                                'end': new_target[1],
                                'pos': target['pos']}],
                    'frame': item['frame'],
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
        label = self.label2idx[d1["frame"]]
        sentence_id = d1["sentence_id"]

        return input_ids, attention_mask, target, label, sentence_id


if __name__ == '__main__':
    tokenizer = BertTokenizer(
        vocab_file='./chinese_bert_wwm_ext/vocab.txt',
        do_lower_case=True)
    dataset = Dataset("./dataset/cfn-train.json",
                      "./dataset/frame_info.json",
                      tokenizer=tokenizer)

    dataset[0]




