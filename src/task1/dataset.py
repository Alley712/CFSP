import torch
import codecs
import json
from transformers import AutoTokenizer


class Dataset(torch.utils.data.Dataset):

    def __init__(self, json_file, label_file, tokenizer, for_test=False):
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

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        d1 = self.all_data[item]
        # transformers >=5.x: use __call__ instead of encode_plus
        data = self.tokenizer(list(d1['text']), is_split_into_words=True)
        input_ids = data.data['input_ids']
        attention_mask = data.data['attention_mask']
        target = [d1["target"][-1]["start"] + 1, d1["target"][-1]["end"] + 1]
        label = self.label2idx[d1["frame"]]
        sentence_id = d1["sentence_id"]
        return input_ids, attention_mask, target, label, sentence_id
