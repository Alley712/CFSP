import torch
import codecs
import json
from transformers import BertTokenizer


class Dataset(torch.utils.data.Dataset):

    def __init__(self, json_file, label_file, tokenizer, for_test=False):
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

        pass

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        d1 = self.all_data[item]
        chars = list(d1['text'])
        data = self.tokenizer(chars, is_split_into_words=True)
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        word_ids = data.word_ids()

        # 建立字符位置到token位置的映射（处理tokenizer剥离空白字符等情况）
        char_to_tokens = {}
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is not None:
                if word_idx not in char_to_tokens:
                    char_to_tokens[word_idx] = [token_idx, token_idx]
                else:
                    char_to_tokens[word_idx][1] = token_idx

        # 用word_ids映射target的起止位置
        target_start_char = d1["target"][-1]["start"]
        target_end_char = d1["target"][-1]["end"]
        if target_start_char not in char_to_tokens or target_end_char not in char_to_tokens:
            # 字符被tokenizer剥离的fallback，使用相邻token位置
            target = [d1["target"][-1]["start"] + 1, d1["target"][-1]["end"] + 1]
        else:
            target = [char_to_tokens[target_start_char][0], char_to_tokens[target_end_char][1]]

        target_cls = self.label2cls[d1["frame"]]

        # 用word_ids映射label位置
        label = []
        for line in d1["cfn_spans"]:
            char_start = line["start"]
            char_end = line["end"]
            if char_start not in char_to_tokens or char_end not in char_to_tokens:
                continue  # 跳过被tokenizer剥离的字符上的标注
            token_start = char_to_tokens[char_start][0]
            token_end = char_to_tokens[char_end][1]
            if token_end < target[0]:
                label.append([token_start, token_end])
            elif token_start > target[1]:
                # 插入[1]和[2]后，target之后的token整体后移2位
                label.append([token_start + 2, token_end + 2])

        input_ids = input_ids[0: target[0]] + [1] + input_ids[target[0]: target[1] + 1] + [2] + input_ids[target[1] + 1:]
        attention_mask = attention_mask + [1, 1]
        sentence_id = d1["sentence_id"]

        return input_ids, attention_mask, target, label, sentence_id, target_cls


if __name__ == '__main__':
    tokenizer = BertTokenizer(
        vocab_file='./chinese_bert_wwm_ext/vocab.txt',
        do_lower_case=True)
    dataset = Dataset("../data/cfn-dataset/cfn-train.json",
                      "../data/cfn-dataset/frame_info.json",
                      tokenizer=tokenizer)

    dataset[0]





