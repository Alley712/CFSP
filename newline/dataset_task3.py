import torch
import codecs
import json
import random
from transformers import BertTokenizer


class Dataset(torch.utils.data.Dataset):

    def __init__(self, json_file, label_file, tokenizer, for_test=False,
                 add_negatives=False, neg_per_sentence=1):
        aeda_chars = [".", ";", "?", ":", "!", ",", "，", "。"]
        self.for_test = for_test
        self.tokenizer = tokenizer
        with codecs.open(json_file, 'r', encoding='utf8') as f:
            self.all_data = json.load(f)
        with codecs.open(label_file, 'r', encoding='utf8') as f:
            self.ori_labels = json.load(f)
        self.idx2label = []
        for line in self.ori_labels:
            for fes in line["fes"]:
                if fes["fe_name"] not in self.idx2label:
                    self.idx2label.append(fes["fe_name"])
        self.label2idx = {}
        for i in range(len(self.idx2label)):
            self.label2idx[self.idx2label[i]] = i

        # Phase 3.2: frame_name → frame_id mapping
        self.frame2idx = {}
        for i, line in enumerate(self.ori_labels):
            self.frame2idx[line["frame_name"]] = i

        self.data = []
        for line in self.all_data:
            text = line["text"]
            target = [line["target"][-1]["start"] + 1, line["target"][-1]["end"] + 1]
            cfn_spans = line["cfn_spans"]
            frame_id = self.frame2idx[line["frame"]]  # Phase 3.2: gold frame
            for spans in cfn_spans:
                if spans["end"] + 1 < target[0]:
                    label_idx = [spans["start"] + 1, spans["end"] + 1]
                elif spans["start"] + 1 > target[1]:
                    label_idx = [spans["start"] + 3, spans["end"] + 3]
                fe_text = text[spans["start"]: spans["end"] + 1]
                self.data.append({
                    'text': text,
                    "label_class": self.label2idx[spans["fe_name"]],
                    "label_idx": label_idx,
                    "sentence_id": line["sentence_id"],
                    "target": target,
                    "fe_text": fe_text,
                    "frame_id": frame_id  # Phase 3.2
                })

        # Phase 4: Negative sampling — add random non-gold spans as "None" class
        NONE_LABEL = len(self.idx2label)  # index of "None" = current num_labels
        neg_count = 0
        if add_negatives:
            for line in self.all_data:
                text = line["text"]
                tgt = line["target"][-1]
                tgt_start, tgt_end = tgt["start"], tgt["end"]
                cfn_spans = line["cfn_spans"]
                frame_id = self.frame2idx[line["frame"]]

                # Build occupied positions (gold spans + target word)
                occupied = set()
                for span in cfn_spans:
                    for pos in range(span["start"], span["end"] + 1):
                        occupied.add(pos)
                for pos in range(tgt_start, tgt_end + 1):
                    occupied.add(pos)

                available = [i for i in range(len(text)) if i not in occupied]
                if len(available) < 3:
                    continue

                for _ in range(neg_per_sentence * 3):
                    neg_start = random.choice(available)
                    candidates = [p for p in available
                                  if p > neg_start and p - neg_start <= 8]
                    if not candidates:
                        continue
                    neg_end = random.choice(candidates)

                    # Same coordinate offset logic as positive spans
                    if neg_end + 1 < tgt_start:
                        label_idx = [neg_start + 1, neg_end + 1]
                    elif neg_start + 1 > tgt_end:
                        label_idx = [neg_start + 3, neg_end + 3]
                    else:
                        continue  # overlaps with target, try again

                    self.data.append({
                        'text': text,
                        "label_class": NONE_LABEL,
                        "label_idx": label_idx,
                        "sentence_id": line["sentence_id"],
                        "target": [tgt_start + 1, tgt_end + 1],
                        "fe_text": "",
                        "frame_id": frame_id
                    })
                    neg_count += 1
                    break

        self.idx2label.append("None")
        self.num_labels = len(self.idx2label)
        if add_negatives:
            print(f"  Negatives added: {neg_count}, "
                  f"pos:neg = {len(self.data) - neg_count}:{neg_count}, "
                  f"num_labels = {self.num_labels}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        d1 = self.data[item]
        data = self.tokenizer.encode_plus(list(d1['text']))
        input_ids = data.data['input_ids']
        attention_mask = data.data['attention_mask']
        label_idx = d1["label_idx"]
        label = d1["label_class"]
        target = d1["target"]
        input_ids = input_ids[0: target[0]] + [1] + input_ids[target[0]: target[1] + 1] + [2] + input_ids[
                                                                                                target[1] + 1:]
        attention_mask = attention_mask + [1, 1]
        sentence_id = d1["sentence_id"]
        frame_id = d1["frame_id"]  # Phase 3.2

        return input_ids, attention_mask, label_idx, label, sentence_id, frame_id


if __name__ == '__main__':
    tokenizer = BertTokenizer(
        vocab_file='./chinese_bert_wwm_ext/vocab.txt',
        do_lower_case=True)
    dataset = Dataset("./dataset/cfn-train.json",
                      "./dataset/frame_info.json",
                      tokenizer=tokenizer)

    dataset[0]




