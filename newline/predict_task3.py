#!/usr/bin/python3

import torch
import codecs
import json
from functools import partial
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer, BertForTokenClassification
from params import args
from model_task3 import Model


class Dataset(torch.utils.data.Dataset):

    def __init__(self, json_file, label_file, task1_file, task2_file, tokenizer, for_test=False):
        aeda_chars = [".", ";", "?", ":", "!", ",", "，", "。"]
        self.for_test = for_test
        self.tokenizer = tokenizer
        with codecs.open(json_file, 'r', encoding='utf8') as f:
            self.all_data = json.load(f)
        with codecs.open(label_file, 'r', encoding='utf8') as f:
            self.ori_labels = json.load(f)
        with codecs.open(task1_file, 'r', encoding='utf8') as f:  # Phase 3.2
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

        # Phase 3.2: sent2frame from Task 1 output
        self.sent2frame = {}
        for item in task1_data:
            self.sent2frame[item[0]] = item[1]

        # Phase 3.2: frame_name → frame_id mapping
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
            # Phase 3.2: get frame_id from Task 1 prediction
            frame_name = self.sent2frame.get(sent_id, self.ori_labels[0]["frame_name"])
            frame_id = self.frame2idx[frame_name]
            self.data.append({
                'text': text,
                "label_idx": label_idx,
                "sentence_id": sent_id,
                "target": target,
                "ori_target": [line[1], line[2]],
                "frame_id": frame_id  # Phase 3.2
            })
        self.num_labels = len(self.idx2label)
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        d1 = self.data[item]
        data = self.tokenizer.encode_plus(list(d1['text']))
        input_ids = data.data['input_ids']
        attention_mask = data.data['attention_mask']
        label_idx = d1["label_idx"]
        target = d1["target"]
        input_ids = input_ids[0: target[0]] + [1] + input_ids[target[0]: target[1] + 1] + [2] + input_ids[
                                                                                                target[1] + 1:]
        attention_mask = attention_mask + [1, 1]
        sentence_id = d1["sentence_id"]
        ori_target = d1["ori_target"]
        frame_id = d1["frame_id"]  # Phase 3.2

        return input_ids, attention_mask, label_idx, sentence_id, ori_target, frame_id


def get_model_input(data, device=None):
    """

    :param data: input_ids1, input_ids2, label_starts, label_ends, true_label
    :return:
    """

    def pad(d, max_len, v=0):
        return d + [v] * (max_len - len(d))

    bs = len(data)
    max_len = max([len(x[0]) for x in data])

    input_ids_list = []
    attention_mask_list = []
    target = []
    sentence_id = []
    ori_target = []
    frame_ids = []  # Phase 3.2

    for d in data:
        input_ids_list.append(pad(d[0], max_len, 0))
        attention_mask_list.append(pad(d[1], max_len, 0))
        target.append(d[2])
        sentence_id.append(d[3])
        ori_target.append(d[4])
        frame_ids.append(d[5])  # Phase 3.2

    input_ids = np.array(input_ids_list, dtype=np.compat.long)
    attention_mask = np.array(attention_mask_list, dtype=np.compat.long)

    input_ids = torch.from_numpy(input_ids).to(device)
    attention_mask = torch.from_numpy(attention_mask).to(device)
    frame_ids = torch.tensor(frame_ids, dtype=torch.long).to(device)  # Phase 3.2

    return input_ids, attention_mask, target, sentence_id, ori_target, frame_ids


def test(model, val_loader):
    model.eval()
    idx2label = val_loader.dataset.idx2label
    predicts = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc='eval'):
            input_ids, attention_mask, target, sentence_id, ori_target, frame_ids = batch

            output = model(input_ids=input_ids, attention_mask=attention_mask, target=target, labels=None,
                           device=device, for_test=True, frame_ids=frame_ids)
            logits = output["logits"]
            pred = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
            for i in range(len(pred)):
                predicts.append([sentence_id[i], ori_target[i][0], ori_target[i][1], idx2label[pred[i]]])

            pass
    data_json = json.dumps(predicts, indent=1, ensure_ascii=False)
    with open('dataset/B_task3_test.json', 'w', encoding='utf8', newline='\n') as f:
        f.write(data_json)


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer(vocab_file=args.vocab_file,
                              do_lower_case=True)

    test_dataset = Dataset("./dataset/cfn-test-B.json",
                            "./dataset/frame_info.json",
                            "./dataset/B_task1_test.json",   # Phase 3.2: Task 1 output
                            "./dataset/B_task2_test.json",
                            tokenizer)

    config = BertConfig.from_json_file(args.config_file)
    # BertConfig.from_pretrained('hfl/chinese-bert-wwm-ext')
    config.num_labels = test_dataset.num_labels
    config.num_frames = len(test_dataset.frame2idx)  # Phase 3.2
    model = Model(config)
    # load_pretrained_bert(model, args.init_checkpoint)
    state = torch.load("saves/model_task3_best.bin", map_location='cpu')
    msg = model.load_state_dict(state, strict=False)
    # model.load_state_dict(torch.load('', map_location='cpu'))
    model = model.to(device)

    test_loader = DataLoader(
        batch_size=args.batch_size,
        dataset=test_dataset,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(get_model_input, device=device),
        drop_last=False
    )
    test(model, test_loader)
