#!/bin/bash
# 在 dev 集上运行预测并算分
# 用法: bash predict_dev.sh
set -e
cd /root/autodl-tmp/CFSP/newline

echo "=== Task 1: 预测 dev 集 ==="
python -c "
import torch, json, codecs, numpy as np
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer
import torch.nn.functional as F
from params import args
from model_task1 import Model

class Dataset(torch.utils.data.Dataset):
    def __init__(self, json_file, label_file, tokenizer):
        with codecs.open(json_file, 'r', encoding='utf8') as f:
            self.all_data = json.load(f)
        with codecs.open(label_file, 'r', encoding='utf8') as f:
            self.ori_labels = json.load(f)
        self.idx2label = []
        self.label2idx = {}
        for i, line in enumerate(self.ori_labels):
            self.idx2label.append(line['frame_name'])
            self.label2idx[line['frame_name']] = i
        self.num_labels = len(self.idx2label)
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.all_data)
    def __getitem__(self, item):
        d1 = self.all_data[item]
        data = self.tokenizer.encode_plus(list(d1['text']))
        return data.data['input_ids'], data.data['attention_mask'], [d1['target'][-1]['start'] + 1, d1['target'][-1]['end'] + 1], d1['sentence_id']

def collate_fn(data, device):
    def pad(d, max_len, v=0): return d + [v] * (max_len - len(d))
    bs, max_len = len(data), max(len(x[0]) for x in data)
    input_ids = torch.from_numpy(np.array([pad(d[0], max_len, 0) for d in data], dtype=np.int64)).to(device)
    attention_mask = torch.from_numpy(np.array([pad(d[1], max_len, 0) for d in data], dtype=np.int64)).to(device)
    target = [d[2] for d in data]
    sentence_id = [d[3] for d in data]
    return input_ids, attention_mask, target, sentence_id

device = torch.device('cuda')
tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
ds = Dataset('./dataset/cfn-dev.json', './dataset/frame_info.json', tokenizer)
config = BertConfig.from_json_file(args.config_file)
config.num_labels = ds.num_labels
model = Model(config)
state = torch.load('saves/model_task1_best.bin', map_location='cpu')
model.load_state_dict(state, strict=False)
model = model.to(device)
model.eval()
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=partial(collate_fn, device=device))
preds = []
with torch.no_grad():
    for batch in tqdm(loader, desc='Task1 dev'):
        input_ids, attention_mask, target, sentence_id = batch
        logits = model(input_ids=input_ids, attention_mask=attention_mask, target=target, labels=None, device=device, for_test=True)['logits']
        pred = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
        for i in range(len(sentence_id)):
            preds.append([sentence_id[i], ds.idx2label[pred[i]]])
with open('dataset/dev_task1_pred.json', 'w', encoding='utf8') as f:
    json.dump(preds, f, ensure_ascii=False)
print(f'Task1 dev predictions: {len(preds)} saved')
"

echo "=== Task 2: 预测 dev 集 ==="
python -c "
import torch, json, codecs, numpy as np
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer
from params import args
from model_task2 import Model

class Dataset(torch.utils.data.Dataset):
    def __init__(self, json_file, label_file, tokenizer):
        with codecs.open(json_file, 'r', encoding='utf8') as f:
            self.all_data = json.load(f)
        with codecs.open(label_file, 'r', encoding='utf8') as f:
            self.ori_labels = json.load(f)
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.all_data)
    def __getitem__(self, item):
        d1 = self.all_data[item]
        data = self.tokenizer.encode_plus(list(d1['text']))
        input_ids = data.data['input_ids']
        attention_mask = data.data['attention_mask']
        target = [d1['target'][-1]['start'] + 1, d1['target'][-1]['end'] + 1]
        input_ids = input_ids[:target[0]] + [1] + input_ids[target[0]:target[1]+1] + [2] + input_ids[target[1]+1:]
        attention_mask = attention_mask + [1, 1]
        return input_ids, attention_mask, target, d1['sentence_id']

def collate_fn(data, device):
    def pad(d, max_len, v=0): return d + [v] * (max_len - len(d))
    bs, max_len = len(data), max(len(x[0]) for x in data)
    input_ids = torch.from_numpy(np.array([pad(d[0], max_len, 0) for d in data], dtype=np.int64)).to(device)
    attention_mask = torch.from_numpy(np.array([pad(d[1], max_len, 0) for d in data], dtype=np.int64)).to(device)
    target = [d[2] for d in data]
    sentence_id = [d[3] for d in data]
    return input_ids, attention_mask, target, sentence_id

device = torch.device('cuda')
tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
ds = Dataset('./dataset/cfn-dev.json', './dataset/frame_info.json', tokenizer)
config = BertConfig.from_json_file(args.config_file)
config.num_labels = 1
model = Model(config)
state = torch.load('saves/model_task2_best.bin', map_location='cpu')
model.load_state_dict(state, strict=False)
model = model.to(device)
model.eval()
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=partial(collate_fn, device=device))
preds = []
with torch.no_grad():
    for batch in tqdm(loader, desc='Task2 dev'):
        input_ids, attention_mask, target, sentence_id = batch
        logits = model(input_ids=input_ids, attention_mask=attention_mask, target=target, labels=None, device=device, for_test=True)['logits']
        H_attention_mask = torch.triu(torch.matmul(attention_mask.unsqueeze(2).float(), attention_mask.unsqueeze(1).float()), diagonal=0)
        H_pred = torch.where(logits >= 0, torch.ones(logits.shape).to(device), torch.zeros(logits.shape).to(device)) * H_attention_mask
        predict_idx = torch.nonzero(H_pred)
        for idx in predict_idx:
            bid, s, e = idx[0].item(), idx[1].item(), idx[2].item()
            if e < target[bid][0]:
                preds.append([sentence_id[bid], s - 1, e - 1])
            elif s > target[bid][1]:
                preds.append([sentence_id[bid], s - 3, e - 3])
with open('dataset/dev_task2_pred.json', 'w', encoding='utf8') as f:
    json.dump(preds, f, ensure_ascii=False)
print(f'Task2 dev predictions: {len(preds)} spans saved')
"

echo "=== Task 3: 预测 dev 集 ==="
python -c "
import torch, json, codecs, numpy as np
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer
import torch.nn.functional as F
from params import args
from model_task3 import Model

class Dataset(torch.utils.data.Dataset):
    def __init__(self, json_file, label_file, task1_file, task2_file, tokenizer):
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
            for fes in line['fes']:
                if fes['fe_name'] not in self.idx2label:
                    self.idx2label.append(fes['fe_name'])
        self.label2idx = {v:i for i,v in enumerate(self.idx2label)}
        self.sent2frame = {item[0]: item[1] for item in task1_data}
        self.frame2idx = {}
        for i, line in enumerate(self.ori_labels):
            self.frame2idx[line['frame_name']] = i
        self.data_dict = {}
        for line in self.all_data:
            self.data_dict[line['sentence_id']] = {
                'text': line['text'],
                'target': [line['target'][-1]['start'] + 1, line['target'][-1]['end'] + 1]
            }
        self.data = []
        for line in self.task2_data:
            sent_id = line[0]
            if sent_id not in self.data_dict: continue
            info = self.data_dict[sent_id]
            target = info['target']
            if line[2] + 1 < target[0]:
                label_idx = [line[1] + 1, line[2] + 1]
            elif line[1] + 1 > target[1]:
                label_idx = [line[1] + 3, line[2] + 3]
            else: continue
            frame_name = self.sent2frame.get(sent_id, self.ori_labels[0]['frame_name'])
            frame_id = self.frame2idx[frame_name]
            self.data.append({'text': info['text'], 'label_idx': label_idx, 'sentence_id': sent_id, 'target': target, 'ori_target': [line[1], line[2]], 'frame_id': frame_id})
        self.num_labels = len(self.idx2label)
        self.tokenizer = tokenizer
    def __len__(self): return len(self.data)
    def __getitem__(self, item):
        d1 = self.data[item]
        data = self.tokenizer.encode_plus(list(d1['text']))
        input_ids = data.data['input_ids']
        attention_mask = data.data['attention_mask']
        target = d1['target']
        input_ids = input_ids[:target[0]] + [1] + input_ids[target[0]:target[1]+1] + [2] + input_ids[target[1]+1:]
        attention_mask = attention_mask + [1, 1]
        return input_ids, attention_mask, d1['label_idx'], d1['sentence_id'], d1['ori_target'], d1['frame_id']

def collate_fn(data, device):
    def pad(d, max_len, v=0): return d + [v] * (max_len - len(d))
    bs, max_len = len(data), max(len(x[0]) for x in data)
    input_ids = torch.from_numpy(np.array([pad(d[0], max_len, 0) for d in data], dtype=np.int64)).to(device)
    attention_mask = torch.from_numpy(np.array([pad(d[1], max_len, 0) for d in data], dtype=np.int64)).to(device)
    target = [d[2] for d in data]
    sentence_id = [d[3] for d in data]
    ori_target = [d[4] for d in data]
    frame_ids = torch.tensor([d[5] for d in data], dtype=torch.long).to(device)
    return input_ids, attention_mask, target, sentence_id, ori_target, frame_ids

device = torch.device('cuda')
tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
ds = Dataset('./dataset/cfn-dev.json', './dataset/frame_info.json', './dataset/dev_task1_pred.json', './dataset/dev_task2_pred.json', tokenizer)
config = BertConfig.from_json_file(args.config_file)
config.num_labels = ds.num_labels
config.num_frames = len(ds.frame2idx)
model = Model(config)
state = torch.load('saves/model_task3_best.bin', map_location='cpu')
model.load_state_dict(state, strict=False)
model = model.to(device)
model.eval()
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=partial(collate_fn, device=device))
preds = []
with torch.no_grad():
    for batch in tqdm(loader, desc='Task3 dev'):
        input_ids, attention_mask, target, sentence_id, ori_target, frame_ids = batch
        logits = model(input_ids=input_ids, attention_mask=attention_mask, target=target, labels=None, device=device, for_test=True, frame_ids=frame_ids)['logits']
        pred = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
        for i in range(len(pred)):
            preds.append([sentence_id[i], ori_target[i][0], ori_target[i][1], ds.idx2label[pred[i]]])
with open('dataset/dev_task3_pred.json', 'w', encoding='utf8') as f:
    json.dump(preds, f, ensure_ascii=False)
print(f'Task3 dev predictions: {len(preds)} saved')
"

echo ""
echo "=== 算分 ==="
python score.py -g ../data/cfn-dataset/cfn-dev.json \
    -1 dataset/dev_task1_pred.json \
    -2 dataset/dev_task2_pred.json \
    -3 dataset/dev_task3_pred.json
