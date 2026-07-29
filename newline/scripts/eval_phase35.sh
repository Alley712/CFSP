#!/bin/bash
# Phase 3.5 全流程 dev 评测：三任务 NoisyTune OFF
set -e
cd /root/autodl-tmp/CFSP/newline

echo "=== Task1 dev prediction (NoisyTune OFF) ==="
python3 -c "
import torch, json, codecs, numpy as np, torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer
from params import args
from model_task1 import Model

class DS(torch.utils.data.Dataset):
    def __init__(self, jf, lf, tok):
        with codecs.open(jf,'r') as f: self.data = json.load(f)
        with codecs.open(lf,'r') as f: self.labels = json.load(f)
        self.i2l = [l['frame_name'] for l in self.labels]; self.tok = tok
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        d = self.data[i]; e = self.tok.encode_plus(list(d['text']))
        return e.data['input_ids'], e.data['attention_mask'], [d['target'][-1]['start']+1, d['target'][-1]['end']+1], d['sentence_id']

def cf(data, dev):
    def pad(d, mx, v=0): return d+[v]*(mx-len(d))
    mx = max(len(x[0]) for x in data)
    ids = torch.from_numpy(np.array([pad(d[0],mx,0) for d in data], dtype=np.int64)).to(dev)
    am = torch.from_numpy(np.array([pad(d[1],mx,0) for d in data], dtype=np.int64)).to(dev)
    return ids, am, [d[2] for d in data], [d[3] for d in data]

dev = torch.device('cuda')
tok = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
ds = DS('./dataset/cfn-dev.json', './dataset/frame_info.json', tok)
cfg = BertConfig.from_json_file(args.config_file); cfg.num_labels = len(ds.i2l)
m = Model(cfg); m.load_state_dict(torch.load('saves/model_task1_best_noNT.bin', map_location='cpu'), strict=False)
m = m.to(dev); m.eval()
ldr = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=partial(cf, dev=dev))
preds = []
with torch.no_grad():
    for ids, am, tgt, sids in tqdm(ldr, desc='Task1'):
        lg = m(input_ids=ids, attention_mask=am, target=tgt, labels=None, device=dev, for_test=True)['logits']
        p = torch.argmax(F.softmax(lg, dim=-1), dim=-1)
        for i in range(len(p)): preds.append([sids[i], ds.i2l[p[i]]])
json.dump(preds, open('dataset/dev_task1_pred_nnt.json','w'), ensure_ascii=False)
print(f'Task1: {len(preds)} predictions')
"

echo "=== Task2 dev prediction (NoisyTune OFF) ==="
python3 -c "
import torch, json, numpy as np
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer
import codecs
from params import args
from model_task2 import Model

class DS(torch.utils.data.Dataset):
    def __init__(self, jf, lf, tok):
        with codecs.open(jf,'r') as f: self.data = json.load(f)
        with codecs.open(lf,'r') as f: self.labels = json.load(f); self.tok = tok
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        d = self.data[i]; e = self.tok.encode_plus(list(d['text']))
        ids = e.data['input_ids']; am = e.data['attention_mask']
        t = [d['target'][-1]['start']+1, d['target'][-1]['end']+1]
        ids = ids[:t[0]]+[1]+ids[t[0]:t[1]+1]+[2]+ids[t[1]+1:]; am += [1,1]
        return ids, am, t, d['sentence_id']

def cf(data, dev):
    def pad(d, mx, v=0): return d+[v]*(mx-len(d))
    mx = max(len(x[0]) for x in data)
    ids = torch.from_numpy(np.array([pad(d[0],mx,0) for d in data], dtype=np.int64)).to(dev)
    am = torch.from_numpy(np.array([pad(d[1],mx,0) for d in data], dtype=np.int64)).to(dev)
    return ids, am, [d[2] for d in data], [d[3] for d in data]

device = torch.device('cuda')
tok = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
ds = DS('./dataset/cfn-dev.json', './dataset/frame_info.json', tok)
cfg = BertConfig.from_json_file(args.config_file); cfg.num_labels = 1
m = Model(cfg); m.load_state_dict(torch.load('saves/model_task2_best_noNT.bin', map_location='cpu'), strict=False)
m = m.to(device); m.eval()
ldr = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=partial(cf, dev=device))
preds = []
with torch.no_grad():
    for ids, am, tgt, sids in tqdm(ldr, desc='Task2'):
        lg = m(input_ids=ids, attention_mask=am, target=tgt, labels=None, device=device, for_test=True)['logits']
        H_am = torch.triu(torch.matmul(am.unsqueeze(2).float(), am.unsqueeze(1).float()), diagonal=0)
        H_p = torch.where(lg >= 0, torch.ones(lg.shape).to(device), torch.zeros(lg.shape).to(device)) * H_am
        for idx in torch.nonzero(H_p):
            b,s,e = idx[0].item(), idx[1].item(), idx[2].item()
            if e < tgt[b][0]: preds.append([sids[b], s-1, e-1])
            elif s > tgt[b][1]: preds.append([sids[b], s-3, e-3])
json.dump(preds, open('dataset/dev_task2_pred_nnt.json','w'), ensure_ascii=False)
print(f'Task2: {len(preds)} spans')
"

echo "=== Task3 dev prediction (NoisyTune OFF) ==="
python3 -c "
import torch, json, numpy as np, torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer
import codecs
from params import args
from model_task3 import Model

class DS(torch.utils.data.Dataset):
    def __init__(self, jf, lf, t1f, t2f, tok):
        with codecs.open(jf,'r') as f: self.all_data = json.load(f)
        with codecs.open(lf,'r') as f: self.ori_labels = json.load(f)
        with codecs.open(t1f,'r') as f: self.task1_data = json.load(f)
        self.sent2frame = {it[0]: it[1] for it in self.task1_data}
        with codecs.open(t2f,'r') as f: self.task2_data = json.load(f)
        self.i2l = []
        for ln in self.ori_labels:
            for fe in ln['fes']:
                if fe['fe_name'] not in self.i2l: self.i2l.append(fe['fe_name'])
        self.l2i = {v:i for i,v in enumerate(self.i2l)}
        self.frame2idx = {}
        for i, ln in enumerate(self.ori_labels): self.frame2idx[ln['frame_name']] = i
        self.ddict = {}
        for ln in self.all_data:
            self.ddict[ln['sentence_id']] = {'text': ln['text'], 'target': [ln['target'][-1]['start']+1, ln['target'][-1]['end']+1]}
        self.data = []
        for ln in self.task2_data:
            sid = ln[0]
            if sid not in self.ddict: continue
            info = self.ddict[sid]; tgt = info['target']
            if ln[2]+1 < tgt[0]: lidx = [ln[1]+1, ln[2]+1]
            elif ln[1]+1 > tgt[1]: lidx = [ln[1]+3, ln[2]+3]
            else: continue
            fname = self.sent2frame.get(sid, self.ori_labels[0]['frame_name'])
            fid = self.frame2idx[fname]
            self.data.append({'text': info['text'], 'label_idx': lidx, 'sentence_id': sid, 'target': tgt, 'ori_target': [ln[1], ln[2]], 'frame_id': fid})
        self.tok = tok
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        d = self.data[i]; e = self.tok.encode_plus(list(d['text']))
        ids = e.data['input_ids']; am = e.data['attention_mask']; t = d['target']
        ids = ids[:t[0]]+[1]+ids[t[0]:t[1]+1]+[2]+ids[t[1]+1:]; am = am+[1,1]
        return ids, am, d['label_idx'], d['sentence_id'], d['ori_target'], d['frame_id']

def cf(data, dev):
    def pad(d, mx, v=0): return d+[v]*(mx-len(d))
    mx = max(len(x[0]) for x in data)
    ids = torch.from_numpy(np.array([pad(d[0],mx,0) for d in data], dtype=np.int64)).to(dev)
    am = torch.from_numpy(np.array([pad(d[1],mx,0) for d in data], dtype=np.int64)).to(dev)
    fids = torch.tensor([d[5] for d in data], dtype=torch.long).to(dev)
    return ids, am, [d[2] for d in data], [d[3] for d in data], [d[4] for d in data], fids

device = torch.device('cuda')
tok = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
ds = DS('./dataset/cfn-dev.json', './dataset/frame_info.json', './dataset/dev_task1_pred_nnt.json', './dataset/dev_task2_pred_nnt.json', tok)
cfg = BertConfig.from_json_file(args.config_file); cfg.num_labels = len(ds.i2l); cfg.num_frames = len(ds.frame2idx)
m = Model(cfg); m.load_state_dict(torch.load('saves/model_task3_best_noNT.bin', map_location='cpu'), strict=False)
m = m.to(device); m.eval()
ldr = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=partial(cf, dev=device))
preds = []
with torch.no_grad():
    for ids, am, tgt, sids, ori_tgt, fids in tqdm(ldr, desc='Task3'):
        lg = m(input_ids=ids, attention_mask=am, target=tgt, labels=None, device=device, for_test=True, frame_ids=fids)['logits']
        p = torch.argmax(F.softmax(lg, dim=-1), dim=-1)
        for i in range(len(p)): preds.append([sids[i], ori_tgt[i][0], ori_tgt[i][1], ds.i2l[p[i]]])
json.dump(preds, open('dataset/dev_task3_pred_nnt.json','w'), ensure_ascii=False)
print(f'Task3: {len(preds)} predictions')
"

echo ""
echo "=========================================="
echo "  全流程 NoisyTune OFF 评测"
echo "=========================================="
python score.py -g ../data/cfn-dataset/cfn-dev.json \
    -1 dataset/dev_task1_pred_nnt.json \
    -2 dataset/dev_task2_pred_nnt.json \
    -3 dataset/dev_task3_pred_nnt.json

echo ""
echo "=== 对比基线 ==="
python score.py -g ../data/cfn-dataset/cfn-dev.json \
    -1 dataset/dev_task1_pred.json \
    -2 dataset/dev_task2_pred.json \
    -3 dataset/dev_task3_pred.json 2>/dev/null | grep -E "Task [123]|总分|Acc |F1 "
