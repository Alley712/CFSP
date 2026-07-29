#!/bin/bash
# Phase 3.4 E12b: dev 集评测
# 1. Task1 预测 dev → top-K
# 2. Task2 预测 dev → spans
# 3. Task3 E12b 预测 dev → 角色
# 4. 算分
set -e
cd /root/autodl-tmp/CFSP/newline

echo "=========================================="
echo "  E12b dev 集评测"
echo "  TOPK=3  ALPHA=1.0  FUSION=max"
echo "=========================================="

python3 -c "
import torch, json, codecs, numpy as np, torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer
from params import args
from model_task1 import Model
TOPK = 5

class DS(torch.utils.data.Dataset):
    def __init__(self, jf, lf, tok):
        with codecs.open(jf, 'r') as f: self.data = json.load(f)
        with codecs.open(lf, 'r') as f: self.labels = json.load(f)
        self.i2l = [l['frame_name'] for l in self.labels]
        self.tok = tok
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        d = self.data[i]; e = self.tok.encode_plus(list(d['text']))
        return e.data['input_ids'], e.data['attention_mask'], [d['target'][-1]['start']+1, d['target'][-1]['end']+1], d['sentence_id']

def cf(data, dev):
    def pad(d, mx, v=0): return d + [v]*(mx-len(d))
    mx = max(len(x[0]) for x in data)
    ids = torch.from_numpy(np.array([pad(d[0],mx,0) for d in data], dtype=np.int64)).to(dev)
    am = torch.from_numpy(np.array([pad(d[1],mx,0) for d in data], dtype=np.int64)).to(dev)
    return ids, am, [d[2] for d in data], [d[3] for d in data]

dev = torch.device('cuda')
tok = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
ds = DS('./dataset/cfn-dev.json', './dataset/frame_info.json', tok)
cfg = BertConfig.from_json_file(args.config_file); cfg.num_labels = len(ds.i2l)
m = Model(cfg); m.load_state_dict(torch.load('saves/model_task1_best.bin', map_location='cpu'), strict=False)
m = m.to(dev); m.eval()
ldr = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=partial(cf, dev=dev))
preds, topk_preds = [], []
with torch.no_grad():
    for ids, am, tgt, sids in tqdm(ldr, desc='Task1 dev topk'):
        lg = m(input_ids=ids, attention_mask=am, target=tgt, labels=None, device=dev, for_test=True)['logits']
        probs = F.softmax(lg, dim=-1)
        p = torch.argmax(probs, dim=-1)
        tp, ti = torch.topk(probs, k=TOPK, dim=-1)
        for i in range(len(sids)):
            preds.append([sids[i], ds.i2l[p[i]]])
            topk_preds.append({'sentence_id': sids[i], 'topk': [[ds.i2l[ti[i,k].item()], round(tp[i,k].item(),6)] for k in range(TOPK)]})
json.dump(preds, open('dataset/dev_task1_pred.json','w'), ensure_ascii=False)
json.dump(topk_preds, open('dataset/dev_task1_pred_topk.json','w'), ensure_ascii=False)
print(f'Task1 dev: {len(preds)} top-1, {len(topk_preds)} top-K saved')
"

echo ""
echo "=== Task2 dev prediction ==="
python3 -c "
import torch, json, codecs, numpy as np
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer
from params import args
from model_task2 import Model

class DS(torch.utils.data.Dataset):
    def __init__(self, jf, lf, tok):
        with codecs.open(jf,'r') as f: self.data = json.load(f)
        with codecs.open(lf,'r') as f: self.labels = json.load(f)
        self.tok = tok
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        d = self.data[i]; e = self.tok.encode_plus(list(d['text']))
        ids = e.data['input_ids']; am = e.data['attention_mask']
        t = [d['target'][-1]['start']+1, d['target'][-1]['end']+1]
        ids = ids[:t[0]] + [1] + ids[t[0]:t[1]+1] + [2] + ids[t[1]+1:]
        return ids, am+[1,1], t, d['sentence_id']

def cf(data, dev):
    def pad(d, mx, v=0): return d+[v]*(mx-len(d))
    mx = max(len(x[0]) for x in data)
    ids = torch.from_numpy(np.array([pad(d[0],mx,0) for d in data], dtype=np.int64)).to(dev)
    am = torch.from_numpy(np.array([pad(d[1],mx,0) for d in data], dtype=np.int64)).to(dev)
    return ids, am, [d[2] for d in data], [d[3] for d in data]

dev = torch.device('cuda')
tok = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
ds = DS('./dataset/cfn-dev.json', './dataset/frame_info.json', tok)
cfg = BertConfig.from_json_file(args.config_file); cfg.num_labels = 1
m = Model(cfg); m.load_state_dict(torch.load('saves/model_task2_best.bin', map_location='cpu'), strict=False)
m = m.to(dev); m.eval()
ldr = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=partial(cf, dev=dev))
preds = []
with torch.no_grad():
    for ids, am, tgt, sids in tqdm(ldr, desc='Task2 dev'):
        lg = m(input_ids=ids, attention_mask=am, target=tgt, labels=None, device=dev, for_test=True)['logits']
        H_am = torch.triu(torch.matmul(am.unsqueeze(2).float(), am.unsqueeze(1).float()), diagonal=0)
        H_p = torch.where(lg >= 0, torch.ones(lg.shape).to(dev), torch.zeros(lg.shape).to(dev)) * H_am
        for idx in torch.nonzero(H_p):
            b,s,e = idx[0].item(), idx[1].item(), idx[2].item()
            if e < tgt[b][0]: preds.append([sids[b], s-1, e-1])
            elif s > tgt[b][1]: preds.append([sids[b], s-3, e-3])
json.dump(preds, open('dataset/dev_task2_pred.json','w'), ensure_ascii=False)
print(f'Task2 dev: {len(preds)} spans saved')
"

echo ""
echo "=== Task3 E12b dev prediction ==="
python3 -c "
import torch, json, codecs, numpy as np, math, torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer
from params import args
from model_task3 import Model
from frame_roles import build_frame2roles, build_legal_mask

TOPK = 3; ALPHA = 1.0; ILLEGAL = -10.0; FUSION = 'max'

class DS(torch.utils.data.Dataset):
    def __init__(self, jf, lf, t1f, t2f, tok):
        self.tok = tok
        with codecs.open(jf,'r') as f: self.all_data = json.load(f)
        with codecs.open(lf,'r') as f: self.ori_labels = json.load(f)
        with codecs.open(t1f,'r') as f: topk_data = json.load(f)
        self.sent2topk = {it['sentence_id']: it['topk'] for it in topk_data}
        with codecs.open(t2f,'r') as f: self.task2_data = json.load(f)
        self.i2l = []
        for ln in self.ori_labels:
            for fe in ln['fes']:
                if fe['fe_name'] not in self.i2l: self.i2l.append(fe['fe_name'])
        self.l2i = {v:i for i,v in enumerate(self.i2l)}
        self.frame2idx = {}
        for i, ln in enumerate(self.ori_labels): self.frame2idx[ln['frame_name']] = i
        self.frame2roles, _, _ = build_frame2roles(lf, self.l2i)
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
            topk = self.sent2topk.get(sid, [[self.ori_labels[0]['frame_name'], 1.0]])
            self.data.append({'text': info['text'], 'label_idx': lidx, 'sentence_id': sid, 'target': tgt, 'ori_target': [ln[1], ln[2]], 'topk_frames': topk})
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        d = self.data[i]; e = self.tok.encode_plus(list(d['text']))
        ids = e.data['input_ids']; am = e.data['attention_mask']; t = d['target']
        ids = ids[:t[0]]+[1]+ids[t[0]:t[1]+1]+[2]+ids[t[1]+1:]; am = am+[1,1]
        return ids, am, d['label_idx'], d['sentence_id'], d['ori_target'], d['topk_frames']

def cf(data, dev):
    def pad(d, mx, v=0): return d+[v]*(mx-len(d))
    mx = max(len(x[0]) for x in data)
    ids = torch.from_numpy(np.array([pad(d[0],mx,0) for d in data], dtype=np.int64)).to(dev)
    am = torch.from_numpy(np.array([pad(d[1],mx,0) for d in data], dtype=np.int64)).to(dev)
    return ids, am, [d[2] for d in data], [d[3] for d in data], [d[4] for d in data], [d[5] for d in data]

device = torch.device('cuda')
tok = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
ds = DS('./dataset/cfn-dev.json', './dataset/frame_info.json', './dataset/dev_task1_pred_topk.json', './dataset/dev_task2_pred.json', tok)
cfg = BertConfig.from_json_file(args.config_file); cfg.num_labels = len(ds.i2l); cfg.num_frames = len(ds.frame2idx)
m = Model(cfg); m.load_state_dict(torch.load('saves/model_task3_best.bin', map_location='cpu'), strict=False)
m = m.to(device); m.eval()
ldr = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=partial(cf, dev=device))
fr2roles = ds.frame2roles; fr2idx = ds.frame2idx; n_labels = len(ds.i2l)
fallback = ds.ori_labels[0]['frame_name']

# K-round inference
all_round_logits = []; all_sids = []; all_starts = []; all_ends = []
with torch.no_grad():
    for k in range(TOPK):
        r_logits = []; r_sids = []; r_starts = []; r_ends = []
        for ids, am, tgt, sids, ori_tgt, topk_list in tqdm(ldr, desc=f'Round {k+1}/{TOPK}'):
            bfids = []; bprobs = []; bvalid = []
            for topk in topk_list:
                if k < len(topk):
                    fn, fp = topk[k]; fid = fr2idx.get(fn,0); bfids.append(fid); bprobs.append(fp); bvalid.append(True)
                else:
                    fid = fr2idx.get(fallback,0); bfids.append(fid); bprobs.append(1e-6); bvalid.append(False)
            fids_t = torch.tensor(bfids, dtype=torch.long).to(device)
            lg = m(input_ids=ids, attention_mask=am, target=tgt, labels=None, device=device, for_test=True, frame_ids=fids_t)['logits']
            for i in range(lg.size(0)):
                if bvalid[i]: lg[i] = lg[i] + build_legal_mask(bfids[i], fr2roles, n_labels, ILLEGAL).to(device)
                r_logits.append({'logits': lg[i].cpu(), 'prob': bprobs[i], 'valid': bvalid[i]})
                r_sids.append(sids[i]); r_starts.append(ori_tgt[i][0]); r_ends.append(ori_tgt[i][1])
        all_round_logits.append(r_logits)
        if k == 0: all_sids, all_starts, all_ends = r_sids, r_starts, r_ends

# Fusion
print(f'Fusing with {FUSION}...')
preds = []
for i in tqdm(range(len(all_sids)), desc='Fusing'):
    cands = [all_round_logits[k][i] for k in range(TOPK) if all_round_logits[k][i]['valid']]
    if not cands: cands = [all_round_logits[0][i]]
    if FUSION == 'max':
        stacked = torch.stack([c['logits'] + ALPHA*math.log(max(c['prob'],1e-9)) for c in cands], dim=0)
        final, _ = torch.max(stacked, dim=0)
    else:
        final = cands[0]['logits']
    preds.append([all_sids[i], all_starts[i], all_ends[i], ds.i2l[torch.argmax(final).item()]])
json.dump(preds, open('dataset/dev_task3_pred_e12b.json','w'), ensure_ascii=False)
print(f'Task3 E12b dev: {len(preds)} predictions saved')
"

echo ""
echo "=========================================="
echo "  E12b 算分"
echo "=========================================="
python score.py -g ../data/cfn-dataset/cfn-dev.json \
    -1 dataset/dev_task1_pred.json \
    -2 dataset/dev_task2_pred.json \
    -3 dataset/dev_task3_pred_e12b.json

echo ""
echo "=== Baseline (E7) 对比 ==="
python score.py -g ../data/cfn-dataset/cfn-dev.json \
    -3 dataset/dev_task3_pred.json 2>/dev/null | grep -A5 "Task 3"
