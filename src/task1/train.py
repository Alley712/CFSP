#!/usr/bin/python3

import os
from functools import partial
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, BertConfig, AutoTokenizer
from dataset import Dataset
from params import args
from model_task1 import Model


def get_model_input(data, device=None):
    def pad(d, max_len, v=0):
        return d + [v] * (max_len - len(d))

    bs = len(data)
    max_len = max([len(x[0]) for x in data])

    input_ids_list, attention_mask_list, target, labels, sentence_id = [], [], [], [], []
    for d in data:
        input_ids_list.append(pad(d[0], max_len, 0))
        attention_mask_list.append(pad(d[1], max_len, 0))
        target.append(d[2])
        labels.append(d[3])
        sentence_id.append(d[4])

    input_ids = torch.from_numpy(np.array(input_ids_list, dtype=np.int64)).to(device)
    attention_mask = torch.from_numpy(np.array(attention_mask_list, dtype=np.int64)).to(device)
    labels = torch.from_numpy(np.array(labels, dtype=np.int64)).to(device)
    return input_ids, attention_mask, target, labels, sentence_id


def eval_model(model, val_loader, device):
    model.eval()
    correct, total = 0.0, 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, total=len(val_loader), desc='eval'):
            input_ids, attention_mask, target, labels, sentence_id = batch
            output = model(input_ids=input_ids, attention_mask=attention_mask,
                           target=target, labels=labels, device=device, for_test=True)
            pred = torch.argmax(torch.softmax(output["logits"], dim=-1), dim=-1)
            correct += (pred == labels).sum().item()
            total += len(pred)
    return correct / (total + 1e-6)


def train(model, train_loader, val_loader, device, args):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    total_steps = int(len(train_loader) * args.num_train_epochs / args.accumulate_gradients)

    best_acc = 0.0
    global_step = 0

    for i_epoch in range(1, 1 + args.num_train_epochs):
        total_loss = 0.0
        iter_bar = tqdm(train_loader, total=len(train_loader), desc=f'epoch_{i_epoch}')
        model.train()
        for step, batch in enumerate(iter_bar):
            global_step += 1
            input_ids, attention_mask, target, labels, sentence_id = batch
            output = model(input_ids=input_ids, attention_mask=attention_mask,
                           target=target, labels=labels, device=device)
            loss = output['loss']
            total_loss += loss.item()

            if (step + 1) % 100 == 0:
                print(f'loss: {total_loss / (step + 1):.4f}')

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

            if (step + 1) % args.accumulate_gradients == 0:
                lr_this_step = args.lr * warmup_linear(global_step / total_steps, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()

        acc = eval_model(model, val_loader, device)
        if acc > best_acc:
            print(f'saved! new best acc {acc:.4f}, previous {best_acc:.4f}')
            best_acc = acc
            model_to_save = model.module if hasattr(model, 'module') else model
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model_to_save.state_dict(), f'{args.save_dir}/model_task1_best.bin')
        else:
            print(f'current acc: {acc:.4f}')


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return max((x - 1.) / (warmup - 1.), 0)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    train_dataset = Dataset(f"{args.data_dir}/cfn-train.json",
                            f"{args.data_dir}/frame_info.json", tokenizer)
    dev_dataset = Dataset(f"{args.data_dir}/cfn-dev.json",
                          f"{args.data_dir}/frame_info.json", tokenizer)

    config = BertConfig.from_pretrained(args.model_dir)
    config.num_labels = train_dataset.num_labels
    model = Model(config)
    state = torch.load(f'{args.model_dir}/pytorch_model.bin', map_location='cpu')
    model.load_state_dict(state, strict=False)
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=partial(get_model_input, device=device),
                              drop_last=True)
    val_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=partial(get_model_input, device=device),
                            drop_last=False)
    train(model, train_loader, val_loader, device, args)
